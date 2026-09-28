"""Tests for scripts/cluster.py (data-derived cluster labels for coloring unlabeled plots).

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import cluster  # noqa: E402


def blobs(k=3, per=100, dim=5, sep=8.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, dim)) * sep
    Z = np.vstack([c + rng.normal(size=(per, dim)) for c in centers])
    truth = np.repeat(np.arange(k), per)
    return Z, truth


class TestChooseAndFit(unittest.TestCase):
    def test_recovers_the_true_number_of_well_separated_blobs(self):
        Z, _ = blobs(k=4)
        labels, info = cluster.choose_and_fit(Z, [2, 8], None, seed=0)
        self.assertEqual(info["k"], 4)
        self.assertFalse(info["k_was_fixed"])
        self.assertEqual(labels.shape, (400,))
        self.assertEqual(sorted(info["silhouette_by_k"]), [str(k) for k in range(2, 9)])

    def test_fixed_k_is_respected_even_if_another_k_scores_higher(self):
        Z, _ = blobs(k=3)
        labels, info = cluster.choose_and_fit(Z, [2, 8], 5, seed=0)
        self.assertEqual(info["k"], 5)
        self.assertTrue(info["k_was_fixed"])
        self.assertEqual(len(np.unique(labels)), 5)
        self.assertEqual(list(info["silhouette_by_k"]), ["5"])

    def test_labels_are_ordered_by_size_and_sizes_sum_to_n(self):
        rng = np.random.default_rng(1)
        Z = np.vstack([rng.normal(size=(30, 3)), rng.normal(size=(200, 3)) + 20, rng.normal(size=(80, 3)) - 20])
        # several seeds: KMeans numbers its clusters arbitrarily, so a single seed could be in size order by luck
        for seed in range(8):
            labels, info = cluster.choose_and_fit(Z, [2, 5], 3, seed=seed)
            self.assertEqual(info["cluster_sizes"], [200, 80, 30], seed)
            self.assertEqual(sum(info["cluster_sizes"]), 310)
            self.assertEqual(int(np.sum(labels == 0)), 200)
            self.assertEqual(int(np.sum(labels == 2)), 30)

    def test_same_seed_gives_identical_labels(self):
        Z, _ = blobs(k=3)
        a, _ = cluster.choose_and_fit(Z, [2, 6], None, seed=7)
        b, _ = cluster.choose_and_fit(Z, [2, 6], None, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_scan_subsamples_the_silhouette_above_the_size_threshold(self):
        orig = cluster.SILHOUETTE_SCAN_SIZE
        cluster.SILHOUETTE_SCAN_SIZE = 150
        try:
            Z, _ = blobs(k=3, per=100)
            labels, info = cluster.choose_and_fit(Z, [2, 5], None, seed=0)
        finally:
            cluster.SILHOUETTE_SCAN_SIZE = orig
        self.assertTrue(info["silhouette_subsampled"])
        self.assertEqual(info["n_scored"], 150)
        self.assertEqual(labels.shape[0], 300)  # KMeans still fit on every row

    def test_no_subsampling_flag_for_small_data(self):
        Z, _ = blobs(k=2, per=50)
        _, info = cluster.choose_and_fit(Z, [2, 4], None, seed=0)
        self.assertFalse(info["silhouette_subsampled"])
        self.assertEqual(info["n_scored"], 100)

    def test_k_range_is_clipped_to_fewer_samples(self):
        Z = np.random.default_rng(0).normal(size=(5, 2))
        _, info = cluster.choose_and_fit(Z, [2, 10], None, seed=0)
        self.assertLessEqual(int(max(info["silhouette_by_k"], key=int)), 4)

    def test_invalid_k_is_an_error(self):
        Z, _ = blobs(k=2, per=10)
        for bad in (1, 20, 0):
            with self.assertRaises(ValueError):
                cluster.choose_and_fit(Z, [2, 5], bad, seed=0)

    def test_metrics_are_json_safe(self):
        Z, _ = blobs(k=3)
        _, info = cluster.choose_and_fit(Z, [2, 5], None, seed=0)
        json.dumps(info, allow_nan=False)


def satellite_scenario(seed=0):
    """A: big blob at the origin. B: a small, real satellite blob close to A -- close enough that
    a coarse k merges it with A even though it's genuinely a separate group. C: a large, obviously
    separate blob far from both. Mirrors the pbmc case that motivated the density cross-check: a
    silhouette-argmax scan alone picks k=2 (A+B merged, C alone) because that split scores highest,
    even though B is real. Returns (Z, viz, truth) with truth in {0: A, 1: B, 2: C}."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(150, 5))
    B = rng.normal(size=(40, 5)) * 0.5 + np.array([6, 0, 0, 0, 0])
    C = rng.normal(size=(60, 5)) + np.array([20, 0, 0, 0, 0])
    Z = np.vstack([A, B, C])
    truth = np.array([0] * 150 + [1] * 40 + [2] * 60)
    return Z, Z[:, :2], truth  # first two dims stand in for a "viz" embedding; the same gap is visible there


class TestKneeEps(unittest.TestCase):
    def test_finds_a_larger_eps_for_widely_separated_blobs_than_for_one_tight_blob(self):
        rng = np.random.default_rng(0)
        tight = rng.normal(size=(200, 2)) * 0.2
        spread_out = np.vstack([rng.normal(size=(100, 2)) * 0.2, rng.normal(size=(100, 2)) * 0.2 + 20])
        eps_tight = cluster._knee_eps(tight, min_samples=10)
        eps_spread = cluster._knee_eps(spread_out, min_samples=10)
        self.assertGreater(eps_tight, 0)
        # the knee sits at the density drop; two far-apart blobs' curve breaks at a much larger gap
        self.assertLess(eps_tight, eps_spread)


class TestDensityReferenceGroups(unittest.TestCase):
    def test_recovers_the_known_groups_and_drops_noise(self):
        rng = np.random.default_rng(0)
        Z = np.vstack([rng.normal(size=(100, 2)), rng.normal(size=(80, 2)) + 15, rng.normal(size=(3, 2)) + 40])
        groups, eps = cluster._density_reference_groups(Z, min_group_size=10)
        self.assertGreater(eps, 0)
        sizes = sorted(int(m.sum()) for m in groups.values())
        self.assertEqual(sizes, [80, 100])  # the 3-point group is below min_group_size, excluded

    def test_min_group_size_excludes_a_group_dbscan_itself_would_still_keep(self):
        # Same data as above, but with min_group_size raised past the 80-point group: DBSCAN's own
        # min_samples doesn't exclude it (it's a real, dense cluster), only the explicit filter does.
        rng = np.random.default_rng(0)
        Z = np.vstack([rng.normal(size=(100, 2)), rng.normal(size=(80, 2)) + 15, rng.normal(size=(3, 2)) + 40])
        groups, _ = cluster._density_reference_groups(Z, min_group_size=90)
        self.assertEqual(sorted(int(m.sum()) for m in groups.values()), [100])

    def test_a_single_uniform_blob_yields_at_most_one_group(self):
        Z = np.random.default_rng(0).normal(size=(200, 2))
        groups, _ = cluster._density_reference_groups(Z, min_group_size=10)
        self.assertLessEqual(len(groups), 1)


class TestGroupsMapToDistinctClusters(unittest.TestCase):
    def test_true_when_every_group_has_its_own_majority_cluster(self):
        groups = {0: np.array([True, True, False, False]), 1: np.array([False, False, True, True])}
        labels = np.array([0, 0, 1, 1])
        distinct, majority = cluster._groups_map_to_distinct_clusters(groups, labels, n_clusters=2)
        self.assertTrue(distinct)
        self.assertEqual(majority, {0: 0, 1: 1})

    def test_false_when_two_groups_share_a_majority_cluster(self):
        groups = {0: np.array([True, True, False, False]), 1: np.array([False, False, True, True])}
        labels = np.array([0, 0, 0, 0])  # both groups end up mostly in cluster 0
        distinct, majority = cluster._groups_map_to_distinct_clusters(groups, labels, n_clusters=1)
        self.assertFalse(distinct)
        self.assertEqual(majority, {0: 0, 1: 0})


class TestSelectK(unittest.TestCase):
    def test_density_cross_check_finds_the_merged_satellite_that_silhouette_alone_misses(self):
        Z, viz, truth = satellite_scenario()
        labels, info = cluster.choose_and_fit(Z, [2, 5], None, seed=0, viz_embedding=viz)
        self.assertEqual(info["k"], 3)  # not 2: silhouette alone favors 2 (A+B merged, see below)
        self.assertEqual(info["k_selection"]["selected_by"], "density_cross_check")
        self.assertEqual(info["k_selection"]["n_significant_groups"], 3)
        # each true group lands in its own cluster
        for t in range(3):
            self.assertEqual(len(np.unique(labels[truth == t])), 1)

    def test_without_a_viz_embedding_the_same_data_falls_back_to_plain_silhouette_and_merges_the_satellite(self):
        Z, _, truth = satellite_scenario()
        labels, info = cluster.choose_and_fit(Z, [2, 5], None, seed=0, viz_embedding=None)
        self.assertEqual(info["k"], 2)  # the failure mode this mechanism exists to catch
        self.assertEqual(info["k_selection"], {"used": False, "selected_by": "silhouette_argmax"})
        self.assertEqual(len(np.unique(labels[truth == 0])) | len(np.unique(labels[truth == 1])), 1)

    def test_no_forced_extra_split_when_there_is_no_real_density_substructure(self):
        # A single blob plus one separate blob: no hidden satellite to find, so the density check
        # (if it finds <2 significant groups on the viz side) must not force k past what fits.
        rng = np.random.default_rng(0)
        Z = np.vstack([rng.normal(size=(150, 5)), rng.normal(size=(150, 5)) + 20])
        viz = Z[:, :2]
        _, info = cluster.choose_and_fit(Z, [2, 6], None, seed=0, viz_embedding=viz)
        self.assertEqual(info["k"], 2)

    def test_never_satisfied_in_range_falls_back_to_silhouette_argmax(self):
        Z, viz, _ = satellite_scenario()
        # range excludes k=3, the only k where the density check passes for this scenario
        labels, info = cluster.choose_and_fit(Z, [2, 2], None, seed=0, viz_embedding=viz)
        self.assertEqual(info["k"], 2)
        self.assertIn(info["k_selection"]["selected_by"], ("density_cross_check_never_satisfied_in_range",))

    def test_fixed_k_skips_the_density_check_entirely(self):
        Z, viz, _ = satellite_scenario()
        _, info = cluster.choose_and_fit(Z, [2, 5], 4, seed=0, viz_embedding=viz)
        self.assertEqual(info["k"], 4)
        self.assertEqual(info["k_selection"], {"selected_by": "fixed"})

    def test_info_is_json_safe_with_a_viz_embedding(self):
        Z, viz, _ = satellite_scenario()
        _, info = cluster.choose_and_fit(Z, [2, 5], None, seed=0, viz_embedding=viz)
        json.dumps(info, allow_nan=False)


class TestRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        out = Path("outputs/d")
        (out / "embeddings").mkdir(parents=True)
        Z, self.truth = blobs(k=3)
        np.save(out / "embeddings" / "pca.npy", Z)
        (out / "embeddings" / "pca.json").write_text(json.dumps({"hyperparameters": {"n_components": 5}, "seed": 0}))
        np.save(out / "embeddings" / "umap.npy", Z[:, :2])
        self.plan = {"seed": 0, "clustering": {"source": "pca", "k_range": [2, 6], "reason": "r"}}

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_writes_labels_and_metrics_and_labels_match_the_blobs(self):
        self.assertEqual(cluster.run("d", self.plan), "ok")
        labels = np.load("outputs/d/clusters.npy")
        self.assertEqual(labels.shape, (300,))
        m = json.loads(Path("outputs/d/metrics/clustering.json").read_text())
        self.assertEqual(m["k"], 3)
        self.assertEqual(m["params"]["source"], "pca")
        # every true blob maps to exactly one cluster
        for t in range(3):
            self.assertEqual(len(np.unique(labels[self.truth == t])), 1)

    def test_second_identical_run_is_reused_and_does_not_rewrite(self):
        cluster.run("d", self.plan)
        mtime = Path("outputs/d/clusters.npy").stat().st_mtime_ns
        self.assertEqual(cluster.run("d", self.plan), "reused")
        self.assertEqual(Path("outputs/d/clusters.npy").stat().st_mtime_ns, mtime)

    def test_changed_parameters_or_changed_source_embedding_are_recomputed(self):
        cluster.run("d", self.plan)
        changed_k = {"seed": 0, "clustering": {"source": "pca", "k": 2, "reason": "r"}}
        self.assertEqual(cluster.run("d", changed_k), "ok")
        self.assertEqual(cluster.run("d", self.plan), "ok")  # back to the scan: params differ from what is stored
        Path("outputs/d/embeddings/pca.json").write_text(json.dumps({"hyperparameters": {"n_components": 9}, "seed": 0}))
        self.assertEqual(cluster.run("d", self.plan), "ok")  # source embedding was regenerated with other settings

    def test_visualization_only_source_is_refused(self):
        plan = {"seed": 0, "clustering": {"source": "umap", "reason": "r"}}
        with self.assertRaisesRegex(ValueError, "never be the input to clustering"):
            cluster.run("d", plan)
        self.assertFalse(Path("outputs/d/clusters.npy").exists())

    def test_local_structure_source_is_refused_too(self):
        plan = {"seed": 0, "clustering": {"source": "lle", "reason": "r"}}
        with self.assertRaisesRegex(ValueError, "general_purpose"):
            cluster.run("d", plan)

    def test_missing_source_embedding_is_a_clear_error(self):
        plan = {"seed": 0, "clustering": {"source": "kernel_pca", "reason": "r"}}
        with self.assertRaisesRegex(FileNotFoundError, "run the plan's methods first"):
            cluster.run("d", plan)

    def test_unknown_algorithm_is_refused(self):
        plan = {"seed": 0, "clustering": {"source": "pca", "algorithm": "dbscan", "reason": "r"}}
        with self.assertRaisesRegex(ValueError, "algorithm must be one of"):
            cluster.run("d", plan)


class TestRunUsesVizEmbeddingWhenDeclared(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        out = Path("outputs/d")
        (out / "embeddings").mkdir(parents=True)
        self.Z, self.viz, self.truth = satellite_scenario()
        np.save(out / "embeddings" / "pca.npy", self.Z)
        (out / "embeddings" / "pca.json").write_text(json.dumps({"hyperparameters": {}, "seed": 0}))
        np.save(out / "embeddings" / "umap.npy", self.viz)
        (out / "embeddings" / "umap.json").write_text(json.dumps({"hyperparameters": {"n_neighbors": 15}, "seed": 0}))
        np.save(out / "embeddings" / "tsne.npy", self.viz)
        (out / "embeddings" / "tsne.json").write_text(json.dumps({"hyperparameters": {}, "seed": 0}))
        self.plan = {
            "seed": 0,
            "methods": [
                {"name": "pca", "role": "general_purpose"},
                {"name": "umap", "role": "visualization_only"},
                {"name": "tsne", "role": "visualization_only"},
            ],
            "clustering": {"source": "pca", "k_range": [2, 5], "reason": "r"},
        }

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_a_umap_embedding_declared_in_methods_is_used_and_finds_the_satellite(self):
        self.assertEqual(cluster.run("d", self.plan), "ok")
        m = json.loads(Path("outputs/d/metrics/clustering.json").read_text())
        self.assertEqual(m["k"], 3)
        self.assertEqual(m["k_selection"]["selected_by"], "density_cross_check")
        self.assertEqual(m["params"]["viz_source"], "umap")  # preferred over tsne per VIZ_PREFERENCE

    def test_umap_is_preferred_over_tsne_when_both_are_present(self):
        cluster.run("d", self.plan)
        m = json.loads(Path("outputs/d/metrics/clustering.json").read_text())
        self.assertEqual(m["params"]["viz_source"], "umap")

    def test_tsne_is_used_when_umap_is_not_in_the_plan(self):
        plan = dict(self.plan, methods=[m for m in self.plan["methods"] if m["name"] != "umap"])
        cluster.run("d", plan)
        m = json.loads(Path("outputs/d/metrics/clustering.json").read_text())
        self.assertEqual(m["params"]["viz_source"], "tsne")

    def test_a_regenerated_viz_embedding_invalidates_the_cache(self):
        cluster.run("d", self.plan)
        (Path("outputs/d/embeddings/umap.json")).write_text(json.dumps({"hyperparameters": {"n_neighbors": 30}, "seed": 0}))
        self.assertEqual(cluster.run("d", self.plan), "ok")  # not "reused": the viz fingerprint changed

    def test_no_visualization_only_method_in_the_plan_falls_back_to_silhouette_only(self):
        plan = dict(self.plan, methods=[m for m in self.plan["methods"] if m["name"] == "pca"])
        cluster.run("d", plan)
        m = json.loads(Path("outputs/d/metrics/clustering.json").read_text())
        self.assertEqual(m["k_selection"]["selected_by"], "silhouette_argmax")
        self.assertIsNone(m["params"]["viz_source"])


if __name__ == "__main__":
    unittest.main()
