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


if __name__ == "__main__":
    unittest.main()
