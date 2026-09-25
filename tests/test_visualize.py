"""Tests for the coloring logic in scripts/visualize.py.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import visualize  # noqa: E402


def labels(n_classes, per=5):
    return np.repeat(np.arange(n_classes), per)


class TestCategoricalColors(unittest.TestCase):
    def test_up_to_ten_classes_get_ten_distinct_colors_and_no_other(self):
        colors, legend = visualize.categorical_colors(labels(10))
        self.assertEqual(len({tuple(c) for c in colors}), 10)
        self.assertEqual(len(legend), 10)
        self.assertFalse(any("other" in t for t, _ in legend))

    def test_eleven_to_twenty_classes_no_longer_repeat_colors(self):
        for k in (11, 15, 20):
            colors, legend = visualize.categorical_colors(labels(k))
            self.assertEqual(len({tuple(c) for c in colors}), k, k)
            self.assertEqual(len(legend), k)
            self.assertFalse(any("other" in t for t, _ in legend), k)  # nothing is folded up to 20 classes

    def test_more_than_twenty_classes_keep_the_largest_and_fold_the_rest_into_gray_other(self):
        # class i has i+1 members, so classes 5..24 are the 20 largest of 25 and class 24 is biggest
        y = np.concatenate([np.full(i + 1, i) for i in range(25)])
        colors, legend = visualize.categorical_colors(y)
        self.assertEqual(len(legend), 20)
        self.assertEqual(legend[-1][0], "other (6 classes)")
        kept = {t for t, _ in legend[:-1]}
        self.assertIn("24", kept)
        self.assertNotIn("0", kept)  # the smallest class is folded
        other = matplotlib.colors.to_rgba(visualize.OTHER_COLOR)
        self.assertTrue(np.allclose(colors[y == 0][0], other))
        self.assertFalse(np.allclose(colors[y == 24][0], other))
        self.assertEqual(len({tuple(c) for c in colors}), 20)  # 19 kept + gray, no sharing

    def test_string_labels_and_label_fn(self):
        y = np.array(["b", "a", "b", "c"])
        _, legend = visualize.categorical_colors(y, label_fn=lambda c: f"class {c}")
        self.assertEqual([t for t, _ in legend], ["class a", "class b", "class c"])

    def test_same_class_always_gets_the_same_color(self):
        y = np.array([3, 1, 3, 2, 1])
        colors, _ = visualize.categorical_colors(y)
        np.testing.assert_array_equal(colors[0], colors[2])
        np.testing.assert_array_equal(colors[1], colors[4])


class TestMakeScatter(unittest.TestCase):
    def setUp(self):
        self.emb = np.random.default_rng(0).normal(size=(60, 3))

    def tearDown(self):
        plt.close("all")

    def test_density_labels_and_clusters_all_render(self):
        for y, note in ((None, None), (np.repeat([0, 1, 2], 20), "colored by data-derived clusters (k=3)")):
            fig = visualize.make_scatter(self.emb, y, "d", "pca", color_note=note)
            self.assertEqual(len(fig.axes), 1)

    def test_subtitle_carries_the_color_note_and_the_dimension_note(self):
        fig = visualize.make_scatter(self.emb, None, "d", "pca", color_note="colored by X")
        texts = [t.get_text() for t in fig.axes[0].texts]
        self.assertTrue(any("colored by X" in t and "dims 1-2 of 3" in t for t in texts), texts)

    def test_legend_uses_the_label_function(self):
        fig = visualize.make_scatter(self.emb, np.repeat([0, 1], 30), "d", "pca", label_fn=lambda c: f"cluster {int(c) + 1}")
        self.assertEqual([t.get_text() for t in fig.axes[0].get_legend().get_texts()], ["cluster 1", "cluster 2"])

    def test_umap_axes_stay_tickless_with_cluster_colors(self):
        fig = visualize.make_scatter(self.emb[:, :2], np.repeat([0, 1], 30), "d", "umap")
        self.assertEqual(list(fig.axes[0].get_xticks()), [])


class TestLoadClusterColoring(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.out = Path("outputs/d")
        (self.out / "metrics").mkdir(parents=True)
        np.save(self.out / "clusters.npy", np.repeat([0, 1, 2], 10))
        (self.out / "metrics" / "clustering.json").write_text(json.dumps({"k": 3, "params": {"source": "pca"}}))
        self.write_plan(True)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def write_plan(self, with_block):
        plan = {"clustering": {"source": "pca"}} if with_block else {}
        (self.out / "plan.json").write_text(json.dumps(plan))

    def test_loads_labels_and_a_note_naming_source_and_k(self):
        labels, note = visualize.load_cluster_coloring("d", 30)
        self.assertEqual(labels.shape, (30,))
        self.assertIn("data-derived clusters", note)
        self.assertIn("pca", note)
        self.assertIn("k=3", note)

    def test_a_leftover_file_is_ignored_when_the_plan_has_no_clustering_block(self):
        self.write_plan(False)
        self.assertEqual(visualize.load_cluster_coloring("d", 30), (None, None))

    def test_a_size_mismatch_is_ignored(self):
        self.assertEqual(visualize.load_cluster_coloring("d", 31), (None, None))

    def test_missing_files_give_none(self):
        (self.out / "clusters.npy").unlink()
        self.assertEqual(visualize.load_cluster_coloring("d", 30), (None, None))


if __name__ == "__main__":
    unittest.main()
