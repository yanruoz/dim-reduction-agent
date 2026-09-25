"""Tests for the label-silhouette sanity check in scripts/evaluate.py.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import evaluate  # noqa: E402


def blobs(n_per=40, k=3, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(size=(n_per, 2)) + 6 * i for i in range(k)])
    return Z, np.repeat(np.arange(k), n_per)


class TestLabelSanityCheck(unittest.TestCase):
    def test_small_data_is_scored_in_full_and_matches_sklearn(self):
        from sklearn.metrics import silhouette_score

        Z, y = blobs()
        score, subsampled, n_used = evaluate.compute_label_sanity_check(Z, y, seed=0)
        self.assertFalse(subsampled)
        self.assertEqual(n_used, len(y))
        self.assertAlmostEqual(score, float(silhouette_score(Z, y)))

    def test_no_labels_or_one_class_gives_nothing_to_score(self):
        Z, y = blobs()
        self.assertEqual(evaluate.compute_label_sanity_check(Z, None, 0), (None, False, 0))
        self.assertEqual(evaluate.compute_label_sanity_check(Z, np.zeros(len(y), int), 0), (None, False, 0))

    def test_large_data_is_subsampled_and_reports_the_size_used(self):
        orig = (evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE)
        evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE = 100, 60
        try:
            Z, y = blobs(n_per=80)  # 240 > 100
            score, subsampled, n_used = evaluate.compute_label_sanity_check(Z, y, seed=0)
            again = evaluate.compute_label_sanity_check(Z, y, seed=0)
        finally:
            evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE = orig
        self.assertTrue(subsampled)
        self.assertEqual(n_used, 60)
        self.assertGreater(score, 0.5)  # well separated blobs stay well separated in a subsample
        self.assertEqual((score, subsampled, n_used), again)  # fixed seed, reproducible

    def test_a_subsample_that_loses_all_but_one_class_is_not_scored(self):
        orig = (evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE)
        evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE = 50, 5
        try:
            Z = np.random.default_rng(0).normal(size=(200, 2))
            y = np.zeros(200, int)
            y[0] = 1  # a single member of the second class; a 5-row draw almost surely misses it
            score, subsampled, _ = evaluate.compute_label_sanity_check(Z, y, seed=3)
        finally:
            evaluate.TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD, evaluate.TRUSTWORTHINESS_SUBSAMPLE_SIZE = orig
        self.assertIsNone(score)
        self.assertTrue(subsampled)


if __name__ == "__main__":
    unittest.main()
