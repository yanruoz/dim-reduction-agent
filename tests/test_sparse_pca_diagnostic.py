"""Tests for the QR-subspace variance-explained diagnostic in reduce_dim._run_sparse_pca.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import reduce_dim  # noqa: E402


def blobs(n=300, d=10, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 6
    return np.vstack([c + rng.normal(size=(n // k, d)) for c in centers])


class TestSparsePCADiagnostic(unittest.TestCase):
    def setUp(self):
        self.X = blobs()

    def test_returns_an_embedding_of_the_requested_shape(self):
        embedding, extra = reduce_dim._run_sparse_pca(self.X, {"n_components": 4}, seed=0)
        self.assertEqual(embedding.shape, (self.X.shape[0], 4))

    def test_diagnostic_fields_are_present_and_shaped_like_pcas(self):
        _, extra = reduce_dim._run_sparse_pca(self.X, {"n_components": 4}, seed=0)
        for key in (
            "explained_variance_ratio", "explained_variance_ratio_diagnostic",
            "variance_explained_by_used_components", "diagnostic_fit_components",
            "components_needed_for_90pct_variance", "components_needed_for_90pct_variance_is_lower_bound",
            "variance_diagnostic_is_subspace_based",
        ):
            self.assertIn(key, extra, key)
        self.assertEqual(len(extra["explained_variance_ratio"]), 4)
        self.assertEqual(extra["diagnostic_fit_components"], 4)  # no deeper fit, unlike PCA
        self.assertTrue(extra["variance_diagnostic_is_subspace_based"])

    def test_evr_entries_are_non_negative_fractions_and_sum_matches_reported_total(self):
        _, extra = reduce_dim._run_sparse_pca(self.X, {"n_components": 5}, seed=0)
        evr = np.array(extra["explained_variance_ratio"])
        self.assertTrue((evr >= -1e-9).all(), evr)  # numerically nonnegative
        self.assertLessEqual(evr.sum(), 1.0 + 1e-6)
        self.assertAlmostEqual(extra["variance_explained_by_used_components"], round(float(evr.sum()), 4), places=4)

    def test_more_components_capture_at_least_as_much_variance(self):
        _, small = reduce_dim._run_sparse_pca(self.X, {"n_components": 2}, seed=0)
        _, big = reduce_dim._run_sparse_pca(self.X, {"n_components": 8}, seed=0)
        # not a strict per-component ordering guarantee, but the larger subspace can't capture less
        self.assertGreaterEqual(big["variance_explained_by_used_components"], small["variance_explained_by_used_components"] - 1e-6)

    def test_90pct_lower_bound_flag_is_consistent_with_the_cumulative_curve(self):
        _, extra = reduce_dim._run_sparse_pca(self.X, {"n_components": 3}, seed=0)
        cum = np.cumsum(extra["explained_variance_ratio"])
        if extra["components_needed_for_90pct_variance_is_lower_bound"]:
            self.assertLess(cum[-1], 0.90)
            self.assertEqual(extra["components_needed_for_90pct_variance"], len(cum))
        else:
            k = extra["components_needed_for_90pct_variance"]
            self.assertGreaterEqual(cum[k - 1], 0.90)
            if k > 1:
                self.assertLess(cum[k - 2], 0.90)

    def test_result_is_deterministic_for_a_fixed_seed(self):
        _, a = reduce_dim._run_sparse_pca(self.X, {"n_components": 4}, seed=7)
        _, b = reduce_dim._run_sparse_pca(self.X, {"n_components": 4}, seed=7)
        np.testing.assert_allclose(a["explained_variance_ratio"], b["explained_variance_ratio"])

    def test_default_n_components_is_two(self):
        embedding, extra = reduce_dim._run_sparse_pca(self.X, {}, seed=0)
        self.assertEqual(embedding.shape[1], 2)
        self.assertEqual(len(extra["explained_variance_ratio"]), 2)

    def test_variance_explained_is_invariant_to_a_constant_shift(self):
        # Regression guard: if the data isn't centered before computing variance, adding a large
        # constant offset would swamp total_var and change every EVR value.
        _, base = reduce_dim._run_sparse_pca(self.X, {"n_components": 4}, seed=0)
        shifted = self.X + 1000.0
        _, shifted_extra = reduce_dim._run_sparse_pca(shifted, {"n_components": 4}, seed=0)
        np.testing.assert_allclose(base["explained_variance_ratio"], shifted_extra["explained_variance_ratio"], rtol=1e-4)

    def test_a_subspace_containing_all_the_variance_reports_the_full_total(self):
        # n_components == n_features: components_.T is square and (generically) full rank, so QR
        # spans the whole feature space and must capture ~100% of the variance.
        X = blobs(d=4)
        _, extra = reduce_dim._run_sparse_pca(X, {"n_components": 4}, seed=0)
        self.assertAlmostEqual(extra["variance_explained_by_used_components"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
