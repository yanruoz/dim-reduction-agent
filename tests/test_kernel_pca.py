"""Tests for reduce_dim._run_kernel_pca's gamma hyperparameter.

Run from the repo root:  venv/bin/python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import reduce_dim  # noqa: E402


def blobs(n=200, d=8, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 5
    return np.vstack([c + rng.normal(size=(n // k, d)) for c in centers])


class TestKernelPCAGamma(unittest.TestCase):
    def setUp(self):
        self.X = blobs()

    def test_default_gamma_is_none_same_as_before_this_hyperparameter_existed(self):
        embedding, extra = reduce_dim._run_kernel_pca(self.X, {"n_components": 3}, seed=0)
        self.assertEqual(embedding.shape, (self.X.shape[0], 3))
        self.assertEqual(extra, {})

    def test_an_explicit_gamma_is_passed_through_to_the_model(self):
        from unittest.mock import patch

        with patch("sklearn.decomposition.KernelPCA") as MockKPCA:
            MockKPCA.return_value.fit_transform.return_value = np.zeros((self.X.shape[0], 2))
            reduce_dim._run_kernel_pca(self.X, {"n_components": 2, "gamma": 0.05}, seed=0)
        self.assertEqual(MockKPCA.call_args.kwargs["gamma"], 0.05)

    def test_omitted_gamma_is_passed_as_none_not_dropped(self):
        from unittest.mock import patch

        with patch("sklearn.decomposition.KernelPCA") as MockKPCA:
            MockKPCA.return_value.fit_transform.return_value = np.zeros((self.X.shape[0], 2))
            reduce_dim._run_kernel_pca(self.X, {"n_components": 2}, seed=0)
        self.assertIn("gamma", MockKPCA.call_args.kwargs)
        self.assertIsNone(MockKPCA.call_args.kwargs["gamma"])

    def test_a_larger_gamma_changes_the_embedding_relative_to_the_default(self):
        # sanity check against the real sklearn implementation, not a mock: a much larger gamma
        # (tighter RBF kernel) must produce a numerically different embedding than the default.
        default_embedding, _ = reduce_dim._run_kernel_pca(self.X, {"n_components": 3}, seed=0)
        tuned_embedding, _ = reduce_dim._run_kernel_pca(self.X, {"n_components": 3, "gamma": 5.0}, seed=0)
        self.assertFalse(np.allclose(default_embedding, tuned_embedding))

    def test_result_is_deterministic_for_a_fixed_seed_and_gamma(self):
        a, _ = reduce_dim._run_kernel_pca(self.X, {"n_components": 3, "gamma": 0.1}, seed=3)
        b, _ = reduce_dim._run_kernel_pca(self.X, {"n_components": 3, "gamma": 0.1}, seed=3)
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
