"""Tests for the missing-value preprocessing steps in scripts/reduce_dim.py.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import plan_schema  # noqa: E402
import reduce_dim  # noqa: E402


def step(name, **params):
    return {"step": name, "params": params, "reason": "test"}


class TestImpute(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, np.nan], [3.0, 10.0], [np.nan, 20.0], [5.0, 30.0]])

    def test_median_is_per_feature_over_observed_values(self):
        out = reduce_dim._impute(self.X, {"strategy": "median"})
        np.testing.assert_array_equal(out, [[1, 20], [3, 10], [3, 20], [5, 30]])

    def test_mean_strategy(self):
        out = reduce_dim._impute(self.X, {"strategy": "mean"})
        self.assertAlmostEqual(out[2, 0], 3.0)  # mean(1,3,5)
        self.assertAlmostEqual(out[0, 1], 20.0)  # mean(10,20,30)

    def test_constant_strategy_uses_fill_value_and_defaults_to_zero(self):
        self.assertEqual(reduce_dim._impute(self.X, {"strategy": "constant", "fill_value": -1})[0, 1], -1)
        self.assertEqual(reduce_dim._impute(self.X, {"strategy": "constant"})[0, 1], 0)

    def test_default_strategy_is_median(self):
        np.testing.assert_array_equal(reduce_dim._impute(self.X, {}), reduce_dim._impute(self.X, {"strategy": "median"}))

    def test_observed_values_are_untouched_and_input_is_not_mutated(self):
        original = self.X.copy()
        out = reduce_dim._impute(self.X, {})
        observed = ~np.isnan(original)
        np.testing.assert_array_equal(out[observed], original[observed])
        np.testing.assert_array_equal(np.isnan(self.X), np.isnan(original))  # input still has its NaNs

    def test_result_has_no_nan(self):
        self.assertFalse(np.isnan(reduce_dim._impute(self.X, {})).any())

    def test_no_missing_values_is_a_no_op(self):
        clean = np.arange(6.0).reshape(3, 2)
        np.testing.assert_array_equal(reduce_dim._impute(clean, {}), clean)

    def test_entirely_missing_feature_is_refused_not_invented(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan]])
        with self.assertRaisesRegex(ValueError, "entirely missing.*drop_missing_features"):
            reduce_dim._impute(X, {})

    def test_unknown_strategy_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "strategy must be one of"):
            reduce_dim._impute(self.X, {"strategy": "knn"})


class TestDropMissingFeatures(unittest.TestCase):
    def setUp(self):
        # feature missing fractions: 0, 0.25, 0.5, 0.75, 1.0
        nan = np.nan
        self.X = np.array(
            [[1, 1, 1, 1, nan],
             [2, 2, 2, nan, nan],
             [3, 3, nan, nan, nan],
             [4, nan, nan, nan, nan]],
            dtype=float,
        )

    def test_drops_only_features_above_the_threshold(self):
        out = reduce_dim._drop_missing_features(self.X, {"max_fraction": 0.5})
        self.assertEqual(out.shape, (4, 3))  # keeps the 0, 0.25, 0.5 features

    def test_threshold_is_inclusive_of_equal(self):
        self.assertEqual(reduce_dim._drop_missing_features(self.X, {"max_fraction": 0.25}).shape[1], 2)

    def test_default_threshold_is_half(self):
        self.assertEqual(reduce_dim._drop_missing_features(self.X, {}).shape[1], 3)

    def test_entirely_missing_features_are_dropped_even_at_permissive_thresholds(self):
        out = reduce_dim._drop_missing_features(self.X, {"max_fraction": 0.99})
        self.assertEqual(out.shape[1], 4)
        self.assertFalse(np.isnan(out[:, 0]).any())

    def test_entirely_missing_features_are_dropped_even_when_the_threshold_allows_everything(self):
        # at max_fraction >= 1.0 the fraction test alone would keep a 100%-missing feature,
        # which impute could then never fill; the explicit guard must still remove it
        for threshold in (1.0, 5.0):
            out = reduce_dim._drop_missing_features(self.X, {"max_fraction": threshold})
            self.assertEqual(out.shape[1], 4, threshold)
            self.assertFalse(np.isnan(out).all(axis=0).any(), threshold)

    def test_surviving_columns_keep_their_original_order_and_values(self):
        out = reduce_dim._drop_missing_features(self.X, {"max_fraction": 0.5})
        np.testing.assert_array_equal(out[:, 0], self.X[:, 0])
        np.testing.assert_array_equal(out[:, 2], self.X[:, 2])

    def test_refuses_to_remove_every_feature(self):
        X = np.full((3, 2), np.nan)
        with self.assertRaisesRegex(ValueError, "remove every feature"):
            reduce_dim._drop_missing_features(X, {})

    def test_fully_observed_data_is_unchanged(self):
        clean = np.ones((3, 4))
        np.testing.assert_array_equal(reduce_dim._drop_missing_features(clean, {}), clean)


class TestApplyPreprocessing(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(60, 6))
        self.X[rng.random(self.X.shape) < 0.1] = np.nan
        self.X[:, 5] = np.where(np.arange(60) < 50, np.nan, 1.0)  # feature 5 is ~83% missing

    def test_backstop_refuses_nan_when_the_plan_does_not_handle_it(self):
        with self.assertRaisesRegex(ValueError, "non-finite.*impute"):
            reduce_dim.apply_preprocessing(self.X, [step("standardize")])

    def test_backstop_also_catches_inf_from_a_step(self):
        # log1p of a value below -1 is nan/-inf: caught rather than passed to a method
        with self.assertRaisesRegex(ValueError, "non-finite"):
            reduce_dim.apply_preprocessing(np.array([[-5.0, 1.0], [2.0, 3.0]]), [step("log1p")])

    def test_drop_then_impute_then_standardize_gives_finite_standardized_data(self):
        out = reduce_dim.apply_preprocessing(
            self.X, [step("drop_missing_features", max_fraction=0.5), step("impute"), step("standardize")]
        )
        self.assertEqual(out.shape, (60, 5))  # the ~83%-missing feature is gone
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-9)
        np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-9)

    def test_order_matters_and_impute_first_differs_from_standardize_first(self):
        # standardizing before imputing leaves NaN behind, which the backstop must catch
        with self.assertRaises(ValueError):
            reduce_dim.apply_preprocessing(self.X, [step("standardize"), step("impute")])

    def test_data_with_no_missing_values_passes_through_a_plan_without_impute(self):
        clean = np.random.default_rng(1).normal(size=(20, 4))
        out = reduce_dim.apply_preprocessing(clean, [step("standardize")])
        self.assertTrue(np.isfinite(out).all())

    def test_the_preprocessed_result_can_feed_a_real_method(self):
        out = reduce_dim.apply_preprocessing(
            self.X, [step("drop_missing_features", max_fraction=0.5), step("impute"), step("standardize")]
        )
        embedding, extra = reduce_dim._run_pca(out, {"n_components": 3}, seed=0)
        self.assertEqual(embedding.shape, (60, 3))
        self.assertTrue(np.isfinite(embedding).all())

    def test_imputation_is_deterministic(self):
        plan = [step("drop_missing_features"), step("impute"), step("standardize")]
        a = reduce_dim.apply_preprocessing(self.X, plan)
        b = reduce_dim.apply_preprocessing(self.X, plan)
        np.testing.assert_array_equal(a, b)


class TestRegistry(unittest.TestCase):
    def test_registry_and_schema_agree_and_new_steps_are_known(self):
        self.assertEqual(set(reduce_dim.PREPROCESSORS), plan_schema.KNOWN_PREPROCESSING_STEPS)
        self.assertLessEqual(set(plan_schema.MISSING_VALUE_STEPS), plan_schema.KNOWN_PREPROCESSING_STEPS)


if __name__ == "__main__":
    unittest.main()
