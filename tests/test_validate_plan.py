"""Tests for the missing-value guards in scripts/validate_plan.py.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import validate_plan  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def pre(name, **params):
    step = {"step": name, "reason": "specific to this test dataset"}
    if params:
        step["params"] = params
    return step


def make_plan(*steps):
    return {
        "dataset": "t",
        "preprocessing": list(steps),
        "methods": [
            {"name": "pca", "role": "general_purpose", "hyperparameters": {"n_components": 2}, "reason": "specific reason"},
            {"name": "umap", "role": "visualization_only", "hyperparameters": {"random_state": 0}, "reason": "specific reason"},
        ],
        "evaluation": {},
        "seed": 0,
        "revision": 1,
        "critique_applied": None,
    }


CLEAN = {"n_samples": 100, "n_features": 10, "has_missing": False, "n_missing_cells": 0, "missing_fraction": 0.0}


def with_missing(fraction=0.05, all_missing=0):
    p = dict(CLEAN, has_missing=True, missing_fraction=fraction, n_missing_cells=int(fraction * 1000))
    p["n_features_all_missing"] = all_missing
    return p


def check(plan, profile):
    return validate_plan.validate_plan(plan, profile)


def has(messages, text):
    return any(text in m for m in messages)


class TestMissingDataRequiresHandling(unittest.TestCase):
    def test_missing_values_without_impute_is_an_error(self):
        errors, _ = check(make_plan(pre("standardize")), with_missing())
        self.assertTrue(has(errors, "no 'impute' step"))

    def test_impute_satisfies_the_guard(self):
        errors, _ = check(make_plan(pre("impute", strategy="median"), pre("standardize")), with_missing())
        self.assertEqual(errors, [])

    def test_drop_then_impute_satisfies_the_guard_including_all_missing_features(self):
        plan = make_plan(pre("drop_missing_features", max_fraction=0.5), pre("impute"), pre("standardize"))
        errors, _ = check(plan, with_missing(all_missing=2))
        self.assertEqual(errors, [])

    def test_entirely_missing_features_need_drop_missing_features(self):
        errors, _ = check(make_plan(pre("impute")), with_missing(all_missing=2))
        self.assertTrue(has(errors, "entirely missing"))
        self.assertTrue(has(errors, "drop_missing_features"))

    def test_clean_data_needs_no_handling_step(self):
        errors, warnings = check(make_plan(pre("standardize")), CLEAN)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])


class TestOrdering(unittest.TestCase):
    def test_a_step_before_impute_is_an_error_and_is_named(self):
        for early in ("standardize", "log1p", "normalize_total", "select_hvg", "scale_unit_range"):
            args = {"n_top_genes": 5} if early == "select_hvg" else {}
            errors, _ = check(make_plan(pre(early, **args), pre("impute")), with_missing())
            self.assertTrue(has(errors, f"'impute' must come before ['{early}']"), early)

    def test_drop_after_impute_is_an_error(self):
        errors, _ = check(make_plan(pre("impute"), pre("drop_missing_features")), with_missing())
        self.assertTrue(has(errors, "'drop_missing_features' must come before 'impute'"))

    def test_a_step_before_drop_missing_features_is_an_error(self):
        errors, _ = check(make_plan(pre("log1p"), pre("drop_missing_features"), pre("impute")), with_missing())
        self.assertTrue(has(errors, "'drop_missing_features' must come before ['log1p']"))

    def test_ordering_is_enforced_even_without_a_profile(self):
        errors, _ = check(make_plan(pre("standardize"), pre("impute")), None)
        self.assertTrue(has(errors, "'impute' must come before"))


class TestParameters(unittest.TestCase):
    def test_invalid_impute_strategy(self):
        errors, _ = check(make_plan(pre("impute", strategy="knn")), with_missing())
        self.assertTrue(has(errors, "strategy must be one of"))

    def test_fill_value_must_be_a_number(self):
        errors, _ = check(make_plan(pre("impute", strategy="constant", fill_value="zero")), with_missing())
        self.assertTrue(has(errors, "fill_value must be a number"))

    def test_max_fraction_must_be_in_zero_to_one_exclusive_of_one(self):
        for bad in (1.0, 1.5, -0.1, "half", True):
            errors, _ = check(make_plan(pre("drop_missing_features", max_fraction=bad), pre("impute")), with_missing())
            self.assertTrue(has(errors, "max_fraction must be a number in [0, 1)"), bad)
        for good in (0, 0.5, 0.99):
            errors, _ = check(make_plan(pre("drop_missing_features", max_fraction=good), pre("impute")), with_missing())
            self.assertFalse(has(errors, "max_fraction"), good)


class TestWarnings(unittest.TestCase):
    def test_high_missing_fraction_warns_but_is_not_an_error(self):
        errors, warnings = check(make_plan(pre("impute")), with_missing(fraction=0.3))
        self.assertEqual(errors, [])
        self.assertTrue(has(warnings, "30.0% of cells are missing"))

    def test_low_missing_fraction_does_not_warn(self):
        _, warnings = check(make_plan(pre("impute")), with_missing(fraction=0.05))
        self.assertFalse(has(warnings, "of cells are missing"))

    def test_unneeded_impute_is_only_a_warning(self):
        errors, warnings = check(make_plan(pre("impute")), CLEAN)
        self.assertEqual(errors, [])
        self.assertTrue(has(warnings, "unnecessary"))

    def test_absent_profile_warns_that_the_guard_could_not_run(self):
        errors, warnings = check(make_plan(pre("standardize")), None)
        self.assertEqual(errors, [])
        self.assertTrue(has(warnings, "could not check"))


class TestExistingRulesStillApply(unittest.TestCase):
    def test_placeholder_reason_on_impute_is_still_rejected(self):
        plan = make_plan({"step": "impute", "reason": "standard choice"})
        errors, _ = check(plan, with_missing())
        self.assertTrue(has(errors, "placeholder"))

    def test_unknown_step_names_are_still_rejected(self):
        errors, _ = check(make_plan(pre("knn_impute")), with_missing())
        self.assertTrue(has(errors, "not a known preprocessing step"))

    def test_real_committed_plans_still_pass_without_new_warnings(self):
        for name in ("pbmc", "pathmnist"):
            plan_path, profile_path = REPO / "outputs" / name / "plan.json", REPO / "outputs" / name / "profile.json"
            if not (plan_path.exists() and profile_path.exists()):
                self.skipTest(f"{name} artifacts not present")
            errors, warnings = check(json.loads(plan_path.read_text()), json.loads(profile_path.read_text()))
            self.assertEqual(errors, [], name)
            self.assertFalse(has(warnings, "missing"), name)


if __name__ == "__main__":
    unittest.main()
