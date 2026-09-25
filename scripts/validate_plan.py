"""Validates a plan.json against the required schema and CLAUDE.md's rules.
   Deterministic, no LLM involved: mirrors Sentinelle's validate_submission.py
   but for our plan format instead of a submission file.
"""
import argparse
import json
import sys
from pathlib import Path

from plan_schema import (
    ALLOWED_METHODS,
    ALLOWED_ROLES,
    CLUSTERING_ALGORITHMS,
    IMPUTE_STRATEGIES,
    KNOWN_PREPROCESSING_STEPS,
    METHOD_ROLES,
    MISSING_VALUE_STEPS,
)

# CLAUDE.md rule 12: "standard choice"/"default" alone is not an acceptable reason.
LAZY_REASONS = {"", "standard choice", "default", "standard", "n/a", "na"}

# CLAUDE.md guard rail: MDS/Isomap/Kernel PCA/Diffusion Maps are all O(N^2) or
# worse above this size (Kernel PCA's RBF kernel and this project's diffusion
# maps implementation both build an N x N distance/affinity matrix).
LARGE_N_METHODS = {"mds", "isomap", "kernel_pca", "diffusion_maps"}
LARGE_N_THRESHOLD = 5000


def _is_lazy_reason(reason):
    return not isinstance(reason, str) or reason.strip().lower() in LAZY_REASONS


# Above this share of missing cells, imputed values make up a large part of what every
# method sees; not an error, but the plan's reasons should say why proceeding is sound.
HIGH_MISSING_FRACTION = 0.2


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_missing_value_handling(plan, profile, errors, warnings):
    """Mechanical guard: data with missing values can't reach a method unhandled,
    and the steps that handle them must run first and in the right order."""
    steps = [p.get("step") for p in plan.get("preprocessing", []) if isinstance(p, dict)]
    by_name = {p.get("step"): p for p in plan.get("preprocessing", []) if isinstance(p, dict)}

    # Parameter sanity for the two steps, independent of the data.
    if "impute" in by_name:
        params = by_name["impute"].get("params") or {}
        strategy = params.get("strategy", "median")
        if strategy not in IMPUTE_STRATEGIES:
            errors.append(f"preprocessing 'impute': strategy must be one of {sorted(IMPUTE_STRATEGIES)}, got {strategy!r}")
        if "fill_value" in params and not _is_number(params["fill_value"]):
            errors.append("preprocessing 'impute': fill_value must be a number")
    if "drop_missing_features" in by_name:
        max_fraction = (by_name["drop_missing_features"].get("params") or {}).get("max_fraction", 0.5)
        if not _is_number(max_fraction) or not (0 <= max_fraction < 1):
            errors.append(f"preprocessing 'drop_missing_features': max_fraction must be a number in [0, 1), got {max_fraction!r}")

    # Order: whatever handles missing values must come before every other step (they
    # either propagate NaN or, like select_top_variance's variance ranking, misbehave on it),
    # and dropping mostly-missing features must precede imputing what's left.
    if "impute" in steps:
        i = steps.index("impute")
        offenders = [s for s in steps[:i] if s not in MISSING_VALUE_STEPS]
        if offenders:
            errors.append(f"'impute' must come before {offenders}: those steps cannot run on missing values")
        if "drop_missing_features" in steps and steps.index("drop_missing_features") > i:
            errors.append("'drop_missing_features' must come before 'impute' (impute refuses entirely-missing features)")
    if "drop_missing_features" in steps:
        i = steps.index("drop_missing_features")
        offenders = [s for s in steps[:i] if s not in MISSING_VALUE_STEPS]
        if offenders:
            errors.append(f"'drop_missing_features' must come before {offenders}")

    if profile is None:
        warnings.append("No profile.json next to the plan, so the missing-value guard could not check it against the data "
                        "(preprocessing still refuses NaN at run time).")
        return

    if profile.get("has_missing"):
        if "impute" not in steps:
            errors.append(
                f"profile.json shows {profile.get('n_missing_cells', 'some')} missing cell(s) "
                f"({profile.get('missing_fraction', 0) * 100:.2f}% of the data) but the plan has no 'impute' step; "
                "no method can run on missing values."
            )
        if profile.get("n_features_all_missing", 0) > 0 and "drop_missing_features" not in steps:
            errors.append(
                f"profile.json shows {profile['n_features_all_missing']} entirely missing feature(s), which "
                "'impute' cannot fill; add 'drop_missing_features' before 'impute'."
            )
        if profile.get("missing_fraction", 0) > HIGH_MISSING_FRACTION:
            warnings.append(
                f"{profile['missing_fraction'] * 100:.1f}% of cells are missing; imputed values will make up a large "
                "share of what every method sees. The plan's reasons should say why proceeding is still sound."
            )
    elif "impute" in steps:
        warnings.append("profile.json shows no missing values, so the 'impute' step is unnecessary.")


CLUSTERING_KEYS = {"source", "algorithm", "k_range", "k", "reason"}


def _check_clustering(plan, profile, errors, warnings):
    """Optional `clustering` block: data-derived labels used only to color plots. The source must be a
    general_purpose method that is in this plan, so a visualization_only embedding can never feed it."""
    block = plan.get("clustering")
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append("'clustering' must be a dict")
        return

    unknown = sorted(set(block) - CLUSTERING_KEYS)
    if unknown:
        errors.append(f"clustering: unknown key(s) {unknown}; allowed keys are {sorted(CLUSTERING_KEYS)}")

    if _is_lazy_reason(block.get("reason")):
        errors.append("clustering: 'reason' is missing or a placeholder like 'standard choice'/'default'")

    algorithm = block.get("algorithm", "kmeans")
    if algorithm not in CLUSTERING_ALGORITHMS:
        errors.append(f"clustering: algorithm must be one of {sorted(CLUSTERING_ALGORITHMS)}, got {algorithm!r}")

    source = block.get("source")
    methods = {m.get("name"): m for m in plan.get("methods", []) if isinstance(m, dict)}
    if source not in methods:
        errors.append(f"clustering: source {source!r} is not a method in this plan (methods: {sorted(n for n in methods if n)})")
    elif METHOD_ROLES.get(source) != "general_purpose":
        errors.append(
            f"clustering: source {source!r} has role {METHOD_ROLES.get(source)!r}; clustering may only use a "
            "general_purpose embedding (a visualization_only or local-structure embedding must never feed it)"
        )

    n = (profile or {}).get("n_samples")
    if "k" in block and block["k"] is not None:
        k = block["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 2:
            errors.append(f"clustering: k must be an integer >= 2, got {k!r}")
        elif n is not None and k >= n:
            errors.append(f"clustering: k ({k}) must be smaller than n_samples ({n})")
    if "k_range" in block:
        kr = block["k_range"]
        ok = (
            isinstance(kr, list) and len(kr) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in kr)
            and 2 <= kr[0] <= kr[1]
        )
        if not ok:
            errors.append(f"clustering: k_range must be [low, high] integers with 2 <= low <= high, got {kr!r}")
        elif n is not None and kr[1] >= n:
            warnings.append(f"clustering: k_range upper bound ({kr[1]}) is not below n_samples ({n}); it will be clipped")

    if profile is not None and profile.get("has_labels"):
        warnings.append(
            "clustering: the dataset has ground-truth labels, so plots are colored by those labels and the "
            "data-derived clusters go unused; drop the clustering block unless you want them for comparison"
        )


def validate_plan(plan, profile=None):
    """Returns (errors, warnings); both lists of strings. Empty errors = pass."""
    errors = []
    warnings = []

    required_keys = ["dataset", "preprocessing", "methods", "evaluation", "seed", "revision", "critique_applied"]
    for key in required_keys:
        if key not in plan:
            errors.append(f"Missing required top-level key: '{key}'")
    if errors:
        return errors, warnings  # can't check structure further without the basics

    if not isinstance(plan["dataset"], str) or not plan["dataset"].strip():
        errors.append("'dataset' must be a non-empty string")

    if not isinstance(plan["seed"], int):
        errors.append("'seed' must be an integer")

    if not isinstance(plan["revision"], int) or plan["revision"] < 1:
        errors.append("'revision' must be an integer >= 1")

    if not isinstance(plan["preprocessing"], list):
        errors.append("'preprocessing' must be a list")
    else:
        for i, step in enumerate(plan["preprocessing"]):
            if not isinstance(step.get("step"), str) or not step["step"].strip():
                errors.append(f"preprocessing[{i}]: missing or empty 'step'")
            elif step["step"] not in KNOWN_PREPROCESSING_STEPS:
                errors.append(
                    f"preprocessing[{i}]: '{step['step']}' is not a known preprocessing step "
                    f"({sorted(KNOWN_PREPROCESSING_STEPS)})"
                )
            if _is_lazy_reason(step.get("reason")):
                errors.append(
                    f"preprocessing[{i}] ('{step.get('step')}'): 'reason' is missing or a placeholder "
                    f"like 'standard choice'/'default'"
                )

    if not isinstance(plan["methods"], list) or not plan["methods"]:
        errors.append("'methods' must be a non-empty list")
    else:
        method_names = []
        for i, method in enumerate(plan["methods"]):
            name = method.get("name")
            method_names.append(name)

            if name not in ALLOWED_METHODS:
                errors.append(f"methods[{i}]: '{name}' is not in the approved method list ({sorted(ALLOWED_METHODS)})")

            role = method.get("role")
            if role not in ALLOWED_ROLES:
                errors.append(f"methods[{i}] ('{name}'): 'role' must be one of {sorted(ALLOWED_ROLES)}, got {role!r}")
            elif name in METHOD_ROLES and role != METHOD_ROLES[name]:
                errors.append(
                    f"methods[{i}] ('{name}'): 'role' is {role!r} but '{name}' is always "
                    f"{METHOD_ROLES[name]!r} (see plan_schema.METHOD_ROLES); a visualization-only "
                    f"embedding must never be declared general_purpose or vice versa"
                )

            hp = method.get("hyperparameters")
            if not isinstance(hp, dict):
                errors.append(f"methods[{i}] ('{name}'): 'hyperparameters' must be a dict")
                hp = {}

            if _is_lazy_reason(method.get("reason")):
                errors.append(
                    f"methods[{i}] ('{name}'): 'reason' is missing or a placeholder "
                    f"like 'standard choice'/'default'"
                )

            if "n_components" in hp:
                nc = hp["n_components"]
                if not isinstance(nc, int) or nc < 1:
                    errors.append(f"methods[{i}] ('{name}'): hyperparameters.n_components must be a positive integer")
                elif profile is not None:
                    n_max = min(profile.get("n_samples", nc), profile.get("n_features", nc))
                    if nc > n_max:
                        errors.append(
                            f"methods[{i}] ('{name}'): n_components ({nc}) exceeds "
                            f"min(n_samples, n_features)={n_max} from profile.json"
                        )

            if "random_state" in hp and hp["random_state"] != plan.get("seed"):
                warnings.append(
                    f"methods[{i}] ('{name}'): hyperparameters.random_state ({hp['random_state']}) "
                    f"does not match the plan's top-level seed ({plan.get('seed')})"
                )

            if name in LARGE_N_METHODS and profile is not None and profile.get("n_samples", 0) > LARGE_N_THRESHOLD:
                has_subsample_step = any(
                    "subsample" in (p.get("step") or "").lower() for p in plan.get("preprocessing", [])
                )
                if not has_subsample_step:
                    warnings.append(
                        f"methods[{i}] ('{name}'): n_samples ({profile['n_samples']}) exceeds "
                        f"{LARGE_N_THRESHOLD} and no subsampling preprocessing step is documented; "
                        f"'{name}' is O(N^2) or worse"
                    )

        if "pca" not in method_names:
            warnings.append("No 'pca' method present; guidance is to always include PCA as a baseline")

    if not isinstance(plan["evaluation"], dict):
        errors.append("'evaluation' must be a dict")

    if isinstance(plan["preprocessing"], list):
        _check_missing_value_handling(plan, profile, errors, warnings)
    _check_clustering(plan, profile, errors, warnings)

    return errors, warnings


def write_reports(errors, warnings, plan_path):
    plan_path = Path(plan_path)
    report = {
        "plan": str(plan_path),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    json_path = plan_path.with_name("plan_validation.json")
    md_path = plan_path.with_name("plan_validation.md")

    json_path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [f"# Plan validation: {plan_path}", "", f"**Result:** {'PASS' if not errors else 'FAIL'}", ""]
    if errors:
        lines.append("## Errors")
        lines += [f"- {e}" for e in errors]
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines += [f"- {w}" for w in warnings]
        lines.append("")
    if not errors and not warnings:
        lines.append("No issues found.")
    md_path.write_text("\n".join(lines) + "\n")

    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a plan.json against the required schema and rules.")
    parser.add_argument("plan", help="Path to plan.json")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())

    # If a profile.json sits next to the plan, use it for range cross-checks.
    profile_path = plan_path.with_name("profile.json")
    profile = json.loads(profile_path.read_text()) if profile_path.exists() else None

    errors, warnings = validate_plan(plan, profile)
    json_path, md_path = write_reports(errors, warnings, plan_path)

    print(f"{'PASS' if not errors else 'FAIL'}: {plan_path}")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARNING: {w}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    sys.exit(0 if not errors else 1)
