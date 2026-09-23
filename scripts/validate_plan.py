"""Validates a plan.json against the required schema and CLAUDE.md's rules.
   Deterministic, no LLM involved: mirrors Sentinelle's validate_submission.py
   but for our plan format instead of a submission file.
"""
import argparse
import json
import sys
from pathlib import Path

ALLOWED_METHODS = {
    "pca", "kernel_pca", "sparse_pca",
    "mds", "isomap", "lle", "laplacian_eigenmaps", "diffusion_maps",
    "tsne", "umap", "gplvm",
}

# CLAUDE.md rule 12: "standard choice"/"default" alone is not an acceptable reason.
LAZY_REASONS = {"", "standard choice", "default", "standard", "n/a", "na"}

# CLAUDE.md guard rail: MDS/Isomap are O(N^2) or worse above this size.
LARGE_N_METHODS = {"mds", "isomap"}
LARGE_N_THRESHOLD = 5000


def _is_lazy_reason(reason):
    return not isinstance(reason, str) or reason.strip().lower() in LAZY_REASONS


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
