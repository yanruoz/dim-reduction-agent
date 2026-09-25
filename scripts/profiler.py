"""Dataset profiler. Fully generic: takes the common (X, y, metadata)
   interface from loaders.py and computes summary statistics. This file
   should never contain dataset-specific branching.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from loaders import load_dataset


def profile_dataset(X, y=None, metadata=None):
    n_samples, n_features = X.shape

    sparsity = float(np.mean(X == 0))
    has_missing = bool(np.isnan(X).any())

    profile = {
        "n_samples": n_samples,
        "n_features": n_features,
        "sparsity": round(sparsity, 4),
        "has_missing": has_missing,
        "value_range": [float(np.min(X)), float(np.max(X))],
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "has_labels": y is not None,
    }

    if y is not None:
        profile["n_classes"] = int(len(np.unique(y)))

    if metadata:
        profile["metadata"] = metadata

    return profile


def print_profile(name, profile):
    print(f"\n{'='*50}")
    print(f"  Dataset: {name}")
    print(f"{'='*50}")
    print(f"  Samples:        {profile['n_samples']:,}")
    print(f"  Features:       {profile['n_features']:,}")
    print(f"  Sparsity:       {profile['sparsity']*100:.1f}% zeros")
    print(f"  Missing values: {'Yes' if profile['has_missing'] else 'No'}")
    print(f"  Value range:    [{profile['value_range'][0]:.2f}, {profile['value_range'][1]:.2f}]")
    print(f"  Mean / Std:     {profile['mean']:.3f} / {profile['std']:.3f}")
    print(f"  Has labels:     {'Yes (' + str(profile['n_classes']) + ' classes)' if profile['has_labels'] else 'No'}")

    if "metadata" in profile:
        print(f"  {'-'*46}")
        for k, v in profile["metadata"].items():
            if isinstance(v, (list, tuple)) and len(v) > 5:
                v_display = f"[{v[0]!r}, {v[1]!r}, ... ({len(v)} total)]"
            else:
                v_display = v
            print(f"  {k}: {v_display}")
    print(f"{'='*50}\n")


def write_profile_summary_md(name, profile, path):
    """Render a profile dict as a short Markdown summary, a quick-to-skim
    companion to profile.json (same JSON+MD pairing as validate_plan.py's
    reports)."""
    lines = [f"# Profile summary: {name}", ""]
    lines.append(f"- **Samples:** {profile['n_samples']:,}")
    lines.append(f"- **Features:** {profile['n_features']:,}")
    lines.append(f"- **Sparsity:** {profile['sparsity'] * 100:.1f}% zeros")
    lines.append(f"- **Missing values:** {'Yes' if profile['has_missing'] else 'No'}")
    lines.append(
        f"- **Value range:** [{profile['value_range'][0]:.2f}, {profile['value_range'][1]:.2f}]"
    )
    lines.append(f"- **Mean / Std:** {profile['mean']:.3f} / {profile['std']:.3f}")
    if profile["has_labels"]:
        lines.append(f"- **Labels:** Yes ({profile['n_classes']} classes)")
    else:
        lines.append("- **Labels:** No")

    if "metadata" in profile:
        lines.append("")
        lines.append("## Metadata")
        for k, v in profile["metadata"].items():
            if isinstance(v, (list, tuple)) and len(v) > 5:
                v_display = f"`{v[0]!r}`, `{v[1]!r}`, ... ({len(v)} total)"
            else:
                v_display = v
            lines.append(f"- **{k}:** {v_display}")

    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile a dataset; writes profile.json + profile_summary.md."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name: a folder under data/")
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write profile.json (default: outputs/<dataset>/profile.json)",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path("outputs") / args.dataset / "profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(args.dataset)
    profile = profile_dataset(X, y, metadata)
    print_profile(args.dataset, profile)

    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    md_path = out_path.with_name("profile_summary.md")
    write_profile_summary_md(args.dataset, profile, md_path)

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")