"""Dataset profiler. Fully generic: takes the common (X, y, metadata)
   interface from loaders.py and computes summary statistics. This file
   should never contain dataset-specific branching.
"""
import numpy as np
import sys
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


if __name__ == "__main__":

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "pbmc3k"
    X, y, metadata = load_dataset(dataset_name)
    profile = profile_dataset(X, y, metadata)
    print_profile(dataset_name, profile)