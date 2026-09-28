"""Computes quantitative metrics for one method's embedding: trustworthiness
   for every method (subsampled above a size threshold, since sklearn's
   trustworthiness is itself O(N^2)), a labels-based sanity check if labels
   exist, and whatever method-specific diagnostics reduce_dim.py already
   saved (explained variance, stress, etc.), pulled in rather than
   recomputed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from loaders import load_dataset
from reduce_dim import apply_preprocessing

TRUSTWORTHINESS_N_NEIGHBORS = 10
# Matches validate_plan.py's LARGE_N_THRESHOLD: trustworthiness needs a full
# pairwise-distance computation internally, the same O(N^2) problem as
# MDS/Isomap/Kernel PCA/Diffusion Maps, but it applies to every method's
# embedding, not just those four, so it needs its own subsampling guard.
TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD = 5000
TRUSTWORTHINESS_SUBSAMPLE_SIZE = 5000

# Method-specific fields reduce_dim.py may have already computed; copied
# straight into metrics.json rather than recomputed, since they're already
# tied to the exact embedding on disk.
SIDECAR_FIELDS_TO_COPY = (
    "explained_variance_ratio",
    "explained_variance_ratio_diagnostic",
    "variance_explained_by_used_components",
    "diagnostic_fit_components",
    "components_needed_for_90pct_variance",
    "components_needed_for_90pct_variance_is_lower_bound",
    "variance_diagnostic_is_subspace_based",
    "stress",
    "eigenvalues",
)


def compute_trustworthiness(X, embedding, seed):
    from sklearn.manifold import trustworthiness

    n = X.shape[0]
    if n > TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=TRUSTWORTHINESS_SUBSAMPLE_SIZE, replace=False)
        X_used, embedding_used = X[idx], embedding[idx]
        subsampled = True
    else:
        X_used, embedding_used = X, embedding
        subsampled = False

    score = trustworthiness(X_used, embedding_used, n_neighbors=TRUSTWORTHINESS_N_NEIGHBORS)
    return float(score), subsampled, X_used.shape[0]


def compute_label_sanity_check(embedding, y, seed):
    """Silhouette score of the TRUE labels in embedding space. A supervised
    sanity check only, per CLAUDE.md: never used to tune hyperparameters,
    and never computed at all if there are no labels (e.g. pbmc). Silhouette
    is O(N^2), so above the same size threshold as trustworthiness it is scored
    on a fixed-seed random subsample. Returns (score, subsampled, n_used), or
    (None, False, 0) when there is nothing to score."""
    from sklearn.metrics import silhouette_score

    if y is None or len(np.unique(y)) < 2:
        return None, False, 0
    n = embedding.shape[0]
    if n > TRUSTWORTHINESS_SUBSAMPLE_THRESHOLD:
        idx = np.random.default_rng(seed).choice(n, size=TRUSTWORTHINESS_SUBSAMPLE_SIZE, replace=False)
        emb_used, y_used, subsampled = embedding[idx], np.asarray(y)[idx], True
        if len(np.unique(y_used)) < 2:  # a subsample that lost every but one class can't be scored
            return None, True, len(idx)
    else:
        emb_used, y_used, subsampled = embedding, y, False
    return float(silhouette_score(emb_used, y_used)), subsampled, emb_used.shape[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quantitative metrics for one method's embedding.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--embedding", default=None, help="Path to the embedding .npy (default: outputs/<dataset>/embeddings/<method>.npy)"
    )
    parser.add_argument("--out", default=None, help="Path for the metrics JSON (default: outputs/<dataset>/metrics/<method>.json)")
    args = parser.parse_args()

    plan_path = Path("outputs") / args.dataset / "plan.json"
    plan = json.loads(plan_path.read_text())

    X, y, metadata = load_dataset(args.dataset)  # y used below for scoring only, never as a feature
    X = apply_preprocessing(X, plan.get("preprocessing", []))  # same preprocessing reduce_dim.py used

    embedding_path = (
        Path(args.embedding) if args.embedding else Path("outputs") / args.dataset / "embeddings" / f"{args.method}.npy"
    )
    embedding = np.load(embedding_path)

    seed = plan.get("seed", 0)
    trust_score, subsampled, n_used = compute_trustworthiness(X, embedding, seed)

    metrics = {
        "method": args.method,
        "trustworthiness": round(trust_score, 4),
        "trustworthiness_n_neighbors": TRUSTWORTHINESS_N_NEIGHBORS,
        "trustworthiness_subsampled": subsampled,
        "trustworthiness_n_samples_used": n_used,
    }

    label_score, label_subsampled, label_n_used = compute_label_sanity_check(embedding, y, seed)
    if label_score is not None:
        metrics["label_silhouette_sanity_check"] = round(label_score, 4)
        metrics["label_silhouette_subsampled"] = label_subsampled
        metrics["label_silhouette_n_samples_used"] = label_n_used
        metrics["label_silhouette_note"] = (
            "Ground-truth-label silhouette in embedding space; a supervised sanity check only, "
            "never used to tune hyperparameters."
        )

    sidecar_path = embedding_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        for key in SIDECAR_FIELDS_TO_COPY:
            if key in sidecar:
                metrics[key] = sidecar[key]

    out_path = Path(args.out) if args.out else Path("outputs") / args.dataset / "metrics" / f"{args.method}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Wrote {out_path}")
    print(json.dumps(metrics, indent=2))
