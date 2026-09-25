"""Data-derived cluster labels, used ONLY to color plots when a dataset has no
   ground-truth labels. Reads the plan's optional `clustering` block, clusters
   a general_purpose embedding (never a visualization_only one), and writes
   outputs/<dataset>/clusters.npy plus outputs/<dataset>/metrics/clustering.json.
   These clusters are a description of the embedding, not a finding: no
   biological or domain meaning is attached to them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plan_schema import CLUSTERING_ALGORITHMS, METHOD_ROLES

SILHOUETTE_SCAN_SIZE = 5000  # silhouette is O(N^2); the k scan scores a fixed-seed subsample above this
KMEANS_N_INIT = 10


def choose_and_fit(Z, k_range, k, seed):
    """KMeans on Z (n, d). With `k` given, uses it; otherwise picks the k in k_range with the
    highest silhouette on a fixed-seed subsample. Returns (labels (n,), info dict)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = Z.shape[0]
    candidates = [k] if k is not None else list(range(k_range[0], min(k_range[1], n - 1) + 1))
    if not candidates or min(candidates) < 2 or max(candidates) >= n:
        raise ValueError(f"need 2 <= k < n_samples ({n}); got candidates {candidates}")

    # Fixed-seed subsample for scoring only (KMeans itself is fit on all n rows)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=SILHOUETTE_SCAN_SIZE, replace=False) if n > SILHOUETTE_SCAN_SIZE else np.arange(n)

    fits, scores = {}, {}
    for kk in candidates:
        labels = KMeans(n_clusters=kk, n_init=KMEANS_N_INIT, random_state=seed).fit_predict(Z)  # (n,)
        fits[kk] = labels
        # silhouette needs at least 2 distinct labels among the scored rows
        scores[kk] = float(silhouette_score(Z[idx], labels[idx])) if len(np.unique(labels[idx])) > 1 else None

    scored = {kk: s for kk, s in scores.items() if s is not None}
    if not scored:
        raise ValueError("no candidate k produced a scorable clustering")
    best_k = k if k is not None else max(scored, key=scored.get)

    # Relabel by cluster size (0 = largest) so label numbers are stable across reruns
    labels = fits[best_k]
    order = np.argsort(-np.bincount(labels, minlength=best_k), kind="stable")
    remap = np.empty(best_k, dtype=int)
    remap[order] = np.arange(best_k)
    labels = remap[labels]

    info = {
        "k": int(best_k),
        "k_was_fixed": k is not None,
        "silhouette_by_k": {str(kk): s for kk, s in scores.items()},
        "silhouette_at_k": scores[best_k],
        "cluster_sizes": [int(c) for c in np.bincount(labels, minlength=best_k)],
        "n_samples": int(n),
        "n_scored": int(len(idx)),
        "silhouette_subsampled": bool(n > SILHOUETTE_SCAN_SIZE),
    }
    return labels, info


def _source_fingerprint(dataset, source):
    sidecar = Path("outputs") / dataset / "embeddings" / f"{source}.json"
    side = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return {"hyperparameters": side.get("hyperparameters", {}), "seed": side.get("seed")}


def _params(block, seed, fingerprint):
    return {
        "source": block["source"],
        "algorithm": block.get("algorithm", "kmeans"),
        "k_range": block.get("k_range", [2, 10]),
        "k": block.get("k"),
        "seed": seed,
        "source_embedding": fingerprint,
    }


def run(dataset, plan):
    """Returns 'ok', 'reused', or raises. Writes clusters.npy and metrics/clustering.json."""
    block = plan["clustering"]
    source = block["source"]
    if METHOD_ROLES.get(source) != "general_purpose":
        raise ValueError(
            f"clustering source {source!r} is not a general_purpose method; a visualization_only or "
            "local-structure embedding must never be the input to clustering"
        )
    if block.get("algorithm", "kmeans") not in CLUSTERING_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {sorted(CLUSTERING_ALGORITHMS)}")

    out = Path("outputs") / dataset
    emb_path = out / "embeddings" / f"{source}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"source embedding {emb_path} does not exist; run the plan's methods first")

    params = _params(block, plan.get("seed", 0), _source_fingerprint(dataset, source))
    labels_path, metrics_path = out / "clusters.npy", out / "metrics" / "clustering.json"
    if labels_path.exists() and metrics_path.exists():
        if json.loads(metrics_path.read_text()).get("params") == params:
            return "reused"

    Z = np.load(emb_path)  # (n, n_components) of the source method
    labels, info = choose_and_fit(Z, params["k_range"], params["k"], params["seed"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, labels.astype(np.int32))  # (n,)
    metrics_path.write_text(json.dumps({"params": params, **info}, indent=2, allow_nan=False) + "\n")
    return "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster a general_purpose embedding to color unlabeled plots.")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    plan = json.loads((Path("outputs") / args.dataset / "plan.json").read_text())
    if "clustering" not in plan:
        print("plan.json has no 'clustering' block; nothing to do")
        sys.exit(0)
    status = run(args.dataset, plan)
    print(f"clustering: {status}")
