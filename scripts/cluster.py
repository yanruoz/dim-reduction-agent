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

# --- Density cross-check: picking k so k-means doesn't merge genuinely distinct groups ---
#
# Plain silhouette-argmax favors the coarsest split in the range (its average is dominated by
# whichever partition has the fewest, best-separated blobs), so it can systematically miss a real,
# smaller group that sits close to a much larger one. Found on pbmc: a ~350-cell group, visibly its
# own island in every figure, stayed merged into the main cluster at every k up to a manually raised
# floor. This automates that manual investigation: DBSCAN on a visualization_only embedding (UMAP
# preferred; that objective preserves local neighborhoods/density directly, unlike t-SNE) nominates
# candidate groups from density alone, with no k or count chosen by hand. Those candidates are used
# ONLY to check the real clustering (k-means on the general_purpose source): the smallest scanned k
# where every candidate group has its own distinct majority k-means cluster is selected. The
# visualization_only embedding never feeds the clustering itself, only this diagnostic question,
# matching CLAUDE.md's guard rail (its own geometry is never trusted directly, only used to raise a
# candidate that's then verified against the general-purpose embedding before anything acts on it).
DENSITY_MIN_SAMPLES = 10
DENSITY_MIN_GROUP_FRACTION = 0.01  # a DBSCAN group must be >= 1% of the scored sample (and >= DENSITY_MIN_SAMPLES) to count
VIZ_PREFERENCE = ("umap", "tsne")  # UMAP preferred: its objective preserves local density/neighborhoods more directly


def _knee_eps(Z, min_samples):
    """Automatic DBSCAN eps via the k-distance-graph knee: the point of maximum perpendicular
    distance from the chord connecting the sorted k-distance curve's endpoints. Standard,
    parameter-free heuristic for where a point cloud's density visibly drops off; no eps chosen by
    hand for any particular dataset."""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    dists, _ = nbrs.kneighbors(Z)
    kdist = np.sort(dists[:, -1])  # distance to each point's min_samples-th neighbor, ascending

    x = np.arange(len(kdist))
    p1, p2 = np.array([x[0], kdist[0]]), np.array([x[-1], kdist[-1]])
    line_norm = (p2 - p1) / np.linalg.norm(p2 - p1)
    vecs = np.stack([x - p1[0], kdist - p1[1]], axis=1)
    perp = vecs - np.outer(vecs @ line_norm, line_norm)
    return float(kdist[np.argmax(np.linalg.norm(perp, axis=1))])


def _density_reference_groups(viz_Z, min_group_size):
    """DBSCAN candidate groups on a 2D (or other) visualization embedding, dropping noise (-1) and
    anything smaller than min_group_size. Returns {label: boolean mask (len(viz_Z),)} and the eps used."""
    from sklearn.cluster import DBSCAN

    eps = _knee_eps(viz_Z, DENSITY_MIN_SAMPLES)
    labels = DBSCAN(eps=eps, min_samples=DENSITY_MIN_SAMPLES).fit_predict(viz_Z)
    groups = {
        int(lab): (labels == lab)
        for lab in np.unique(labels)
        if lab != -1 and int((labels == lab).sum()) >= min_group_size
    }
    return groups, eps


def _groups_map_to_distinct_clusters(groups, kmeans_labels, n_clusters):
    """True if every reference group's majority k-means cluster is unique to it, i.e. no two
    groups share a majority cluster (k-means hasn't merged genuinely distinct groups together)."""
    majority = {}
    for lab, mask in groups.items():
        counts = np.bincount(kmeans_labels[mask], minlength=n_clusters)
        majority[lab] = int(np.argmax(counts))
    return len(set(majority.values())) == len(groups), majority


def _select_k(fits, scores, candidates, viz_Z, idx):
    """Picks k. With a usable visualization_only embedding and >=2 significant density groups on
    it, picks the smallest scanned k where every group gets its own distinct k-means cluster.
    Otherwise (no such embedding, or fewer than 2 significant groups, or the check is never
    satisfied within the range) falls back to the silhouette argmax, as before. Returns (k, info)."""
    density = {"used": False}
    if viz_Z is not None:
        min_size = max(int(DENSITY_MIN_GROUP_FRACTION * len(idx)), DENSITY_MIN_SAMPLES)
        groups, eps = _density_reference_groups(viz_Z[idx], min_size)
        density = {
            "used": True,
            "eps": eps,
            "min_samples": DENSITY_MIN_SAMPLES,
            "n_significant_groups": len(groups),
            "group_sizes": sorted((int(m.sum()) for m in groups.values()), reverse=True),
        }
        if len(groups) >= 2:
            for kk in sorted(candidates):
                if scores.get(kk) is None:
                    continue
                distinct, majority = _groups_map_to_distinct_clusters(groups, fits[kk][idx], kk)
                if distinct:
                    density["selected_by"] = "density_cross_check"
                    density["majority_cluster_per_group"] = {str(g): c for g, c in majority.items()}
                    return kk, density
            density["selected_by"] = "density_cross_check_never_satisfied_in_range"

    scored = {kk: s for kk, s in scores.items() if s is not None}
    density.setdefault("selected_by", "silhouette_argmax")
    return max(scored, key=scored.get), density


def choose_and_fit(Z, k_range, k, seed, viz_embedding=None):
    """KMeans on Z (n, d). With `k` given, uses it. Otherwise picks k via `_select_k` (density
    cross-check against `viz_embedding` if given, else silhouette argmax) on a fixed-seed
    subsample. Returns (labels (n,), info dict)."""
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

    if not any(s is not None for s in scores.values()):
        raise ValueError("no candidate k produced a scorable clustering")

    if k is not None:
        best_k, k_selection = k, {"selected_by": "fixed"}
    else:
        best_k, k_selection = _select_k(fits, scores, candidates, viz_embedding, idx)

    # Relabel by cluster size (0 = largest) so label numbers are stable across reruns
    labels = fits[best_k]
    order = np.argsort(-np.bincount(labels, minlength=best_k), kind="stable")
    remap = np.empty(best_k, dtype=int)
    remap[order] = np.arange(best_k)
    labels = remap[labels]

    info = {
        "k": int(best_k),
        "k_was_fixed": k is not None,
        "k_selection": k_selection,
        "silhouette_by_k": {str(kk): s for kk, s in scores.items()},
        "silhouette_at_k": scores[best_k],
        "cluster_sizes": [int(c) for c in np.bincount(labels, minlength=best_k)],
        "n_samples": int(n),
        "n_scored": int(len(idx)),
        "silhouette_subsampled": bool(n > SILHOUETTE_SCAN_SIZE),
    }
    return labels, info


def _embedding_fingerprint(dataset, method):
    sidecar = Path("outputs") / dataset / "embeddings" / f"{method}.json"
    side = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return {"hyperparameters": side.get("hyperparameters", {}), "seed": side.get("seed")}


def _find_viz_embedding(dataset, plan):
    """Path to a visualization_only method's embedding already computed for this dataset, if any
    (preferring UMAP, see VIZ_PREFERENCE), plus its name for the cache fingerprint. Used only to
    nominate candidate groups for choose_and_fit's density cross-check, never as clustering input."""
    in_plan = {m["name"] for m in plan.get("methods", []) if METHOD_ROLES.get(m["name"]) == "visualization_only"}
    out = Path("outputs") / dataset / "embeddings"
    ordered = [n for n in VIZ_PREFERENCE if n in in_plan] + sorted(in_plan - set(VIZ_PREFERENCE))
    for name in ordered:
        path = out / f"{name}.npy"
        if path.exists():
            return name, path
    return None, None


def _params(dataset, plan, block, seed):
    viz_name, _ = _find_viz_embedding(dataset, plan)
    return {
        "source": block["source"],
        "algorithm": block.get("algorithm", "kmeans"),
        "k_range": block.get("k_range", [2, 10]),
        "k": block.get("k"),
        "seed": seed,
        "source_embedding": _embedding_fingerprint(dataset, block["source"]),
        "viz_source": viz_name,
        "viz_embedding": _embedding_fingerprint(dataset, viz_name) if viz_name else None,
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

    seed = plan.get("seed", 0)
    params = _params(dataset, plan, block, seed)
    labels_path, metrics_path = out / "clusters.npy", out / "metrics" / "clustering.json"
    if labels_path.exists() and metrics_path.exists():
        if json.loads(metrics_path.read_text()).get("params") == params:
            return "reused"

    Z = np.load(emb_path)  # (n, n_components) of the source method
    _, viz_path = _find_viz_embedding(dataset, plan)
    viz_Z = np.load(viz_path) if viz_path is not None else None
    labels, info = choose_and_fit(Z, params["k_range"], params["k"], seed, viz_embedding=viz_Z)

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
