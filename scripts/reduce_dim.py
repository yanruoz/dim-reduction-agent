"""Runs one dimension-reduction method against one dataset, applying the
   preprocessing block from that dataset's plan.json (never re-decides it).
   Labels are never accepted here by design; they're evaluation-only,
   handled in evaluate.py/visualize.py instead.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from loaders import load_dataset
from plan_schema import KNOWN_PREPROCESSING_STEPS

# ── Preprocessing steps: generic, work on any (n_samples, n_features) array ──


def _normalize_total(X, params):
    # (n_samples, n_features) -> same shape; each row rescaled to a common total
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    target = params.get("target_sum") or float(np.median(X.sum(axis=1)))
    return X / row_sums * target


def _log1p(X, params):
    return np.log1p(X)


def _select_hvg(X, params):
    """Keep the n_top_genes columns with the highest variance. Generic top-variance
    feature selection; the scRNA-seq name is descriptive, not dataset-specific."""
    n_top = min(params.get("n_top_genes", 2000), X.shape[1])
    variances = X.var(axis=0)
    top_idx = np.sort(np.argsort(variances)[::-1][:n_top])  # keep original column order
    return X[:, top_idx]


def _standardize(X, params):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def _scale_unit_range(X, params):
    lo = params.get("min", float(X.min()))
    hi = params.get("max", float(X.max()))
    span = (hi - lo) or 1.0
    return (X - lo) / span


PREPROCESSORS = {
    "normalize_total": _normalize_total,
    "log1p": _log1p,
    "select_hvg": _select_hvg,
    "standardize": _standardize,
    "scale_unit_range": _scale_unit_range,
}
assert set(PREPROCESSORS) == KNOWN_PREPROCESSING_STEPS, "PREPROCESSORS must match plan_schema's known steps"


def apply_preprocessing(X, preprocessing_steps):
    for step in preprocessing_steps:
        name = step["step"]
        if name not in PREPROCESSORS:
            raise ValueError(f"Unknown preprocessing step '{name}'; not in {sorted(PREPROCESSORS)}")
        X = PREPROCESSORS[name](X, step.get("params") or {})
    return X


# ── Dimension-reduction methods ──
# Each takes (X, hyperparameters, seed) -> (embedding, extra_info_dict).
# `seed` is always the authoritative source of randomness, not
# hyperparameters.get("random_state") (validate_plan.py already warns if the
# two disagree; here we just never read the hyperparameter copy).


def _run_pca(X, hp, seed):
    """Fits deeper than n_components asks for, so the fixed embedding it
    returns can be reported alongside how much variance it actually captured
    and how many components a 90%-variance target would need. The requested
    n_components is still exactly what gets returned as the embedding; this
    only adds diagnostic numbers to the sidecar output, per-component reasons
    still come from plan.json, not this heuristic."""
    from sklearn.decomposition import PCA

    n_requested = hp.get("n_components", 2)
    # Look 4x deeper than requested, capped at 200 and at what's available,
    # so the diagnostic fit stays cheap even on a large dataset.
    diagnostic_cap = min(4 * n_requested, 200, X.shape[0], X.shape[1])

    model = PCA(n_components=diagnostic_cap, random_state=seed)
    full_embedding = model.fit_transform(X)  # (n_samples, diagnostic_cap)
    embedding = full_embedding[:, :n_requested]  # (n_samples, n_requested): the actual returned embedding

    evr = model.explained_variance_ratio_  # length == diagnostic_cap
    cumulative = np.cumsum(evr)
    reached_90 = np.where(cumulative >= 0.90)[0]
    if len(reached_90) > 0:
        components_for_90pct = int(reached_90[0] + 1)
        is_lower_bound = False
    else:
        # 90% wasn't reached within the diagnostic fit; report a true lower
        # bound rather than guessing how many more components it would take.
        components_for_90pct = diagnostic_cap
        is_lower_bound = True

    return embedding, {
        "explained_variance_ratio": evr[:n_requested].tolist(),
        # Full spectrum from the diagnostic fit (up to diagnostic_cap components),
        # kept so a scree plot can show more than just the components used.
        "explained_variance_ratio_diagnostic": evr.tolist(),
        "variance_explained_by_used_components": round(float(evr[:n_requested].sum()), 4),
        "diagnostic_fit_components": diagnostic_cap,
        "components_needed_for_90pct_variance": components_for_90pct,
        "components_needed_for_90pct_variance_is_lower_bound": is_lower_bound,
    }


def _run_kernel_pca(X, hp, seed):
    from sklearn.decomposition import KernelPCA

    model = KernelPCA(
        n_components=hp.get("n_components", 2),
        kernel=hp.get("kernel", "rbf"),
        random_state=seed,
    )
    return model.fit_transform(X), {}


def _run_sparse_pca(X, hp, seed):
    from sklearn.decomposition import SparsePCA

    model = SparsePCA(n_components=hp.get("n_components", 2), random_state=seed)
    return model.fit_transform(X), {}


def _run_mds(X, hp, seed):
    from sklearn.manifold import MDS

    # normalized_stress=True gives Kruskal's Stress-1, a scale-free number. sklearn's default
    # ("auto") returns raw stress for metric MDS, which depends on the data's scale and can't
    # be interpreted on its own. Supported for metric MDS since sklearn 1.7.
    model = MDS(n_components=hp.get("n_components", 2), random_state=seed, normalized_stress=True)
    embedding = model.fit_transform(X)
    return embedding, {"stress": float(model.stress_)}


def _run_isomap(X, hp, seed):
    from sklearn.manifold import Isomap

    model = Isomap(n_components=hp.get("n_components", 2), n_neighbors=hp.get("n_neighbors", 5))
    return model.fit_transform(X), {}


def _run_lle(X, hp, seed):
    from sklearn.manifold import LocallyLinearEmbedding

    model = LocallyLinearEmbedding(
        n_components=hp.get("n_components", 2),
        n_neighbors=hp.get("n_neighbors", 10),
        random_state=seed,
    )
    return model.fit_transform(X), {}


def _run_laplacian_eigenmaps(X, hp, seed):
    from sklearn.manifold import SpectralEmbedding

    model = SpectralEmbedding(
        n_components=hp.get("n_components", 2),
        n_neighbors=hp.get("n_neighbors", 10),
        random_state=seed,
    )
    return model.fit_transform(X), {}


def _run_diffusion_maps(X, hp, seed):
    """Minimal diffusion map: RBF affinity -> row-normalized Markov matrix ->
    its top eigenvectors, scaled by eigenvalues^alpha. No installed package
    implements this, so it's done directly with numpy/scipy."""
    from scipy.linalg import eigh
    from scipy.spatial.distance import pdist, squareform

    n_components = hp.get("n_components", 2)
    alpha = hp.get("alpha", 1.0)
    epsilon = hp.get("epsilon")

    sq_dists = squareform(pdist(X, "sqeuclidean"))  # (n_samples, n_samples)
    if epsilon is None:
        epsilon = np.median(sq_dists[sq_dists > 0])
    K = np.exp(-sq_dists / epsilon)

    d = K.sum(axis=1)
    d[d == 0] = 1.0
    d_sqrt = np.sqrt(d)
    P_sym = (K / d_sqrt[:, None]) / d_sqrt[None, :]  # symmetrized transition matrix

    eigvals, eigvecs = eigh(P_sym)
    order = np.argsort(eigvals)[::-1]
    # skip the trivial top eigenvalue/eigenvector (constant, carries no structure)
    eigvals = eigvals[order][1 : n_components + 1]
    eigvecs = eigvecs[:, order][:, 1 : n_components + 1] / d_sqrt[:, None]

    embedding = eigvecs * (eigvals**alpha)
    return embedding, {"eigenvalues": eigvals.tolist()}


def _run_tsne(X, hp, seed):
    from sklearn.manifold import TSNE

    model = TSNE(
        n_components=hp.get("n_components", 2),
        perplexity=hp.get("perplexity", 30),
        random_state=seed,
        init="pca",
    )
    return model.fit_transform(X), {}


def _run_umap(X, hp, seed):
    import umap

    model = umap.UMAP(
        n_components=hp.get("n_components", 2),
        n_neighbors=hp.get("n_neighbors", 15),
        min_dist=hp.get("min_dist", 0.1),
        random_state=seed,
        n_jobs=1,  # required by umap-learn for exact reproducibility with a fixed seed
    )
    return model.fit_transform(X), {}


def _run_gplvm(X, hp, seed):
    raise NotImplementedError(
        "GPLVM is an optional stretch goal; no lightweight scikit-learn implementation "
        "is available. Skip it and document why, per CLAUDE.md."
    )


METHODS = {
    "pca": _run_pca,
    "kernel_pca": _run_kernel_pca,
    "sparse_pca": _run_sparse_pca,
    "mds": _run_mds,
    "isomap": _run_isomap,
    "lle": _run_lle,
    "laplacian_eigenmaps": _run_laplacian_eigenmaps,
    "diffusion_maps": _run_diffusion_maps,
    "tsne": _run_tsne,
    "umap": _run_umap,
    "gplvm": _run_gplvm,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one dimension-reduction method against a dataset's plan.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--params", default="{}", help="JSON dict of hyperparameters")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--out", default=None, help="Path for the embedding .npy (default: outputs/<dataset>/embeddings/<method>.npy)"
    )
    args = parser.parse_args()

    plan_path = Path("outputs") / args.dataset / "plan.json"
    if not plan_path.exists():
        sys.exit(f"No plan.json found at {plan_path}; write one first (see validate_plan.py).")
    plan = json.loads(plan_path.read_text())

    X, y, metadata = load_dataset(args.dataset)  # y is loaded but never used below: labels are eval-only
    X = apply_preprocessing(X, plan.get("preprocessing", []))  # (n_samples, n_features_after_preprocessing)

    hyperparameters = json.loads(args.params)

    start = time.time()
    embedding, extra = METHODS[args.method](X, hyperparameters, args.seed)  # (n_samples, n_components)
    runtime = time.time() - start

    out_path = Path(args.out) if args.out else Path("outputs") / args.dataset / "embeddings" / f"{args.method}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embedding)

    sidecar = {
        "method": args.method,
        "hyperparameters": hyperparameters,
        "seed": args.seed,
        "runtime_seconds": round(runtime, 3),
        "embedding_shape": list(embedding.shape),
        **extra,
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"Wrote {out_path} (shape {embedding.shape}, {runtime:.2f}s)")
    print(f"Wrote {sidecar_path}")
