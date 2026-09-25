"""Shared constants describing the plan.json schema. Imported by both
   validate_plan.py and reduce_dim.py so the approved-methods list and the
   known preprocessing steps can't drift out of sync between the two.
"""

ALLOWED_METHODS = {
    "pca", "kernel_pca", "sparse_pca",
    "mds", "isomap", "lle", "laplacian_eigenmaps", "diffusion_maps",
    "tsne", "umap", "gplvm",
}

KNOWN_PREPROCESSING_STEPS = {
    "normalize_total",
    "log1p",
    "select_hvg",
    "standardize",
    "scale_unit_range",
    "impute",
    "drop_missing_features",
}

# Steps that have to run before anything else because every other step either
# propagates NaN or (select_hvg's variance ranking) silently misbehaves on it.
MISSING_VALUE_STEPS = ("drop_missing_features", "impute")
IMPUTE_STRATEGIES = {"median", "mean", "constant"}

# Every plan.json method entry must declare a "role" matching its method's
# fixed category here (validate_plan.py checks this). Not a strict binary:
# - general_purpose: preserves enough global structure to trust as an input
#   to further quantitative work (clustering, distances); pca/kernel_pca/
#   sparse_pca/mds/diffusion_maps/gplvm live here, isomap too (it targets
#   geodesic distance) though it's used this way less often in practice.
# - local_structure_only: preserves local neighborhoods, not global
#   distances; a real middle category, not as extreme as the next one.
# - visualization_only: optimizes purely for a good-looking local layout at
#   the direct cost of global distance fidelity (t-SNE, UMAP). An embedding
#   in this category must never feed any further computation, only plotting.
METHOD_ROLES = {
    "pca": "general_purpose",
    "kernel_pca": "general_purpose",
    "sparse_pca": "general_purpose",
    "mds": "general_purpose",
    "isomap": "general_purpose",
    "diffusion_maps": "general_purpose",
    "gplvm": "general_purpose",
    "lle": "local_structure_only",
    "laplacian_eigenmaps": "local_structure_only",
    "tsne": "visualization_only",
    "umap": "visualization_only",
}
assert set(METHOD_ROLES) == ALLOWED_METHODS, "METHOD_ROLES must cover exactly the approved methods"

# Optional plan.json "clustering" block: labels used only to color unlabeled plots.
CLUSTERING_ALGORITHMS = {"kmeans"}

ALLOWED_ROLES = {"general_purpose", "local_structure_only", "visualization_only"}
