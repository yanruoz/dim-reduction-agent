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
}
