"""Dataset loaders. Each loader returns a common interface:
   (X, y, metadata) where X is a 2D array (n_samples x n_features),
   y is optional labels (or None), and metadata is a dict of extra info.

   All dataset-specific logic lives here. Nothing downstream of this
   file should ever branch on which dataset is loaded.
"""
from pathlib import Path
import numpy as np
import scanpy as sc
from medmnist import PathMNIST

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_pbmc3k():
    sc.settings.datasetdir = DATA_ROOT / "pbmc"
    adata = sc.datasets.pbmc3k()

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    metadata = {
        "source": "pbmc3k",
        "modality": "single_cell_rna_seq",
        "feature_names": adata.var_names.tolist(),
    }
    return X, None, metadata


def load_pathmnist(split="train"):
    root = DATA_ROOT / "pathmnist"
    root.mkdir(parents=True, exist_ok=True)
    dataset = PathMNIST(split=split, download=True, root=str(root))

    images = dataset.imgs  # shape: (N, H, W, C)
    X = images.reshape(images.shape[0], -1).astype(np.float64)
    y = dataset.labels.flatten()

    metadata = {
        "source": "pathmnist",
        "modality": "image",
        "image_shape": images.shape[1:],
    }
    return X, y, metadata


# Registry so the rest of the pipeline can look datasets up by name
LOADERS = {
    "pbmc3k": load_pbmc3k,
    "pathmnist": load_pathmnist,
}


def load_dataset(name: str):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(LOADERS)}")
    return LOADERS[name]()


if __name__ == "__main__":
    for name in LOADERS:
        X, y, meta = load_dataset(name)
        print(name, "-> X:", X.shape, "y:", None if y is None else y.shape, "meta:", meta)