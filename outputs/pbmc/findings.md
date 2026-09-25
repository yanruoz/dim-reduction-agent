## Method results

PCA (general purpose, 50 components) has trustworthiness 0.9344 and captures 0.2904 of the total variance. 90% of the variance was not reached within the 200-component diagnostic fit (more than 200 PCs are needed). The variance spectrum is flat, so variance in this dataset is spread across many directions rather than concentrated in a few, and a linear projection to 50 dimensions keeps under a third of it.

UMAP (visualization only, 2 components) has trustworthiness 0.7740. This is not a like-for-like comparison with PCA's 0.9344, because the two embeddings have 2 and 50 dimensions; a 2-dimensional layout is expected to preserve less neighbourhood structure than a 50-dimensional one. The two methods do different jobs: PCA is the numeric representation, UMAP is the picture.

## What the figures show

The UMAP plot shows three visually separated groups and one isolated point. The PCA plot, in its first two components, shows a more continuous and less separated cloud. This dataset has no labels, so the groups cannot be checked against known cell types, and UMAP's group sizes and the gaps between groups should not be read as measurements. Treat the structure as exploratory.
