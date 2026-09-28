## What each method produced

All five methods in the plan ran successfully on all 2,700 cells, using the same preprocessing: per-cell total-count normalization, log1p, and the 2,000 highest-variance genes. No method needed a retry or a fallback, and no metric was subsampled. The seed is 0 throughout.

PCA (50 components) captured 0.2904 of the total variance of the 2,000-gene matrix. The first component alone accounts for 0.1044, the second 0.0354 and the third 0.0226, after which the scree plot flattens quickly into a long tail of small, similar-sized components. 90% of the variance was not reached within the 200-component diagnostic fit (reported as >200). So most of the variance in the log-expression data is spread thinly across many directions, while a few leading components carry a clearly larger share. This fits sparse single-cell counts, where much of the per-gene variance is sampling noise.

Trustworthiness (10 nearest neighbors) was 0.9344 for PCA (50 dimensions), 0.9199 for Kernel PCA with an RBF kernel (50 dimensions), 0.8466 for Diffusion Maps (10 dimensions), 0.8023 for t-SNE (2 dimensions) and 0.7740 for UMAP (2 dimensions). These are not a head-to-head ranking. The embeddings have different numbers of dimensions, and a 50-dimensional embedding has far more room to keep neighborhoods intact than a 2-dimensional one. The fair comparisons are PCA against Kernel PCA (both 50 dimensions, similar scores) and UMAP against t-SNE (both 2 dimensions, t-SNE slightly higher on this measure). The leading Diffusion Maps eigenvalues were 0.1069 and 0.0342, with a similar steep drop after the first as in the PCA spectrum.

## What the figures show

Every figure shows the same broad pattern: one group of cells (cluster 2 in the coloring) sits clearly apart from the rest. It is separated along the first component in PCA and Kernel PCA, along the first diffusion coordinate in Diffusion Maps, and it forms its own island in both UMAP and t-SNE. Because this split shows up in the linear methods, in both nonlinear general-purpose methods, and in two visualization methods with different objectives, it is the most robust structural observation in this analysis.

The remaining cells are not one uniform mass. In PCA, Kernel PCA and Diffusion Maps they spread along an elongated direction, and in PCA and Kernel PCA a smaller, more compact group sits just below that main body. UMAP and t-SNE both show that compact group as a separate island as well. Since it appears in both visualization methods and in the linear PCA view, it is likely a real feature of the preprocessed data, not a layout artifact. It is not, however, captured by the k-means coloring: at k=3 it stays inside cluster 1.

Both UMAP and t-SNE also place a very small number of cells as an isolated speck far from everything else. How many cells that is, and what they are, can't be read off a 2D visualization. Given that DATA_DESCRIPTION.md says no quality-control filtering was applied, a small set of unusual or low-quality cells is one possible explanation, but this is not tested here.

## Data-derived clusters

The figures are colored by k-means on the 50-component PCA embedding. The default scan over k = 2 to 10 picked k = 2, the lower bound of the range, but the figures clearly showed more structure than two colors, so the plan was revised to scan k = 3 to 10, which picked k = 3 (silhouette 0.304). Silhouette fell almost steadily as k grew (k=7, at 0.206, is the only small uptick, over k=6 at 0.204), so the data do not strongly favor any particular larger number of groups. The resulting cluster sizes are 1,642, 686 and 372 cells. The third cluster splits the elongated main body along its elongated direction (the long arm in the Diffusion Maps and Kernel PCA plots) rather than picking out the compact group described above. These clusters are a coloring aid only. They are not validated groups, and their sizes should not be read as proportions of any biological population.

## How the methods relate

PCA, Kernel PCA and Diffusion Maps are the general-purpose embeddings: they are the ones whose coordinates could reasonably feed further quantitative work, and PCA is the one the clustering was built on. They broadly agree with each other on the main split. UMAP and t-SNE are only pictures. They were included together so that no structural claim rests on a single visualization layout, and they agree on the main separated group, the compact second island and the small isolated speck. Neither was used for any computation beyond its own trustworthiness score.

## What cannot be concluded

This dataset has no labels, so none of the groups can be named, and nothing here shows that they correspond to cell types. The distances between islands and the sizes and shapes of islands in UMAP and t-SNE carry no quantitative meaning. The data were not quality-filtered, and the pipeline has no QC step, so low-quality cells or doublets may influence both the leading components and the small outlying group. Feature selection ranked genes by raw variance rather than mean-adjusted dispersion, which can favor highly expressed genes; a different gene selection could change the finer structure. Finally, each method was run with one seed; the UMAP/t-SNE agreement is the only cross-check on the visual structure.
