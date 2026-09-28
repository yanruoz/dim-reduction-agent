## What each method produced

All four methods in the plan ran successfully on all 89,996 images after one shared preprocessing step (rescaling pixel intensities from [0, 255] to [0, 1]). No method needed a retry or a fallback. The seed is 0 throughout. Because the dataset is large, trustworthiness and the label silhouette were both computed on a random subset of 5,000 images, not on the full set.

PCA (50 components) captured 0.7907 of the total variance. The first component alone accounts for 0.5465; the next ones are much smaller (0.0229, 0.0209, 0.0185), and the scree plot shows one dominant component followed by a long, slowly decaying tail. Reaching 90% of the variance takes 178 components. So one direction carries over half of the pixel variance, while the rest is spread across many small directions. What that first direction represents was not tested here (the pipeline does not inspect loadings); in image data a single dominant component often tracks overall brightness, but that is a possibility, not a finding.

Sparse PCA was run with 20 components (reduced from 50 for run time, as recorded in the plan). sklearn's Sparse PCA does not report explained variance, so there is no scree plot or variance figure for it.

Trustworthiness (10 nearest neighbors) was 0.9820 for PCA (50 dimensions), 0.9435 for Sparse PCA (20 dimensions), 0.8347 for t-SNE (2 dimensions) and 0.8127 for UMAP (2 dimensions). These are not a ranking: the embeddings have different numbers of dimensions, and more dimensions leave more room to preserve neighborhoods. The fair comparison is between the two 2D methods, where t-SNE scores slightly higher on this measure.

## Label sanity check

The 9 tissue labels were used only to color plots and to compute a silhouette score in each embedding. The label silhouette was slightly negative for every method: -0.0662 for PCA, -0.0533 for Sparse PCA, -0.0336 for t-SNE and -0.0604 for UMAP. A value near zero or below means that, measured by distances in these embeddings, images are on average about as close to images of other labels as to images of their own label. This says the unsupervised embeddings of raw pixels do not arrange the 9 labels into compact, well-separated groups overall. It does not mean the tissue types are indistinguishable: the figures show that some labels do separate while most overlap, and a single average score hides that difference.

## What the figures show

The labels are stored as integers 0 to 8. DATA_DESCRIPTION.md lists the 9 tissue names but does not state which integer maps to which name, so this report refers to labels by number only rather than guessing the mapping.

Label 1 is the clearest separate group. It sits apart from the main body of points in PCA and Sparse PCA (at the low end of the second component and, in PCA, also the first), forms its own detached structure in UMAP, and forms several separate islands in t-SNE. Since this shows up in both linear methods and in both visualization methods, it is the most robust structural observation in this analysis.

Label 0 occupies its own region at one end of the main body in all four figures: at the high end of the first component in PCA and Sparse PCA, at the lower tip of the elongated UMAP shape, and as a distinct arm on the left of the t-SNE layout. It touches the rest of the data rather than being detached.

Label 3 is concentrated in a recognisable region in every figure (the lower-middle part of the PCA and Sparse PCA clouds, one side of the UMAP shape, and a dense band near the bottom of the main t-SNE mass), but it borders and partly overlaps neighbouring labels.

The remaining labels (2, 4, 5, 6, 7 and 8) overlap heavily in all four embeddings. Some show local concentrations in t-SNE (for example small label-5 and label-2 patches), but no method separates them cleanly. A few very small, tight groups of label-2 points sit away from the main mass in UMAP and t-SNE; their size and meaning cannot be read from these plots.

## How the methods relate

PCA and Sparse PCA are the general-purpose embeddings, the ones whose coordinates could reasonably be used for further quantitative work; their 2D views look broadly alike, with the same labels separating. UMAP and t-SNE are pictures only. They were run together so that no structural statement rests on a single layout, and they agree on the points above: label 1 detached, label 0 at one end, label 3 in a concentrated region, and heavy overlap among the rest. Neither was used for any computation beyond its own trustworthiness and label-silhouette scores.

## What cannot be concluded

These embeddings are built from raw 28x28 pixel intensities. DATA_DESCRIPTION.md notes that the downsampling from 224x224 loses fine histological texture and that pixel similarity does not imply tissue similarity, so heavy label overlap here is a statement about this pixel representation, not about how separable the tissue types are in principle. The 2D distances, island sizes and shapes in UMAP and t-SNE carry no quantitative meaning. Both scores were computed on a 5,000-image subset. Sparse PCA used fewer components than PCA (20 against 50), so their trustworthiness scores are not directly comparable either. Each method was run with one seed; the UMAP/t-SNE agreement is the only cross-check on the visual structure. Kernel PCA, MDS, Isomap and Diffusion Maps were not run because their N x N cost is impractical at 89,996 images.
