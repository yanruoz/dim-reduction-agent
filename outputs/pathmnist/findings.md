## Method results

PCA (general purpose, 50 components) has trustworthiness 0.9820 and explains 0.7907 of the total variance; 90% is reached at 178 components. Sparse PCA (general purpose, 50 components) is nearly identical on trustworthiness at 0.9816. UMAP (visualization only, 2 components) has trustworthiness 0.8127. The UMAP score is not comparable head to head with the two 50-dimensional PCA-family scores, since a 2-dimensional layout is expected to preserve less neighbourhood structure. All three trustworthiness values were computed on a random subset of 5,000 of the 89,996 samples, so treat them as estimates.

## Labels

The label silhouette is slightly negative for all three methods (PCA -0.0664, Sparse PCA -0.0487, UMAP -0.0588). By this measure, none of the embeddings groups the nine tissue classes tightly. The plots agree in a limited way: labels 0 and 1 occupy visibly distinct regions in both the PCA and UMAP plots, while the other classes overlap heavily. This means raw 28x28 pixel intensities alone do not separate most of the classes in these embeddings; it does not mean the classes are indistinguishable in general. The labels were used only for coloring and for this sanity check, never to build an embedding or tune a hyperparameter.
