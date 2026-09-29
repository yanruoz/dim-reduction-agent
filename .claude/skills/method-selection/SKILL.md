---
name: method-selection
description: Pick 2-4 dimension-reduction methods and their hyperparameters given a dataset's profile.json and the preprocessing already decided, and defines plan.json's full schema. Used in step 3 of CLAUDE.md's execution sequence, right after data-profiling.
---

# Method selection and plan.json schema

## plan.json schema (authoritative; CLAUDE.md only gives a brief reminder)

```json
{
  "dataset": "pbmc",
  "preprocessing": [
    {"step": "normalize_total", "reason": "..."},
    {"step": "log1p", "reason": "..."}
  ],
  "methods": [
    {
      "name": "pca",
      "role": "general_purpose",
      "hyperparameters": {"n_components": 50},
      "reason": "..."
    }
  ],
  "clustering": {"source": "pca", "algorithm": "kmeans", "k_range": [2, 10], "reason": "..."},
  "evaluation": {"quantitative": ["trustworthiness"], "qualitative": ["seed_stability_check"]},
  "seed": 0,
  "revision": 1,
  "critique_applied": null
}
```

- `preprocessing`: from `data-profiling/SKILL.md`, already decided before this skill runs.
- `methods`: this skill's job, see below. `name` must be in `plan_schema.ALLOWED_METHODS`; `role` must exactly match that method's fixed entry in `plan_schema.METHOD_ROLES` (see next section), `validate_plan.py` rejects a mismatch, it is not a free-text field.
- `clustering`: optional, see "Coloring unlabeled plots" below.
- `evaluation`: informational only, states your intended evaluation approach for the eventual report. `evaluate.py` computes its own fixed metrics per method regardless of what's listed here (trustworthiness for any embedding, explained variance ratio for PCA-family methods, stress for MDS, etc.); this field doesn't control that, it's a note to yourself and to the reporting step about what to emphasize.
- `seed`: one integer for the whole plan; every method's own `random_state` hyperparameter, if it has one, must equal this (`validate_plan.py` warns if not).
- `revision`/`critique_applied`: start at `1`/`null`; only touched by the one-time critique-and-revise loop (step 7 of CLAUDE.md, `.claude/agents/dr-critic.md`). After the one revision that loop makes: `revision` becomes `2`, and `critique_applied` is a short string naming what changed and why, not just `true`.

## Coloring unlabeled plots (`clustering` block)

Without labels, every plot is a single-color density scatter. If `profile.json` has `has_labels: false`, add a `clustering` block so `run_plan.py` colors the plots by data-derived clusters. If the dataset has labels, leave it out: labels color the plots and the clusters would go unused (`validate_plan.py` warns).

- `source`: a method already in this plan whose role is `general_purpose`, normally `pca`. `validate_plan.py` rejects a `visualization_only` or local-structure source, and `cluster.py` refuses it again at run time: UMAP/t-SNE embeddings never feed clustering (CLAUDE.md Guard rails).
- `algorithm`: `kmeans` only.
- `k_range`: `[low, high]`, default `[2, 10]`; `k` (optional) fixes the number of clusters and skips the scan. With no `k`, `cluster.py` scores every candidate on a fixed-seed subsample of at most 5,000 points and picks k by the density cross-check below (or by highest silhouette if that cross-check doesn't apply).
- `reason`: required, dataset-specific, like every other entry.
- **Silhouette alone favors coarse splits.** It often scores the lowest k in the range best, which would color an obviously multi-group embedding with too few colors — its global average is dominated by whichever partition has the fewest, best-separated blobs, so a smaller-but-real group sitting close to a much larger one can get absorbed and still score well. `cluster.py` no longer relies on silhouette alone by default: when the plan includes a `visualization_only` method (UMAP preferred over t-SNE, see `cluster.py`'s `VIZ_PREFERENCE`), it runs a density cross-check automatically — DBSCAN on that visualization (with an automatic, parameter-free `eps` from the k-distance-graph knee) nominates candidate groups purely from density, and the smallest scanned k where every candidate group lands in its own distinct k-means cluster *on the actual `general_purpose` clustering source* is selected. The visualization's geometry never becomes the clustering result or feeds any computation; it only nominates a candidate, which is then verified against the real embedding before anything acts on it — consistent with the guard rail against treating `visualization_only` shapes as literally true.
- **When this runs, you don't need to hand-tune `k_range` for it.** Check `metrics/clustering.json`'s `k_selection.selected_by`: `"density_cross_check"` means it worked as intended (look at `n_significant_groups` and `group_sizes` to sanity-check what it found); `"density_cross_check_never_satisfied_in_range"` means candidate groups were found but no k up to `k_range`'s upper bound separated all of them, worth widening the range as a judgment call, same as before; `"silhouette_argmax"` means no `visualization_only` method was in the plan to cross-check against, so this is plain silhouette (add a UMAP or t-SNE method if you want the cross-check).
- **This is a default, not a guarantee.** DBSCAN's own eps heuristic can still find zero or one significant group on a dataset with genuinely no multi-group structure (correctly falling back to silhouette), or, rarely, find spurious groups (a visualization's own layout noise). Sanity-check `group_sizes` and the resulting figure against what you'd expect before trusting it blindly, the same care any automatic default deserves.
- The clusters are a description of the embedding for coloring only. They are not evaluated against anything, and `findings.md` may call them "data-derived clusters" but must not name cell types or other real-world classes.

## Method roles: not a competition

Every approved method has a fixed category in `plan_schema.METHOD_ROLES`:

| Role | Methods | What it means |
|---|---|---|
| `general_purpose` | pca, kernel_pca, sparse_pca, mds, isomap, diffusion_maps, gplvm | Preserves enough global structure to trust as an input to further quantitative work (clustering, distances). Isomap is here too, it targets geodesic distance, though used this way less often in practice than PCA. |
| `local_structure_only` | lle, laplacian_eigenmaps | Preserves local neighborhoods, not global distances. A real middle category, available if a dataset's structure calls for it, but not required by default. |
| `visualization_only` | tsne, umap | Optimizes for a good-looking local layout at the direct cost of global distance fidelity, by design, not a flaw. **Never feed this embedding into any further computation**, its only job is to be looked at. |

**A `general_purpose` method and a `visualization_only` method in the same plan are not competing for the same job.** Including both isn't redundant, PCA (or another general-purpose method) is the trustworthy numeric representation; UMAP (or another visualization-only method) is purely the picture. Don't reason about them as "which one is better," they answer different questions.

`gplvm` has no implementation in `reduce_dim.py` yet (raises `NotImplementedError` on purpose); never select it. If that changes later, this note needs updating.

## Selection decision tree

Always start from `profile.json`'s `n_samples` (after any subsampling decided in preprocessing), since it gates which methods are computationally reasonable.

1. **Always include PCA**, `role: general_purpose`. It's the required baseline for every dataset, no exceptions.

2. **Always include exactly one PCA variant**, per the assignment's requirement. Choose based on `n_samples`:
   - `n_samples <= 5000`: `kernel_pca` is fine (its `O(N²)` kernel matrix is manageable at this size). Default `kernel: "rbf"`. Its `gamma` defaults to sklearn's own `1/n_features`, which can be too small to make the kernel behave nonlinearly at all on a wide feature space (found live: on a 2,000-gene matrix, RBF Kernel PCA at the default gamma came out nearly identical to linear PCA). If `metrics/kernel_pca.json`'s embedding or trustworthiness looks suspiciously close to plain PCA's, that's worth checking rather than assuming Kernel PCA "just found the same structure"; pass an explicit `gamma` (larger than `1/n_features` shrinks the kernel's effective neighborhood, more nonlinear) as a judgment call, stating why that value in the `reason`.
   - `n_samples > 5000`: use `sparse_pca` instead. `kernel_pca` and `mds`/`isomap`/`diffusion_maps` all become impractical above this threshold (CLAUDE.md's Guard rails); don't select any of them here without an explicit, documented subsample step. **Cost warning, measured:** `sparse_pca` is not cheap either; at 89,996 samples x 2,352 features and 50 components it took about 23 minutes (vs. ~8 seconds for plain PCA and ~2 minutes for UMAP on the same data). It is still the right choice here since the alternatives don't run at all, but expect it to dominate the run time, and keep `n_components` modest for it.

3. **Include one `visualization_only` method**, default `umap` (faster and more scalable than `tsne` at any size we'll see). Only add `tsne` as well if you have a specific reason to compare two visualization layouts, it's a nice-to-have, not required, since it fills the same role as UMAP.

4. **Optionally include one more `general_purpose` manifold method**, when `n_samples <= 5000` and the dataset plausibly has interesting nonlinear/manifold structure worth a second, non-linear "trustworthy" embedding (e.g. `diffusion_maps` is a natural fit for scRNA-seq data specifically, where it has a well-established track record). Above the 5000-sample threshold, skip this rather than forcing an expensive method or an undocumented subsample; three well-chosen methods beats four where one required a shortcut.

5. **`local_structure_only` methods (`lle`, `laplacian_eigenmaps`) are available but not defaulted to.** Consider one only if the dataset's structure specifically calls for a local-neighborhood embedding that isn't purely for visualization; otherwise the visualization-only method already covers that territory.

This should typically land on 3 methods for a large dataset (PCA, Sparse PCA, UMAP) or 4 for a smaller one (PCA, Kernel PCA, Diffusion Maps, UMAP), both within the assignment's "one or more" requirement and this project's own 2-4 guideline. Fewer or more is fine if the reasoning is genuinely dataset-specific, don't force a count.

## Hyperparameter heuristics

- **PCA / Kernel PCA / Sparse PCA**: `n_components = min(50, floor(n_samples / 2), n_features)` (after preprocessing). 50 balances capturing structure against overfitting to a small sample count; the `n_samples / 2` cap avoids asking for more components than the data can support.
- **UMAP**: `n_neighbors = 15` by default; drop to `10` if `n_samples < 500`; raise to `30` if `n_samples > 20000`. `min_dist = 0.1` (UMAP's own default). `n_components = 2` (it's for plotting).
- **Isomap / LLE / Laplacian Eigenmaps** (when selected): `n_neighbors` similar to UMAP's heuristic, generally smaller (~10) since these are more sensitive to a disconnected neighbor graph; `n_components = 2`.
- **Diffusion Maps** (when selected): `n_components` around 10, not 2, since its role is `general_purpose` rather than direct plotting; `visualize.py` can still plot its first two dimensions if a figure is wanted.
- Every stochastic method's `random_state` (if it has one) must equal the plan's top-level `seed`.

## Output

Write the `methods` list into `plan.json`, each entry with `name`, `role`, `hyperparameters`, and a `reason` citing an actual fact about *this* dataset (its size, structure, or something from `DATA_DESCRIPTION.md`), not a generic justification that would apply to any dataset (CLAUDE.md rules 11-12). Then validate with `scripts/validate_plan.py` before moving on.
