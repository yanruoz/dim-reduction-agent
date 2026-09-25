---
name: data-profiling
description: Turn a dataset's profile.json and DATA_DESCRIPTION.md into a concrete, ordered list of preprocessing steps, with a dataset-specific reason for each. Used in step 3 of CLAUDE.md's execution sequence, right after profiling and right before method-selection.
---

# Data profiling → preprocessing decision

By the time this is read, `outputs/<dataset>/profile.json` and `data/<dataset>/DATA_DESCRIPTION.md` already exist and have been read (steps 1-2 of CLAUDE.md). This skill turns those into the `preprocessing` list for `plan.json`.

## Available steps

Exactly seven, implemented in `scripts/reduce_dim.py`'s `PREPROCESSORS` registry (`plan_schema.KNOWN_PREPROCESSING_STEPS`). Never invent a step name outside this list; `validate_plan.py` rejects anything else.

| Step | What it does | Params |
|---|---|---|
| `normalize_total` | Rescales each sample's row to a common total, removing per-sample "depth" differences | optional `target_sum` |
| `log1p` | `log(1+x)`, compresses a heavy right-skewed tail | none |
| `select_top_variance` | Keeps the `n_top_features` columns with the highest variance (ranks features by raw variance, not by mean-adjusted dispersion) | `n_top_features` |
| `standardize` | Zero mean, unit variance per feature | none |
| `scale_unit_range` | Rescales to `[0, 1]` | optional `min`, `max` |
| `drop_missing_features` | Removes features whose missing fraction exceeds `max_fraction`; always removes entirely-missing features | optional `max_fraction` (default 0.5, must be in [0, 1)) |
| `impute` | Fills each missing value with a per-feature statistic computed over observed values | optional `strategy` (`median` default, `mean`, `constant`), `fill_value` for `constant` (default 0) |

## Decision tree

Work through these in order, starting with step 0 (missing values); more than one can apply to the same dataset. In the written plan, missing-value steps go first, then steps 1-4 as they apply.

0. **Missing values come first** (`profile.json`'s `has_missing` is `true`). Numeric data only; the loader refuses non-numeric columns. Every method needs finite input, and `validate_plan.py` fails a plan that leaves missing values unhandled or puts these steps out of order. They must be the first steps in the list.
   - If any feature is entirely missing (`n_features_all_missing > 0`), or some features are mostly missing, add `drop_missing_features` first. Choose `max_fraction` from `top_missing_features` and `max_feature_missing_fraction` in the profile; say in the reason which features it removes and why that threshold. Dropping is a judgment call (alternatives: keep and impute, or a stricter cutoff), so label it as one (rule 11).
   - Then add `impute`. Default `median` because it is robust to the skew and outliers common in real measurements; use `mean` only if the description says values are symmetric, and `constant` only if the description says a missing value means a real zero (e.g. an absent count). Cite `missing_fraction` and `n_features_with_missing` in the reason.
   - If `missing_fraction` exceeds 0.2, `validate_plan.py` warns. The reason must then say why proceeding is still sound, or the plan should say the results are weak.
   - Limitation to state in the reasons and the report: single-value imputation understates uncertainty and pulls imputed samples toward the feature centre, which can shrink apparent structure. Missing-not-at-random data (missingness that depends on the value or on group) will be biased by any of these fills.
   - If `has_missing` is `false`, add neither step.

1. **Count-like data?** Non-negative, integer-valued, high sparsity in `profile.json`, and/or `DATA_DESCRIPTION.md` says the values are counts, reads, or UMIs. The description is the authority here; the loader supplies no modality hint, since every dataset is loaded the same way.
   - If yes: `normalize_total` then `log1p`, in that order. Cite the actual sparsity percentage and/or value range from `profile.json` in the reason (e.g. "97.4% sparse raw UMI counts per profile.json; normalizing removes per-cell sequencing-depth differences before log-compressing the heavy tail").
   - If `DATA_DESCRIPTION.md` says normalization was already applied upstream, skip this and say so; never double-normalize.
   - Do not add `standardize` after `normalize_total` + `log1p` + `select_top_variance` by default. It was tested on one 2,700-cell, 32,738-gene scRNA-seq set with no labels (2,000 top-variance genes, 50-PC PCA): coarse structure was nearly unchanged (k-means ARI 0.96 at k=4, 0.82 at k=8), but fine neighborhoods overlapped only 17% (mean Jaccard of 15-nearest-neighbor sets), and the 50 PCs captured 16.4% of variance instead of 28.9%, because every gene, including low-expression noisy ones, got equal weight. Without labels there is no way to call that better. One dataset is evidence, not a rule: if you add `standardize` here, say in the `reason` that it is a judgment call and name that alternative.

2. **Is `n_features` large relative to `n_samples`?** Rough guide: `n_features` exceeds `n_samples`, or is in the thousands.
   - If yes: add `select_top_variance` after any count-normalization above, with `n_top_features` around `min(2000, n_features)`. Cite the actual `n_samples`/`n_features` numbers in the reason.
   - If `n_features <= n_samples` (a modest, already-manageable feature count), skip this step; cutting features when dimensionality isn't actually a problem just discards information for no reason.

3. **Dense, naturally bounded-range data?** E.g. pixel intensities 0-255, or `DATA_DESCRIPTION.md` states a fixed value range. Not the same case as step 1.
   - If yes: `scale_unit_range`, so no feature dominates purely from raw scale. Cite the actual `value_range` from `profile.json`.

4. **Dense, continuous, no natural bound, varying per-feature scales?** Not covered by 1 or 3.
   - If yes: `standardize`.

## Output

Write the resulting ordered list into `plan.json`'s `"preprocessing"` array: `{"step": ..., "params": {...} (if any), "reason": "..."}`. Every `reason` must cite an actual number or fact from *this* dataset's `profile.json`/`DATA_DESCRIPTION.md` (CLAUDE.md rules 11-12); "standard choice" alone is never acceptable, even when the step itself is a common one.
