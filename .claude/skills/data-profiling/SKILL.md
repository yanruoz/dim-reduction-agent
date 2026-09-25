---
name: data-profiling
description: Turn a dataset's profile.json and DATA_DESCRIPTION.md into a concrete, ordered list of preprocessing steps, with a dataset-specific reason for each. Used in step 3 of CLAUDE.md's execution sequence, right after profiling and right before method-selection.
---

# Data profiling → preprocessing decision

By the time this is read, `outputs/<dataset>/profile.json` and `data/<dataset>/DATA_DESCRIPTION.md` already exist and have been read (steps 1-2 of CLAUDE.md). This skill turns those into the `preprocessing` list for `plan.json`.

## Available steps

Exactly five, implemented in `scripts/reduce_dim.py`'s `PREPROCESSORS` registry (`plan_schema.KNOWN_PREPROCESSING_STEPS`). Never invent a step name outside this list; `validate_plan.py` rejects anything else.

| Step | What it does | Params |
|---|---|---|
| `normalize_total` | Rescales each sample's row to a common total, removing per-sample "depth" differences | optional `target_sum` |
| `log1p` | `log(1+x)`, compresses a heavy right-skewed tail | none |
| `select_hvg` | Keeps the `n_top_genes` columns with the highest variance (generic top-variance feature selection; the name is descriptive, not gene-specific) | `n_top_genes` |
| `standardize` | Zero mean, unit variance per feature | none |
| `scale_unit_range` | Rescales to `[0, 1]` | optional `min`, `max` |

## Decision tree

Work through these in order; more than one can apply to the same dataset.

1. **Count-like data?** Non-negative, integer-valued, high sparsity in `profile.json`, and/or `DATA_DESCRIPTION.md` says the values are counts, reads, or UMIs. The description is the authority here; the loader supplies no modality hint, since every dataset is loaded the same way.
   - If yes: `normalize_total` then `log1p`, in that order. Cite the actual sparsity percentage and/or value range from `profile.json` in the reason (e.g. "97.4% sparse raw UMI counts per profile.json; normalizing removes per-cell sequencing-depth differences before log-compressing the heavy tail").
   - If `DATA_DESCRIPTION.md` says normalization was already applied upstream, skip this and say so; never double-normalize.

2. **Is `n_features` large relative to `n_samples`?** Rough guide: `n_features` exceeds `n_samples`, or is in the thousands.
   - If yes: add `select_hvg` after any count-normalization above, with `n_top_genes` around `min(2000, n_features)`. Cite the actual `n_samples`/`n_features` numbers in the reason.
   - If `n_features <= n_samples` (a modest, already-manageable feature count), skip this step; cutting features when dimensionality isn't actually a problem just discards information for no reason.

3. **Dense, naturally bounded-range data?** E.g. pixel intensities 0-255, or `DATA_DESCRIPTION.md` states a fixed value range. Not the same case as step 1.
   - If yes: `scale_unit_range`, so no feature dominates purely from raw scale. Cite the actual `value_range` from `profile.json`.

4. **Dense, continuous, no natural bound, varying per-feature scales?** Not covered by 1 or 3.
   - If yes: `standardize`.

5. **Missing values** (`profile.json`'s `has_missing` is `true`): no imputation step exists in this pipeline. State this explicitly as a limitation in the plan's reasoning and later in the report, rather than silently proceeding or letting a method crash unexplained on NaN input. If a method then fails on this dataset, that failure is expected and should be documented via the Fallback rules in `CLAUDE.md`, not treated as a surprise bug.

## Output

Write the resulting ordered list into `plan.json`'s `"preprocessing"` array: `{"step": ..., "params": {...} (if any), "reason": "..."}`. Every `reason` must cite an actual number or fact from *this* dataset's `profile.json`/`DATA_DESCRIPTION.md` (CLAUDE.md rules 11-12); "standard choice" alone is never acceptable, even when the step itself is a common one.
