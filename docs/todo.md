# Status and priorities

Deadline: Mon 9/28, 11:59pm EST. Design details live in `docs/design.md`; this is just the running list.

## Done
- Core pipeline, all deterministic tools tested: `profiler`, `validate_plan`, `reduce_dim` (all approved methods except GPLVM), `evaluate`, `visualize` (+ scree plots), `run_plan`, `doctor`, `report`.
- Skills: `data-profiling`, `method-selection`, `reporting`. `CLAUDE.md` written and kept in sync.
- Both required datasets run end to end and produce `generated_report_1.pdf` / `generated_report_2.pdf`.
- Generic loader: every dataset, including pbmc and pathmnist, loads the same way (csv/tsv/txt, npy, npz, h5ad), guided by a `## Loading` section in its `DATA_DESCRIPTION.md`; `fetch_data.py` handles downloads as a separate setup step. 
- Missing values in numeric data (tested piece by piece): nan-aware profiler, `drop_missing_features` and `impute` steps, a runtime finite-check backstop, `validate_plan` guards (required, ordered, parameter checks, high-missing warning), extra NA tokens in the loader (feature columns only), report profile line and Limitations note. Checked end to end on a synthetic CSV with about 11% missing cells and a mostly-missing column.
- Data-derived cluster coloring for unlabeled datasets (Track B, item 1): optional `clustering` block in `plan.json`, `cluster.py` (k-means on a general_purpose embedding, silhouette scan on a subsample), `validate_plan` guards (source must be general_purpose, k/k_range checks), `run_plan.py` runs figures last and redraws them when the coloring changes, report methods page and Limitations line. The label palette now covers more than 10 classes (tab20, then the 19 largest plus a gray "other"). Checked on pbmc: silhouette picked k=2 over the range 2-10, which is coarse; k_range [4,10] gave k=4 and separated the visible islands. The skill tells the agent to treat that as a stated judgment call.
- Gene scaling tested on pbmc (scratch experiment, not a pipeline feature): `standardize` after `select_top_variance` (formerly `select_hvg`) left coarse structure alone (k-means ARI 0.96 at k=4, 0.82 at k=8) but changed fine neighborhoods a lot (15-NN Jaccard 0.17) and flattened the variance spectrum (50 PCs: 16.4% vs 28.9%). No labels to judge which is better, so the default stays unscaled and the data-profiling skill records the evidence.
- `select_hvg` renamed to `select_top_variance` (param `n_top_features`), since it ranks by variance only. A dispersion-based selector was not added: the scaling test gave no evidence it is needed.
- Label silhouette in `evaluate.py` is subsampled above 5,000 samples like trustworthiness (fields `label_silhouette_subsampled`, `label_silhouette_n_samples_used`; the report's Limitations discloses it). Existing pathmnist metrics predate this and were scored on all samples; `run_plan.py` reuses them, so delete `outputs/pathmnist/metrics/` to recompute.
- 158 unit tests. Run with the project venv: `venv/bin/python -m unittest discover -s tests` (the anaconda `python` has no anndata, so 5 tests fail there).

## Next, in order
1. `README.md` (architecture, decision-making, how to run it).
2. First real single-prompt runs (pbmc, then pathmnist), unattended.
3. Manual 4-page `report.pdf`.

## Lower priority / deliberate limitations (state these in the report)
- **Imputation limits**: single-value imputation only (median/mean/constant); no model-based or multiple imputation, and no test of whether values are missing at random.
- **Categorical (non-numeric) features are unsupported**; the loader refuses them with a specific error. Design notes for picking it up later are in `design.md` §3.8.
- No QC filtering of cells/features (row filtering would need label alignment across scripts; the mitochondrial filter is gene-name specific).
- No marker-gene coloring or cell-type annotation (needs domain knowledge; the findings rules forbid naming cell types for unlabeled data).
- UMAP runs on the preprocessed matrix independently, not on a PCA neighbor graph as the Scanpy tutorial does.
- GPLVM is not implemented; the adversarial critic subagent and the synthetic-data generator are deferred.
- Dense-only pipeline: very large sparse datasets hit a clear size-limit error at load time.
