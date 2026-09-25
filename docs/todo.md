# Status and priorities

Deadline: Mon 9/28, 11:59pm EST. Design details live in `docs/design.md`; this is just the running list.

## Done
- Core pipeline, all deterministic tools tested: `profiler`, `validate_plan`, `reduce_dim` (all approved methods except GPLVM), `evaluate`, `visualize` (+ scree plots), `run_plan`, `doctor`, `report`.
- Skills: `data-profiling`, `method-selection`, `reporting`. `CLAUDE.md` written and kept in sync.
- Both required datasets run end to end and produce `generated_report_1.pdf` / `generated_report_2.pdf`.
- Generic loader: every dataset, including pbmc and pathmnist, loads the same way (csv/tsv/txt, npy, npz, h5ad), guided by a `## Loading` section in its `DATA_DESCRIPTION.md`; `fetch_data.py` handles downloads as a separate setup step. 36 unit tests (`python -m unittest discover -s tests`).

## Next, in order
1. **Missing values in numeric data**: nan-aware profiling, an imputation step, and a `validate_plan` guard so data with missing values can't reach a method unhandled. Tested one piece at a time.
2. **Fixes from the Scanpy PBMC3k comparison**, each tested rather than assumed: cluster on PCA and color unlabeled plots by those clusters; test gene scaling before PCA; rename or replace `select_hvg` (it is top-variance, not dispersion-based).
3. `README.md` (architecture, decision-making, how to run it).
4. First real single-prompt runs (pbmc, then pathmnist), unattended.
5. Manual 4-page `report.pdf`.

## Lower priority / deliberate limitations (state these in the report)
- **Categorical (non-numeric) features are unsupported**; the loader refuses them with a specific error. Design notes for picking it up later are in `design.md` §3.8.
- No QC filtering of cells/features (row filtering would need label alignment across scripts; the mitochondrial filter is gene-name specific).
- No marker-gene coloring or cell-type annotation (needs domain knowledge; the findings rules forbid naming cell types for unlabeled data).
- UMAP runs on the preprocessed matrix independently, not on a PCA neighbor graph as the Scanpy tutorial does.
- GPLVM is not implemented; the adversarial critic subagent and the synthetic-data generator are deferred.
- Dense-only pipeline: very large sparse datasets hit a clear size-limit error at load time.
