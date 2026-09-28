# Dimension-reduction EDA agent

An agent that takes a dataset it has never seen and produces a dimension-reduction analysis: a logged plan, embeddings, figures, metrics, and a PDF report. It is driven by one instruction, for example:

```
do the data analysis on data/pbmc
```

The "brain" is Claude Code, guided by `CLAUDE.md` and three skills. There is no separate LLM-API planner and no agent framework. Everything that produces a number or a figure is a deterministic script, so the model's job is to decide and to write, not to compute.

This is unsupervised analysis. If a dataset has labels they are used only to color plots and as a sanity check, never as a model input.

## How it works

```
prompt: "do the data analysis on data/<name>"
   │
   ▼
CLAUDE.md (rules, execution sequence)  ──reads──▶  data/<name>/DATA_DESCRIPTION.md
   │
   ├─ 1  doctor.py        check packages and data file
   ├─ 2  profiler.py      → outputs/<name>/profile.json (+ profile_summary.md)
   ├─ 3  skills           data-profiling → preprocessing steps
   │                      method-selection → methods, hyperparameters, plan schema
   ├─ 4  agent writes     → outputs/<name>/plan.json   (every entry has a dataset-specific "reason")
   ├─ 5  validate_plan.py  must pass before anything runs
   ├─ 6  run_plan.py      reduce_dim → evaluate → (cluster) → visualize, per method
   │                      → embeddings/, metrics/, figures/, run_log.json
   ├─ 7  critic           (not built yet, skipped)
   ├─ 8  agent writes     → outputs/<name>/findings.md ; report.py → reports/<name>/generated_report_<N>.pdf
   └─ 9  self-check       checklist at the end of CLAUDE.md
```

State lives in JSON files, so every step can be rerun or resumed alone. `run_plan.py` reuses an existing embedding when its method, hyperparameters and seed match the plan.

### How the agent makes decisions

- **Profile first.** `profile.json` records sample and feature counts, sparsity, value range, missing values (counts, fractions, worst features) and whether labels exist.
- **Skills hold the decision logic.**
  - `.claude/skills/data-profiling/SKILL.md` maps profile properties to preprocessing steps: missing-value handling first, then count normalization, top-variance feature selection, or rescaling.
  - `.claude/skills/method-selection/SKILL.md` chooses methods by size and role, and defines `plan.json`.
  - `.claude/skills/reporting/SKILL.md` defines what the written findings may and may not say.
- **Method roles are fixed** (`scripts/plan_schema.py`). `general_purpose` methods (PCA, Kernel PCA, Sparse PCA, MDS, Isomap, Diffusion Maps) can feed further quantitative work. `local_structure_only` methods (LLE, Laplacian Eigenmaps) preserve neighborhoods. `visualization_only` methods (t-SNE, UMAP) are for plots and their own quality checks only, and can never be the input to clustering or any other method. `validate_plan.py` enforces this.
- **Every decision is logged.** Each preprocessing step and method carries a `reason` that must cite something about this dataset; "standard choice" is rejected by the validator. Judgment calls must say so and name the alternatives.
- **Size guards.** MDS, Isomap, Kernel PCA and Diffusion Maps are O(N²); above about 5,000 samples the plan must subsample or choose another method. Trustworthiness and the label silhouette are subsampled above 5,000 samples and the report says so.
- **Failure handling.** A method that fails is retried once (larger `n_neighbors` for disconnected graphs), then skipped with PCA reported for that slot. One failure never stops the run. A missing PDF dependency stops only the report step.



### Methods available

PCA, Kernel PCA, Sparse PCA, MDS, Isomap, LLE, Laplacian Eigenmaps, Diffusion Maps (own numpy/scipy implementation), t-SNE, UMAP. GPLVM is not implemented (a stub raises `NotImplementedError`). The agent picks a justified subset per dataset instead of running everything.

### Preprocessing steps available

`drop_missing_features`, `impute` (median, mean or constant), `normalize_total`, `log1p`, `select_top_variance`, `standardize`, `scale_unit_range`. Missing-value steps must come first; `validate_plan.py` and a run-time finite-value check both enforce that no method sees NaN or inf.

### Metrics

- **Trustworthiness** (k=10) for every embedding: how well the embedding keeps each point's nearest neighbors. It measures local fidelity only.
- **PCA family:** variance explained, PCs needed for 90% (shown as `>N` when not reached), scree plot.
- **MDS:** Kruskal Stress-1.
- **Label silhouette**, only when labels exist, as a sanity check.

For unlabeled datasets, plots can be colored by data-derived clusters (k-means on a general-purpose embedding, k chosen by silhouette). These are a coloring only, and the report says so.

## Setup

Requires Python 3.10 or newer (developed on 3.13).

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

Versions are pinned because t-SNE and UMAP layouts are somewhat version-sensitive.

### Data

Each dataset lives in `data/<name>/` with a `DATA_DESCRIPTION.md`, which is the one source of truth for what the data is. Data files themselves are git-ignored. An optional `## Loading` section of `key: value` lines tells the loader where things are:


| key                         | meaning                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `file`                      | data file name (not needed if the folder has exactly one)         |
| `label_column`, `id_column` | table columns to treat as labels / ids (labels are never guessed) |
| `delimiter`                 | for csv/tsv/txt; sniffed if omitted                               |
| `x_key`, `y_key`            | array names inside an `.npz`                                      |
| `url`, `md5`                | where to download the file, and its checksum                      |


Supported files: csv, tsv, txt, npy, npz, h5ad. Every dataset, including the two below, goes through the same loader.

To download the two datasets used here (a one-time setup step, never done mid-analysis):

```bash
venv/bin/python scripts/fetch_data.py --dataset pbmc
venv/bin/python scripts/fetch_data.py --dataset pathmnist
```

- `pbmc`: Scanpy's raw PBMC 3k single-cell counts (2,700 cells, 32,738 genes, no labels).
- `pathmnist`: MedMNIST PathMNIST training images (89,996 flattened 28×28×3 images, 9 tissue classes).



## Running the agent

From this directory, start Claude Code and give it one instruction naming one dataset:

```
do the data analysis on data/pbmc
```

An optional report number can be added ("... as report 1"); otherwise the next free number is used. It writes `outputs/<name>/plan.json` and `reports/<name>/generated_report_<N>.pdf`.

## Running the scripts by hand

Every step is a plain script (`--help` on any of them shows its flags):

```bash
venv/bin/python scripts/doctor.py --check core --dataset pbmc
venv/bin/python scripts/profiler.py --dataset pbmc --out outputs/pbmc/profile.json
venv/bin/python scripts/validate_plan.py outputs/pbmc/plan.json
venv/bin/python scripts/run_plan.py outputs/pbmc/plan.json
venv/bin/python scripts/doctor.py --check report
venv/bin/python scripts/report.py --dataset pbmc --report-number 1
```

`plan.json` is written by the agent (or by hand for a manual run); its schema is in `.claude/skills/method-selection/SKILL.md`.

## Outputs

```
outputs/<name>/
  profile.json, profile_summary.md   dataset profile
  plan.json, plan_validation.*       the agent's plan and its validation result
  embeddings/<method>.npy + .json    embedding and its hyperparameters/seed (.npy files are git-ignored)
  metrics/<method>.json              trustworthiness, diagnostics, label silhouette
  metrics/clustering.json            only when the plan has a clustering block
  figures/<method>.png, *_scree.png  scatter plots and PCA scree plots
  findings.md                        the written interpretation, using only numbers from metrics/
  run_log.json                       what ran, was reused, fell back, or failed
reports/<name>/generated_report_<N>.pdf
```

The report (matplotlib `PdfPages`, no extra dependency) contains the profile, the preprocessing and its reasons, each method with its hyperparameters and reason, a metrics table with definitions, all figures, the written findings, and a limitations page.

## Tests

```bash
venv/bin/python -m unittest discover -s tests
```

About 160 tests cover the loader, profiler, preprocessing, plan validation, clustering, plotting logic, run orchestration and metrics. Use the project venv: the `anndata` tests fail under an interpreter without it.

## Layout

```
CLAUDE.md                    the agent's instructions
.claude/skills/              data-profiling, method-selection, reporting
scripts/                     deterministic tools (doctor, profiler, loaders, validate_plan, run_plan,
                             reduce_dim, evaluate, cluster, visualize, report, fetch_data, plan_schema)
data/<name>/                 DATA_DESCRIPTION.md (+ git-ignored data file)
outputs/, reports/           generated artifacts
tests/                       unit tests
docs/design.md, todo.md      design notes and running status
```



## Limitations

- **Numeric features only.** A dataset with non-numeric feature columns is refused with an error naming them. There is no categorical encoding.
- **Missing values** are handled by single-value imputation only (median, mean or constant). This understates uncertainty and can bias results if values are not missing at random.
- **Dense data only.** Sparse inputs are converted to dense; datasets over about 4 GB dense are refused at load time.
- **No QC filtering** of samples or features, and no domain annotation (for example cell types); the findings rules forbid naming classes the data does not label.
- **Manifold layouts are pictures.** t-SNE and UMAP cluster sizes, shapes and distances are not findings. UMAP is deterministic here (fixed seed, single thread) but still sensitive to small numeric changes in its input.
- **GPLVM** is not implemented, and the adversarial critic subagent is not built.
- **Data-derived clusters** are k-means on PCA with k chosen by silhouette, which tends to favor coarse splits. They only color plots. Confirmed on the live pbmc run: the default scan picked k=2 and under-colored visible structure, so the plan was revised to a higher lower bound as a stated judgment call.
- **Artifacts are keyed by method name only,** so a second run of the same method under a different seed or hyperparameter setting can't be kept alongside the first. The "two seeds or settings" check on a manifold method's structural claims has to use two different methods instead (e.g. UMAP and t-SNE), not two UMAP runs.
- **No run history.** Rerunning `cluster.py` or reusing a method's artifacts overwrites the previous state (`metrics/clustering.json`, and `run_log.json`'s status), so an earlier attempt's own numbers aren't recoverable afterward; a plan revision's reasoning has to live in `plan.json`'s `reason` text, not in a diffable metrics history.
- **No label-name mapping.** `DATA_DESCRIPTION.md` has no key for an index-to-name lookup, so a dataset that documents named classes is still plotted and reported with the raw integer labels.
- **`sparse_pca` reports no variance-explained diagnostic**, since sklearn's `SparsePCA` has no `PCA`-style `explained_variance_ratio_` for its non-orthogonal components; its metrics/report rows are blank in those columns (disclosed by the report, not hidden).

