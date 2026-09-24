# Design: AI Agent for Dimension Reduction and EDA

Research-and-design document for BIOS774 Project 1. No pipeline code has been written; this is the plan.

## 0. Project audit

**Environment** (`venv/`, Python 3.13): numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scanpy 1.12.4, umap-learn 0.5.12, matplotlib 3.11.1, medmnist 3.0.2, anndata 0.13.3.post0, all already installed. Nothing needs installing to start. (`python-docx` was needed to read the STAI-X rules doc; it lives in the system Anaconda Python, not the venv, and was only used read-only for this research, not a project dependency.)

**Existing code:**

| File | What it does | Keep? |
|---|---|---|
| `agent/loaders.py` | Loads pbmc3k / pathmnist into a common `(X, y, metadata)` interface; a `LOADERS` registry keyed by name. Explicitly documents "nothing downstream should branch on which dataset is loaded." | **Keep, move to `scripts/loaders.py`** (see naming fix below). Exactly the right shape for a harness-driven agent. |
| `agent/profiler.py` | Generic `profile_dataset(X, y, metadata)` returning a dict (n_samples, n_features, sparsity, missing, value range, mean/std, labels/classes) plus a human-printer. | **Keep, extend, move to `scripts/profiler.py`.** Needs a CLI mode that writes `profile.json` and a companion `profile_summary.md` (see §1, item 3 below), not just print to stdout. |
| `agent/test_api.py` | Makes a direct `Anthropic()` API call ("Say hello in 5 words") to confirm connectivity. | **Delete once CLAUDE.md/skills work.** It belongs to a design we're no longer building: the harness-only decision means Claude Code itself is the planner, so a direct API call has no role in the pipeline. |
| `docs/todo.md` | "NEXT: build the planner, one LLM call that takes a profile dict and returns a JSON plan" plus tool-schema/agent-loop items. | **Rewrite.** That line describes the API-planner architecture the constraints rule out (no separate LLM-API planner). Replace with the module build order in §3.7. |
| `data/pbmc/pbmc3k_raw.h5ad`, `data/pathmnist/pathmnist.npz` | Raw data, already downloaded. | Keep. Add a `DATA_DESCRIPTION.md` next to each (see §3.1). |
| `notebooks/`, `outputs/` | Empty. | `notebooks/` stays empty and manual-only (CLAUDE.md forbids executing notebooks; the agent never writes here). `outputs/` becomes the intermediate artifact root (see §3.1). |

**Naming fix:** the project's own code folder is currently named `agent/`, which collides in name (though not in purpose) with `.claude/agents/`, the harness convention for subagent definition files. Renaming our folder to `scripts/` removes the ambiguity and matches how both reference repos name their CLI-tool folders. This is a plain `git mv`, no imports break (`profiler.py` imports `loaders` as a same-folder module, not a package path), and it's the first thing to do once implementation starts.

No repo currently exists for `CLAUDE.md`, `.claude/skills`, or `.claude/agents`; these are new.

## 1. Reference findings

Three repos were cloned shallow (`--depth 1`) into `../reference_repos/` and read against a fixed file budget (paths below are evidence, not the full repos). I also read the STAI-X 2026 rules doc's "Award B" section directly (`resources/STAI-X_Challenge_2026_Rules.docx`, paragraphs 52 to 72), since Sentinelle is literally an Award B submission and its structure is dictated by those rules: a single unattended prompt (`claude --dangerously-skip-permissions`, "Do the data analysis"), a `data/DATA_DESCRIPTION.md` the organizers drop in as source of truth, a hard schema check on the output with a worst-score fallback if it's missing or malformed, and a required `README.md` tutorial.

### Comparison table

| | Sentinelle (Award B) | biostat-superpowers | staix-submission (Award C skill) |
|---|---|---|---|
| **Orchestration brain** | Root `CLAUDE.md`: numbered "required execution sequence," "don't stop until deliverables exist," fallback rules, final checklist | `CLAUDE.md` + `AGENTS.md`; `AGENTS.md` maps skill-tool names to non-Claude runtimes and degrades gracefully with no subagent mechanism | None (single skill, not a full orchestrator) |
| **Skill anatomy** | One `SKILL.md`, frontmatter is `name`+`description` only, loaded by a **static pointer** from CLAUDE.md/agent file, not dynamic matching | Frontmatter is `name`+`description`, where `description` doubles as the trigger text; body always: framing, numbered workflow, constraints, hand-offs, output; `references/`/`scripts/` cited **just-in-time** inline at the step that needs them | Frontmatter adds an explicit `Triggers:` list of literal phrases; body: what it does, when to use it, how to use it (CLI + Python), what you get back, how it models, companion tool, dependencies |
| **Critic mechanism** | None (self-check checklist only) | `method-evaluation` skill: explicitly dispatched to a **separate-context sub-agent** given only artifacts, not the builder's reasoning, told to "try to break the analysis"; returns a 12-section report with a readiness rating (`not_ready` loops back, doesn't ship); schema validation is opt-in | None |
| **State between steps** | Files: `artifacts/`, `pipeline_run_log.json` (status `pre_report`), final run log | Convention only: each skill's "Output" section is what the next skill is told to consume; no JSON state machine except the critic's optional schema | A script's return value is always `(text_report, result_dict)`, both human- and machine-readable, no persisted file needed |
| **Testing pattern** | `create_mock_dataset.py`: seeded synthetic data plus a description file, for smoke-testing without real data; `validate_submission.py`: schema/row-id/finite-value checks, writes a JSON+Markdown pass/fail report | Not covered in the read set beyond examples (skipped per budget) | `demo.py` imports the script directly and prints a before/after comparison, a runnable proof of the failure mode the skill guards against, not a test suite |
| **License / reuse caveat** | No LICENSE seen (out of scope to check): **adapt structure, don't copy code** | MIT `LICENSE` present | `LICENSE` present in skill folder |

### 1. CLAUDE.md instructions worth adapting
A second, direct read of Sentinelle's actual `CLAUDE.md` (216 lines) surfaced more specific structural patterns worth adopting than the first subagent pass caught:
- **Anchor the mission to the literal invocation phrase.** Sentinelle opens with "When the user says: **Do the data analysis.** you must..." and ties "don't stop until X exists" directly to it. Ours does the same: don't stop until `plan.json` and `reports/<dataset>/generated_report_<N>.pdf` exist for every dataset in scope, with a narrow documented-failure exception.
- **An explicit "Evaluation setting" section.** Sentinelle spells out the exact adversarial scenario (judges clone the repo, drop a hidden dataset into `./data`, treat every column as unknown) right up front. Ours states the analogous scenario: a fresh Claude Code session may see a dataset it's never encountered, `DATA_DESCRIPTION.md` is the only source of truth, never assume a new dataset looks like PBMC3k or PathMNIST.
- **A flat, numbered "Non-negotiable rules" list**, not prose bullets. Sentinelle's own list mixes supervised-specific rules (target column, row count, MAE scoring, not reusable) with general operational ones that do generalize: read the description first, never touch `data/`, no external data, degrade gracefully if a library is missing, never finish with only intermediate artifacts. Ours pulls exactly those generalizable ones into its own numbered list.
- **A single preferred orchestrating command, with a manual per-step fallback documented alongside it.** Sentinelle's `run_pipeline.py` runs everything; if it fails, the file documents exactly which manual scripts to rerun and which artifacts to inspect. We didn't have an equivalent, so `scripts/run_plan.py` was added (see §3.4): it loops `reduce_dim.py` -> `evaluate.py` -> `visualize.py` over every method in a validated plan and writes a `run_log.json`, with the individual scripts still directly callable for redoing one method by hand.
- **A dedicated "Output requirements" section**, separate from tool contracts and the self-check, spelling out exactly what the final deliverable file must contain. We didn't have this pulled out on its own before; it now mirrors the assignment's own rubric language (profile and preprocessing rationale, methods with hyperparameters and reasons, metrics table, visualizations, limitations).
- **A "Final response style" section.** Sentinelle tells the agent exactly how to summarize itself at the end (status, what was decided, validation result, output paths, warnings), concise and factual, "do not claim accuracy you cannot verify." Since a live grading run is literally "capture the agent's output," this is worth having and is now in ours.
- **Formatting discipline** (`---` between sections, bold on the one load-bearing sentence per section) is most of why Sentinelle's file reads as tighter despite being roughly 3x the length of our first draft. Adopted directly.

**One thing deliberately not adopted:** Sentinelle inlines its entire method-selection-equivalent (the MAE/block-averaging interpretation logic) directly into `CLAUDE.md`, which is why the file is 216 lines. biostat-superpowers does the opposite: `CLAUDE.md` stays about rules and sequencing, and all decision logic is deferred to `SKILL.md` files loaded just-in-time (§1, item 2 below). We follow biostat's pattern, not Sentinelle's, here: `plan.json`'s full schema and the actual method/hyperparameter decision logic live in `method-selection/SKILL.md`, not in `CLAUDE.md`, keeping the brain file focused on rules and sequencing and the skill file swappable on its own.
- Redundant dual checklists (Sentinelle has both a "Final self-check" and a near-duplicate "Pre-flight checklist" right after it) are not copied; ours has one.

### 2. SKILL.md anatomy
Frontmatter across all three examples is minimal (`name` + `description`; staix-submission adds a `Triggers:` phrase list inside the description). Bodies consistently separate *what/when* from *how*, and `references/`/`scripts/` are **never preloaded**: biostat-superpowers' skills cite a specific reference or script file inline at the exact workflow step that needs it (e.g. "run `scripts/profile_dataset.py`," then later "read `references/preprocessing_decision_tree.md`"). We should write our skills the same way: numbered steps, each naming the one file it needs, nothing front-loaded.

### 3. Orchestrator routing, the critic, and the profiling/EDA scripts
biostat-superpowers' orchestrator (`skills/biostatistics/SKILL.md`) is a routing table over a "research arc": classify the goal, find the entry point, follow "sequencing rules," loop back to an earlier phase on failure rather than halting ("a map, not a track"). The critic is a hand-off, not a call: "critique before you report, ideally in a fresh-context sub-agent that did not build it," given artifacts only, told to actively try to break the analysis; a `not_ready` verdict blocks shipping. If no subagent mechanism exists, the fallback is "adopt a deliberately fresh, adversarial stance" in the same context, a graceful degradation worth keeping as our own fallback if a subagent call is ever unavailable.

Separately, Sentinelle's `scripts/analyze_schema.py` and `scripts/run_eda.py` (read directly, not part of the original subagent budget) are both thin CLI wrappers: parse arguments, call a library function (`build_schema_summary`, `run_eda`, both defined in `src/io_utils.py`/`src/eda.py`, which were not opened, out of scope as supervised schema-inference logic), then print a concise stdout summary. The one new, reusable detail: `analyze_schema.py` writes **both** a JSON artifact and a companion human-readable Markdown summary (`schema_summary.json` + `.md`). That's now folded into `profiler.py`'s contract in §3.4.

### 4. State persistence and failure handling
No repo uses a database or complex state machine; it's all plain files (JSON/Markdown artifacts) treated as the interface between steps, plus a pass/fail validator script with a written report. Failures are handled as **routing rules** (route back to an earlier skill) or **fallback substitutions** (median baseline, PCA fallback), never bare exceptions that halt the run.

### 5. Mock/synthetic testing
Two patterns, both worth copying structurally: (a) a seeded synthetic-data generator that also writes its own description file, so the full pipeline can be smoke-tested before touching real data (Sentinelle's `create_mock_dataset.py`); (b) a standalone contract-checker script that validates an output's structure/shape and writes a JSON+Markdown pass/fail report (`validate_submission.py`). For us: a Swiss-roll-style generator with known ground-truth manifold structure, and a `validate_plan.py` that checks `plan.json` against a schema the same way.

### 6. What not to copy
Everything tied to a supervised target: median-baseline fallbacks, MAE/block-averaging, `row_id`/target-column submission schemas, GroupKFold-by-period logic, panel/time-series leakage checks, and biostat-superpowers' causal-inference, missing-data, predictive-modeling, and study-design skills (each needs a treatment, outcome, or MCAR/MAR/MNAR theory with no analog in target-less DR). The reusable part is always the *shape* (sequencing, hand-off, critic pattern, file-based state), never the statistical content. Sentinelle shows no visible license; treat it as a pattern to imitate, not code to copy.

### 7. Who writes `DATA_DESCRIPTION.md`
Not the pipeline, in either repo. The STAI-X rules doc, paragraph 66 ("Held-out evaluation procedure"), lists it as something the **organizers** do when setting up a held-out run: "Add `data/DATA_DESCRIPTION.md` to the same folder describing: the training file, the covariates, the validation file with hidden target, and what `submission.csv` must look like." Sentinelle's own `CLAUDE.md` agrees: "Judges clone this repo and place a hidden dataset in `./data`... `./data/DATA_DESCRIPTION.md` is the source of truth." It's an externally-authored input the agent only ever reads, in the same category as the data file itself, never something `scripts/` produces.

For us, that makes `data/pbmc/DATA_DESCRIPTION.md` and `data/pathmnist/DATA_DESCRIPTION.md` hand-written artifacts, not pipeline output. Writing them well matters: they should describe what a domain expert handing off the dataset would know (what it is, its modality, provenance, anything a raw file can't tell you), but must **not** prescribe the analysis itself (no "use UMAP with these settings"), since that would hand the agent its decisions instead of letting it make them, undermining the "quality of the analysis decisions made by the agent" grading criterion. This is now an explicit early build-order item, see §3.7.

### 8. Preflight/doctor pattern and artifact reuse (from int-brain-lab/ibl-ai-agent)
A fourth repo, `int-brain-lab/ibl-ai-agent` (a Claude-Code-driven scientific analysis agent, unrelated to STAI-X), was read directly for two things: how it handles missing dependencies, and how it treats reusable intermediate computations. Its `CLAUDE.md` just points to `AGENTS.md`; the relevant guidance lives in `AGENTS.md`'s "Save intermediate results" and "Installation and preflight" sections plus `skills/install/SKILL.md`.

**Save intermediate results.** `AGENTS.md`: "Keep checkpoints incremental so new computations can be added without rerunning existing ones... If you find yourself running a computation twice, that is a sign you should have saved an artifact." We already save intermediate artifacts (`outputs/<dataset>/*`); what this added was the explicit *reuse* discipline, check for a matching artifact before recomputing, which became CLAUDE.md rule 13 and a corresponding change to `run_plan.py`'s contract (skip a method if its artifact already matches the plan, in §3.4).

**Preflight/doctor.** Two things worth taking:
- **A deterministic diagnostic tool, not ad hoc agent checks.** There's an actual `doctor` CLI subcommand (confirmed via `tests/test_doctor.py`, which asserts it "outputs required schema") that does the environment checking, rather than asking the LLM to freeform-check things with bash each time. `AGENTS.md` just tells the agent to run it and act on the structured result. Same "deterministic script does the work, agent reads the result" principle we already use everywhere else; this became `scripts/doctor.py` (see §3.4).
- **Checks scoped per-activity, not one blanket gate.** Quoting `skills/install/SKILL.md`: "Do not require rendering or publishing tools for local analysis, or an execution environment for conceptual answers." A missing Quarto install doesn't block their local analysis; missing `gh` auth doesn't block local work. Ours: a missing PDF-rendering dependency should only block the report step, never profiling through evaluation, hence `doctor.py --check {core,report,all}` rather than one all-or-nothing check.

**What we didn't take:** their actual response to a failed check is "stop the dependent activity and complete setup interactively," which needs a human present mid-run. That conflicts with the single-prompt autonomous mode built first (§2, §3.7). Adapted instead to: run the check, and if something needed for the *current* step is missing, stop that step and document exactly what's missing in the response (and the report, if far enough along) rather than waiting for anyone or failing silently, the same shape as the existing "data genuinely insufficient to analyze" fallback.

## 2. Constraints carried forward (not re-litigated)

Harness-only (Claude Code + CLAUDE.md, no LangChain/API planner); Claude Code writes `plan.json` with a reason per decision; deterministic parameterized CLI tools execute it; one critique-and-revise loop in a fresh subagent context; `DATA_DESCRIPTION.md` per dataset as source of truth, no dataset-specific code; methods from the approved list, agent picks a subset per dataset; datasets are PathMNIST + PBMC3k (required), a synthetic Swiss-roll-style smoke test, and an unseen third dataset for robustness; deadline Mon Sept 28, 11:59pm EST (today: Mon Sept 21).

Your answers to the three clarifying questions from the first pass shape the design below: (1) grading mode, build the single-prompt/unattended contract as the *real* workflow so a live re-run would work, but treat submitted reports/artifacts as the primary graded evidence; (2) Claude Pro/Max subscription, keep the critic loop to one pass and avoid unnecessary subagent fan-out, since usage is rate-limited, not pay-per-token; (3) no fallback needed for a missing API key, assume Claude Code is available wherever this runs.

From the second pass: the critic loop and the synthetic smoke-test dataset are still part of the target design (they satisfy the "one feedback loop" and "smoke test" goals from the original constraints), but are explicitly deprioritized in build order, see §3.7. They're designed as file-based, pluggable add-ons specifically so deprioritizing them doesn't require re-architecting anything once they do get built.

## 3. Proposed design

### 3.1 Target repo layout

```
dim-reduction-agent/
├── CLAUDE.md                  # NEW, orchestration brain, drafted FIRST (see §3.7)
├── README.md                  # NEW, architecture + tutorial (Award B pattern)
├── report.pdf                 # NEW, manual 4-page report (written last, by hand)
├── .claude/
│   ├── skills/
│   │   ├── data-profiling/SKILL.md        # NEW
│   │   ├── method-selection/SKILL.md      # NEW
│   │   ├── embedding-evaluation/SKILL.md  # NEW, deferred (see §3.7)
│   │   └── reporting/SKILL.md             # NEW
│   └── agents/
│       └── dr-critic.md                   # NEW, deferred (see §3.7), fresh-context critic subagent
├── scripts/                    # RENAMED from agent/, avoids the .claude/agents/ name collision
│   ├── doctor.py                # NEW, deterministic preflight check (core deps / report deps)
│   ├── loaders.py               # KEEP unchanged
│   ├── profiler.py              # KEEP, add JSON+MD CLI mode
│   ├── reduce_dim.py            # NEW, runs one DR method
│   ├── evaluate.py              # NEW, quantitative metrics for one embedding
│   ├── visualize.py             # NEW, scatter plot for one embedding
│   ├── validate_plan.py         # NEW, schema-checks plan.json
│   ├── run_plan.py              # NEW, loops reduce_dim/evaluate/visualize over a validated plan
│   ├── report.py                # NEW, assembles a report PDF from outputs/ into reports/
│   └── make_synthetic.py        # NEW, deferred, Swiss-roll generator + DATA_DESCRIPTION.md
├── data/
│   ├── pbmc/{pbmc3k_raw.h5ad, DATA_DESCRIPTION.md}
│   ├── pathmnist/{pathmnist.npz, DATA_DESCRIPTION.md}
│   ├── swissroll/{DATA_DESCRIPTION.md}          # deferred, generated on demand
│   └── <robustness_dataset>/{DATA_DESCRIPTION.md}  # chosen later, untouched by code
├── outputs/                     # INTERMEDIATE / working artifacts, one subfolder per run
│   └── <dataset>/{profile.json, profile_summary.md, plan.json, embeddings/*.npy,
│                   metrics/*.json, figures/*.png, critique.json, report_draft.md, run_log.json}
├── reports/                     # FINAL deliverables, PDF only, nothing outside this folder
│   ├── pbmc/generated_report_1.pdf
│   └── pathmnist/generated_report_2.pdf
├── docs/{design.md, todo.md}
├── resources/                  # unchanged
└── notebooks/                  # unchanged, manual-only, agent never writes here
```

**Why two folders, `outputs/` vs. `reports/`:** `outputs/<dataset>/` is the working directory the pipeline steps read and write as they run: `profile.json` feeds `method-selection`, `plan.json` feeds `reduce_dim.py`, embeddings feed `evaluate.py`/`visualize.py`, and everything feeds `report.py`. It's intermediate, needed so each step can be tested and re-run independently and so re-running a `plan.json` reproduces results, but it is not itself a deliverable and can be excluded from what you submit. `report.py` reads everything out of `outputs/<dataset>/`, including its own working draft `report_draft.md`, and renders a single **self-contained PDF** straight into `reports/<dataset>/`, with figures embedded in the PDF itself rather than shipped as separate files. Nothing gets copied to the repo root: the dataset subfolder gives the dataset link, and the filename (`generated_report_1.pdf` or `generated_report_2.pdf`) carries the assignment's required name directly. `report.py` takes which number to use as a plain `--report-number {1,2}` CLI argument, so the script itself stays fully dataset-agnostic; the 1-vs-2 assignment (pbmc = 1, pathmnist = 2, picked simply because pbmc is smaller and gets built first) is recorded in the invocation, not hardcoded.

`test_api.py` and the old planner line in `todo.md` are retired per §0, the harness *is* the planner now.

### 3.2 CLAUDE.md outline (headings + key rules only)

Restructured after a close read of Sentinelle's actual `CLAUDE.md` (§1, item 1). Current section order in the real file:

1. **Mission**: identity line, then the mission statement, anchored to a specific dataset always being named in the prompt (e.g. "do the data analysis on data/pbmc"), the two required artifacts for that one dataset, and the "don't stop until both exist" directive with its one narrow exception (genuinely unanalyzable data, documented in the report). Explicitly single-dataset per invocation by design choice: process only the named dataset, never scan for or additionally process others in the same run.
2. **Evaluation setting**: states the possible-hidden-dataset scenario explicitly; `DATA_DESCRIPTION.md` is the one source of truth; never assume a new dataset resembles PBMC3k or PathMNIST.
3. **Non-negotiable rules**: a flat numbered list of general operational rules (read the description first, no hardcoded dataset specifics, labels are eval-only, never touch `data/`, no external data, graceful degradation on a missing library, always validate before executing, always produce the final report, don't stop at intermediate artifacts), plus three rules added after later review passes: never treat an unresolved ambiguity as a settled fact, flag it as a judgment call with a stated reason instead; log a `reason` for every plan entry, including defaults, "standard choice" alone is not an acceptable reason; and reuse a matching existing artifact instead of recomputing it (§1, item 8's "save intermediate results" pattern).
4. **How you're invoked**: the prompt always names exactly one dataset; if it doesn't, stop and say so rather than guessing or defaulting to the first one found. Report-number resolution when the prompt doesn't specify one; what to do if `DATA_DESCRIPTION.md` is missing.
5. **Approved methods**: the method list from the assignment, with the note that not every method needs to run on every dataset.
6. **Required execution sequence**: run `doctor.py --check core` before locating/profiling, locate dataset, profile, consult skills, write `plan.json`, validate, execute (preferred single command `run_plan.py`, which skips methods with a matching existing artifact and prints per-method progress; per-method scripts still directly callable for redoing one method by hand), critique if built, run `doctor.py --check report` before reporting, report, final self-check.
7. **Guard rails**: the DR-specific technical practices that need brief justification, not just a one-liner (t-SNE/UMAP interpretation caution, O(N²) subsampling threshold, shared preprocessing, disconnected neighbor graphs), pointer to §3.5 for the full list.
8. **Fallback rules**: PCA fallback after one retry on manifold-method failure; proceed without the critic if it isn't built; never ship a report with missing figures, NaNs, or a fabricated number; if `doctor.py` reports something missing for the current step, stop that step and document the gap rather than guessing a workaround or waiting.
9. **Tool contracts**: pointer to `scripts/*.py --help`, not full docs inline; includes `run_plan.py`.
10. **Output requirements**: a dedicated section (new, borrowed from Sentinelle's pattern) spelling out exactly what `generated_report_<N>.pdf` must contain, mirroring the assignment's own rubric language.
11. **Skills and subagents index**: which skill or subagent to consult for which decision.
12. **Final self-check checklist**: one checklist, not two; `plan.json` exists and validates; every planned method has a metric and figure or a documented, reasoned skip; critique addressed or explicitly overridden; every report number traces to `metrics/*.json`; seed recorded; every `reason` field is dataset-specific, not a placeholder; exactly the one requested dataset was processed.
13. **Final response style**: a fixed summary format for the agent's own end-of-run report to the user (status, profile highlights, methods run, key results, critique outcome, output path, warnings/fallbacks), concise and factual, no unverified claims. Matters because a live grading run is literally "capture the agent's output."

### 3.3 Skills and subagents

- **`data-profiling`**: wraps `profiler.py`'s output and adds a decision tree for preprocessing: dense vs. sparse, count data vs. continuous, image vs. tabular vs. single-cell, n much greater than p vs. the reverse; recommends log1p/normalization for counts, HVG selection for scRNA-seq, flatten and scale for images, standardization for continuous features. Mirrors biostat-superpowers' reference-decision-tree pattern and Sentinelle's infer-from-free-text-description habit.
- **`method-selection`**: given `profile.json` plus the approved method list, picks 2 to 4 methods with explicit, size/structure-driven criteria (always include PCA as baseline; include one PCA variant when data is sparse/nonlinear; include a manifold method for visualization; prefer Isomap, LLE, Laplacian Eigenmaps, or Diffusion Maps for data with expected manifold structure like the synthetic set) and default hyperparameters (perplexity/`n_neighbors` scaled to N, fixed seed); writes it all into `plan.json` with reasons.
- **`embedding-evaluation`**: defines the quantitative criteria (trustworthiness/continuity, explained variance ratio for PCA, stress for MDS, reconstruction error where defined, silhouette/ARI against labels *as a sanity check only*) and a qualitative checklist (seed/hyperparameter stability, artifact-vs-real-structure heuristics for t-SNE/UMAP); the DR-specific analog of biostat-superpowers' evaluation rubric.
- **`reporting`**: assembles the report from a template (summary, profile, preprocessing rationale, methods with hyperparameters and reasons, metrics table, figures, critique-and-revision log, limitations); borrows the reproducibility half of biostat-superpowers' reporting skill (seed, environment, one-command reproduction) and drops the clinical-trial-specific content.
- **`dr-critic` subagent** (deferred, see §3.7): invoked in a fresh context with only `plan.json`, `metrics/*.json`, and figure paths, never the builder's reasoning, instructed to actively try to break the analysis: check for over-read t-SNE/UMAP cluster geometry, instability across seeds, preprocessing mismatches between methods, unjustified hyperparameters, and any label leakage into the unsupervised path. Returns a structured critique (severity-tagged findings plus a readiness rating) that the main agent must act on exactly once, per the one-feedback-loop constraint.

### 3.4 CLI tool contracts

| Tool | Inputs | Output |
|---|---|---|
| `doctor.py` | `--check {core,report,all}` (default `all`) | pass/fail plus a JSON+Markdown report; `core` checks required packages import plus `data/<dataset>/DATA_DESCRIPTION.md` exists, `report` checks the report-rendering dependency separately so a missing one never blocks profiling through evaluation. Adapted from `int-brain-lab/ibl-ai-agent`'s `doctor` CLI subcommand (§1, item 8), scoped per-activity rather than one blanket gate |
| `profiler.py` | `--dataset <name>` (via `loaders.LOADERS`) | `outputs/<dataset>/profile.json` plus `profile_summary.md`: n_samples, n_features, sparsity, missing, value range, mean/std, has_labels/n_classes, metadata |
| `reduce_dim.py` | `--dataset`, `--method`, `--params '<json>'`, `--seed`; applies the preprocessing block from `plan.json`, never re-decides it | `outputs/<dataset>/embeddings/<method>.npy` (n_samples by n_components) plus a sidecar JSON of actual params used and runtime |
| `evaluate.py` | `--embedding`, `--X` (preprocessed data), optional `--labels` | `outputs/<dataset>/metrics/<method>.json`: method-appropriate quantitative scores; labels used for scoring only, never accepted by `reduce_dim.py` |
| `visualize.py` | `--embedding`, optional `--labels`, `--title` | `outputs/<dataset>/figures/<method>.png`: 2D scatter, colored by label if present else density |
| `validate_plan.py` | `plan.json` | pass/fail plus a JSON+Markdown report (methods in allowed list, hyperparameters in safe ranges, preprocessing steps recognized); mirrors `validate_submission.py` |
| `run_plan.py` | `outputs/<dataset>/plan.json` (already validated) | loops `reduce_dim.py` -> `evaluate.py` -> `visualize.py` over every method already in the plan (never adds to it); skips a method if a matching artifact already exists (same method/hyperparameters/seed, per the reuse rule in §1 item 8); prints a short progress line per method as it runs; applies the disconnected-graph retry and PCA fallback per method rather than aborting the run; writes `outputs/<dataset>/run_log.json` (per-method status: ok / reused / fell back / skipped, with why). The single preferred command for step 6 of the build order and of CLAUDE.md's execution sequence; individual per-method scripts stay directly callable for redoing just one method |
| `report.py` | `outputs/<dataset>/{profile.json, plan.json, metrics/*.json, figures/*.png, critique.json}` (critique optional), `--report-number {1,2}` | `reports/<dataset>/generated_report_<N>.pdf` only, figures embedded inline, self-contained |
| `make_synthetic.py` (deferred) | `--n-samples`, `--seed` | Swiss-roll-style dataset plus its own `DATA_DESCRIPTION.md`, for smoke-testing before real data |

**`plan.json` schema (sketch):**

```json
{
  "dataset": "pbmc",
  "preprocessing": [
    {"step": "normalize_total", "reason": "..."},
    {"step": "log1p", "reason": "..."},
    {"step": "select_hvg", "params": {"n_top_genes": 2000}, "reason": "..."}
  ],
  "methods": [
    {"name": "pca", "hyperparameters": {"n_components": 50}, "role": "baseline", "reason": "..."},
    {"name": "umap", "hyperparameters": {"n_neighbors": 15, "min_dist": 0.1, "random_state": 0}, "role": "primary_visualization", "reason": "..."}
  ],
  "evaluation": {"quantitative": ["trustworthiness", "explained_variance_ratio"], "qualitative": ["seed_stability_check"]},
  "seed": 0,
  "revision": 1,
  "critique_applied": null
}
```
After the one revision: `revision: 2`, `critique_applied` records what changed and why.

### 3.5 DR-specific guard rails

- Never interpret t-SNE/UMAP inter-cluster distances or cluster sizes as meaningful; require at least two seeds or two hyperparameter settings before trusting a shape claimed in the report.
- MDS/Isomap are O(N squared) or worse; subsample above a stated N threshold and record that decision as a `plan.json` reason, don't do it silently.
- Labels are evaluation-only by construction: `reduce_dim.py`'s CLI has no path for labels to enter as a feature; only `evaluate.py`/`visualize.py` accept `--labels`.
- Fixed seeds everywhere (`--seed`) so re-running `plan.json` reproduces results exactly.
- One shared preprocessing block per plan, applied identically before every method, no per-method silent re-normalization.
- Isomap/LLE/Laplacian Eigenmaps neighbor graphs can be disconnected; `reduce_dim.py` must catch that explicitly, retry once with a larger `n_neighbors`, else skip and log a documented failure rather than crash the run.
- GPLVM is a stretch goal; timebox it, skip and document if it threatens the schedule.

### 3.6 Evaluation strategy

**Quantitative:** trustworthiness/continuity (label-free neighborhood preservation), PCA explained variance ratio, MDS stress, reconstruction error where defined; silhouette/ARI against known labels only as a clearly-flagged supervised sanity check, never used to tune hyperparameters.
**Qualitative:** the `embedding-evaluation` skill's checklist (stability across seeds/hyperparameters, correspondence between visual clusters and any available label/metadata, t-SNE/UMAP artifact heuristics), enforced adversarially by `dr-critic` once it exists, which returns a severity-tagged findings list plus a readiness rating. That rating, not just the raw numbers, is how the agent judges whether an embedding is reportable.

### 3.7 Build order: modules first, calendar second

Today is Mon 9/21; the deadline is Mon 9/28, 11:59pm EST, one week. Priority is a working end-to-end pipeline (profile, plan, reduce, evaluate, visualize, report) on real data as soon as possible today, then extend to the second dataset, then refine. The critic and the synthetic smoke test are deliberately last and optional: both are designed to only read/write files that already exist in `outputs/<dataset>/`, so they're pure additions later, nothing built earlier has to change to accommodate them, and the pipeline is fully gradable without them.

**Step 0, before any tool code: draft `CLAUDE.md` and the skill stubs, and hand-write both `DATA_DESCRIPTION.md` files.** Write the full CLAUDE.md text (not just the outline) using §3.2 and §3.3 as the starting point, including `data-profiling` and `method-selection` at least in draft form. This is the design compass and there's no reason to wait for the tools to exist first; the tool-invocation *details* (exact flags, JSON shapes) get filled in and corrected incrementally as each script below is actually built, with a final consistency pass right before the first live single-prompt test (step 10). Also write `data/pbmc/DATA_DESCRIPTION.md` and `data/pathmnist/DATA_DESCRIPTION.md` by hand here (see §1, item 7): these are authored inputs, not pipeline output, and are needed before step 10's live run can be tested at all. `git mv agent scripts` also happens here. Also build `scripts/doctor.py --check core` here (verify required packages import, confirm `data/<dataset>/DATA_DESCRIPTION.md` exists): it's small, standalone, and needed by step 10's very first move. `--check report` gets added at step 8, once the PDF library is picked (see §3.8).

Build order from here, each step named with how to test it standalone before moving to the next:

1. **`profiler.py` JSON+MD mode** (extend the existing function with a CLI that writes to file). *Test:* `python scripts/profiler.py --dataset pbmc --out outputs/pbmc/profile.json` and the same for pathmnist; open both files and sanity-check the numbers (sparsity high for scRNA-seq counts, near zero for images; no unexpected NaNs). Update CLAUDE.md's data-profiling section with the real command syntax. (Note: the dataset identifier is `pbmc`, matching `data/pbmc/`/`outputs/pbmc/`, not `pbmc3k`; `loaders.LOADERS` is keyed by the folder-convention name, not the upstream dataset's own name, since the two differ here.)
2. **`plan.json` schema plus one hand-written plan** for pbmc (PCA plus one manifold method, e.g. UMAP). No agent involved yet, you write this file by hand purely to unblock steps 3 to 7. *Test:* it parses as JSON and matches the shape in §3.4 by eye.
3. **`validate_plan.py`**. *Test:* run it against the valid hand-written plan (passes) and against a deliberately broken copy: unknown method name, missing `reason`, out-of-range hyperparameter (fails with a specific, readable error).
4. **`reduce_dim.py`**. *Test:* run it for each method in the hand-written plan; check each `embeddings/<method>.npy` shape is `(n_samples, n_components)`; run the identical command twice with the same `--seed` and diff the two files, they must be byte-identical (reproducibility).
5. **`evaluate.py`**. *Test:* run against the embeddings from step 4; sanity-check ranges (PCA explained variance ratio in [0,1] and non-decreasing with `n_components`; trustworthiness in [0,1]).
6. **`visualize.py`**. *Test:* run against the same embeddings and actually open the PNGs; PBMC3k's UMAP should show visibly separated groups, its PCA should look blobbier. This is the first real correctness check on the whole numeric pipeline, not just shapes.
7. **`run_plan.py`**, a thin composition of steps 4 to 6 over every method in a plan, plus `outputs/<dataset>/run_log.json`. *Test:* run it against the hand-written plan and diff its `embeddings/`/`metrics/`/`figures/` output against what steps 4 to 6 produced by hand, they must match. Then deliberately break one method's params in a copy of the plan and confirm `run_plan.py` logs the failure and falls back to PCA for that slot instead of aborting the whole run (this is where the Fallback rules in CLAUDE.md get their first real test). Then rerun it unchanged and confirm every method gets skipped as "reused" in `run_log.json` rather than recomputed (the reuse rule from §1 item 8), and that it prints a progress line per method as it goes.
8. **Pick a PDF library, then `report.py`**, writing into `reports/<dataset>/generated_report_<N>.pdf`, and finish `scripts/doctor.py --check report` for whichever library gets picked (see §3.8). *Test:* assemble a report from the artifacts for pbmc with `--report-number 1`; check every number quoted in the report text traces to a real value in `outputs/pbmc/metrics/*.json` (no fabrication), and that the PDF renders with figures embedded, not just referenced by path.
9. **Repeat steps 1 to 8 for pathmnist** (`--report-number 2`), reusing the exact same scripts plus a second hand-written `plan.json`. This is the real test of "no dataset-specific code downstream of `loaders.py`," if any script needs an `if dataset == ...` branch to work, treat that as a design bug and fix it before moving on. Do the CLAUDE.md consistency pass here too.
10. **First real single-prompt Claude Code run.** Wire Claude Code to do steps 1 to 4 itself (profile a dataset, write its own `plan.json` with reasons) and then call the already-hand-tested CLI tools from steps 5 to 8. *Test:* one single-prompt session ("Do the data analysis on data/pbmc") with no hand-holding; confirm it produces a valid `plan.json`, runs `run_plan.py`, and a report lands in `reports/pbmc/` without you hand-writing any file.
11. **Same single-prompt test on pathmnist.** Once both real datasets pass step 10 unattended, the system is working end to end and gradable; everything from here is refinement, not a blocker.

**Deferred / optional, built to be pluggable, not on the critical path:**
- **`dr-critic` subagent plus one revision loop.** It only reads `outputs/<dataset>/{plan.json, metrics/*.json, figures/*.png}` and writes `outputs/<dataset>/critique.json`; CLAUDE.md's corresponding step is written as "if this stage hasn't been built yet or produces nothing, proceed straight to reporting," so the core loop never depends on it existing. Add once the core loop (steps 1 to 11) is solid and only if time remains.
- **`make_synthetic.py`** (Swiss-roll smoke test). Useful for validating manifold methods against known ground truth, but not required for the two graded datasets; build last, as a third dataset that plugs into the same `loaders.py`/profiler/plan interface once it exists, no changes needed elsewhere.

Rough calendar for the remaining days once the module order above is done: report writing, the robustness dataset, README, manual `report.pdf`, and final QA fill the rest of the week, in whatever order the actual pace on Mon to Tue dictates; worth revisiting once steps 1 to 11 are actually done rather than fixing in advance.

### 3.8 Risks and open questions

- **Scaling:** PathMNIST has far more samples than PBMC3k; O(N squared) methods need a subsampling threshold decided and documented, not discovered mid-run.
- **Usage budget:** two full autonomous runs, a deferred critic pass on each, a synthetic smoke test, and a third robustness dataset could brush against Pro/Max weekly limits. Since the critic and synthetic set are deferred anyway (§3.7), the immediate budget risk is just the two required unattended runs (steps 10 to 11); re-check limits before adding the deferred pieces back in.
- **PDF generation:** no PDF library is confirmed installed yet; matplotlib can emit PDF figures, but `generated_report_<N>.pdf` needs a text-plus-figure layout tool (e.g. weasyprint, reportlab, or a markdown-to-pandoc path); pick and test this at step 8, before it's needed for real reports, not after. Whatever gets picked is also what `scripts/doctor.py --check report` verifies.
- **Why the single-prompt contract has to actually work, not just look like it does:** because you said to design for the possibility of a live rerun rather than assuming only the pre-made reports get graded, the CLAUDE.md workflow needs to survive real failures unattended (a method crashing, a disconnected neighbor graph, an odd dataset) without you there to fix things up afterward. That's the whole reason the fallback rules and final self-check checklist get real engineering effort instead of being a nice-to-have; if you were certain only the static reports mattered, you could just rerun by hand until they looked right and skip most of that robustness work.
- **GPLVM:** no lightweight scikit-learn implementation; likely skip with a documented reason rather than force it in.
- **PathMNIST dimensionality:** flattened images are 28x28x3 = 2352 features; a PCA pre-reduction before t-SNE/UMAP may be needed for speed; decide whether that's a `plan.json` preprocessing step or a chained method, and be consistent about it.
- **Version pinning:** t-SNE/UMAP results are somewhat version-sensitive; record the installed versions from §0 in `README.md`/`requirements.txt` for reproducibility.
- **Third dataset choice:** true robustness testing means picking it late, only after the core two-dataset loop works, and not tuning code around it in advance; worth confirming this is really what "unseen" should mean here versus picking it now for convenience.
