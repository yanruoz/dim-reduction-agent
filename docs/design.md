# Design: AI Agent for Dimension Reduction and EDA

Research-and-design document for BIOS774 Project 1. No pipeline code has been written; this is the plan.

## 0. Project audit

**Environment** (`venv/`, Python 3.13): numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scanpy 1.12.4, umap-learn 0.5.12, matplotlib 3.11.1, medmnist 3.0.2, anndata 0.13.3.post0, all already installed. Nothing needs installing to start. (`python-docx` was needed to read the STAI-X rules doc; it lives in the system Anaconda Python, not the venv, and was only used read-only for this research, not a project dependency.)

**Existing code:**

| File | What it does | Keep? |
|---|---|---|
| `agent/loaders.py` | Originally loaded pbmc3k / pathmnist through a `LOADERS` registry of per-dataset functions. Now a single generic loader for any `data/<name>/` folder (csv/tsv/txt, npy, npz, h5ad), guided by an optional `## Loading` section in that dataset's `DATA_DESCRIPTION.md`; pbmc and pathmnist get no special treatment. Downloading moved to a separate `scripts/fetch_data.py`, so the pipeline itself never touches the network. | **Keep, move to `scripts/loaders.py`** (see naming fix below). Exactly the right shape for a harness-driven agent. |
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

**Addendum, generic loader.** `loaders.py` originally only knew the two registered datasets, so a dataset never seen while building the pipeline could not even start. It now falls back to a generic loader (csv/tsv/txt, npy, npz, h5ad) for any `data/<name>/` folder, guided by an optional `## Loading` section in `DATA_DESCRIPTION.md` (`file`, `label_column`, `id_column`, `delimiter`, `x_key`, `y_key`). That section states where data lives, not what analysis to run, so it doesn't conflict with the rule above. Design choices taken from the reference repos (Sentinelle's `io_utils.py`, biostat-superpowers' `profile_dataset.py`): encoding fallback, delimiter sniffing, and failing loudly with a specific message instead of guessing; left behind: everything about targets, leakage, and join keys. Two rules of our own: a label column is never guessed, and a misspelled setting is an error rather than silently ignored. Tests: `python -m unittest discover -s tests`.

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
├── requirements.txt            # DONE, direct dependency versions for reproducibility
├── report.pdf                 # NEW, manual 4-page report (written last, by hand)
├── .claude/
│   ├── skills/
│   │   ├── data-profiling/SKILL.md        # NEW, core, not yet written despite step 0 saying so
│   │   ├── method-selection/SKILL.md      # NEW, core, not yet written despite step 0 saying so
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
│                   metrics/*.json, figures/*.png, findings.md, critique.json, run_log.json}
├── reports/                     # FINAL deliverables, PDF only, nothing outside this folder
│   ├── pbmc/generated_report_1.pdf
│   └── pathmnist/generated_report_2.pdf
├── docs/{design.md, todo.md}
├── resources/                  # unchanged
└── notebooks/                  # unchanged, manual-only, agent never writes here
```

**Why two folders, `outputs/` vs. `reports/`:** `outputs/<dataset>/` is the working directory the pipeline steps read and write as they run: `profile.json` feeds `method-selection`, `plan.json` feeds `reduce_dim.py`, embeddings feed `evaluate.py`/`visualize.py`, and everything feeds `report.py`. It's intermediate, needed so each step can be tested and re-run independently and so re-running a `plan.json` reproduces results, but it is not itself a deliverable and can be excluded from what you submit. `report.py` reads everything out of `outputs/<dataset>/`, including `findings.md` (the one part of the report the agent writes rather than generates; see `.claude/skills/reporting/SKILL.md`), and renders a single **self-contained PDF** straight into `reports/<dataset>/`, with figures embedded in the PDF itself rather than shipped as separate files. Nothing gets copied to the repo root: the dataset subfolder gives the dataset link, and the filename (`generated_report_1.pdf` or `generated_report_2.pdf`) carries the assignment's required name directly. `report.py` takes which number to use as a plain `--report-number {1,2}` CLI argument, so the script itself stays fully dataset-agnostic; the 1-vs-2 assignment (pbmc = 1, pathmnist = 2, picked simply because pbmc is smaller and gets built first) is recorded in the invocation, not hardcoded.

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
- **`reporting`**: assembles the report from a template (summary, profile, preprocessing rationale, methods with hyperparameters and reasons, metrics table, figures, critique-and-revision log, limitations); borrows the reproducibility half of biostat-superpowers' reporting skill (seed, environment, one-command reproduction) and drops the clinical-trial-specific content.

**No separate `embedding-evaluation` skill** (removed; it was inconsistently marked "deferred" in an earlier pass despite supporting a required assignment step, evaluate the embeddings, not an optional one). Its intended job splits cleanly across things that already exist or are already planned: *which metric applies to which method* (trustworthiness for any embedding, explained variance for PCA, stress for MDS, etc.) is a fixed mapping, not a live judgment call, so it's deterministic logic inside `evaluate.py` itself, the same pattern `reduce_dim.py`'s method registry already uses. *Qualitative caution* (don't over-read t-SNE/UMAP geometry, check seed stability before trusting a structural claim) is already in CLAUDE.md's Guard rails, not something that needed its own skill file. *Interpreting and writing up* the resulting numbers for a human reader is the `reporting` skill's job. Net effect: one fewer skill file, no loss of capability.
- **`dr-critic` subagent** (deferred, see §3.7): invoked in a fresh context with only `plan.json`, `metrics/*.json`, and figure paths, never the builder's reasoning, instructed to actively try to break the analysis: check for over-read t-SNE/UMAP cluster geometry, instability across seeds, preprocessing mismatches between methods, unjustified hyperparameters, and any label leakage into the unsupervised path. Returns a structured critique (severity-tagged findings plus a readiness rating) that the main agent must act on exactly once, per the one-feedback-loop constraint.

### 3.4 CLI tool contracts

| Tool | Inputs | Output |
|---|---|---|
| `doctor.py` | `--check {core,report,all}` (default `all`) | pass/fail plus a JSON+Markdown report; `core` checks required packages import plus `data/<dataset>/DATA_DESCRIPTION.md` exists, `report` checks the report-rendering dependency separately so a missing one never blocks profiling through evaluation. Adapted from `int-brain-lab/ibl-ai-agent`'s `doctor` CLI subcommand (§1, item 8), scoped per-activity rather than one blanket gate |
| `profiler.py` | `--dataset <name>` (a folder under `data/`) | `outputs/<dataset>/profile.json` plus `profile_summary.md`: n_samples, n_features, sparsity, missing, value range, mean/std, has_labels/n_classes, metadata |
| `reduce_dim.py` | `--dataset`, `--method`, `--params '<json>'`, `--seed`; applies the preprocessing block from `plan.json`, never re-decides it | `outputs/<dataset>/embeddings/<method>.npy` (n_samples by n_components) plus a sidecar JSON of actual params used and runtime |
| `evaluate.py` | `--dataset`, `--method` (optional `--embedding`/`--out` overrides) | `outputs/<dataset>/metrics/<method>.json`: trustworthiness (subsampled above the same 5,000-sample threshold as the O(N²) methods, since sklearn's trustworthiness is itself O(N²), documented when it happens), method-specific diagnostics pulled from `reduce_dim.py`'s sidecar JSON, and a labels-based sanity check if labels exist. Reconstructs the preprocessed data itself via `plan.json` + `loaders.py`/`reduce_dim.py`'s `apply_preprocessing`, no `--X` file needed; labels used for scoring only, never accepted by `reduce_dim.py` |
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
    {"step": "select_top_variance", "params": {"n_top_features": 2000}, "reason": "..."}
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

**Quantitative:** trustworthiness for every method (label-free neighborhood preservation; sklearn only provides trustworthiness, not its dual "continuity," so that's the one actually computed; subsampled above 5,000 samples since it's itself O(N²), same threshold as the other large-N guard rails), PCA-family explained variance ratio and the 90%-variance diagnostics already built into `reduce_dim.py`, MDS stress from `reduce_dim.py`'s sidecar; silhouette against known labels (not ARI, which would need an actual clustering step this project doesn't have) only as a clearly-flagged supervised sanity check, never used to tune hyperparameters. Reconstruction error was considered and skipped, computing it would need the fitted model object, which isn't saved, not worth the added complexity given what explained-variance already conveys.
**Qualitative:** CLAUDE.md's Guard rails checklist (stability across seeds/hyperparameters, correspondence between visual clusters and any available label/metadata, t-SNE/UMAP artifact heuristics), enforced adversarially by `dr-critic` once it exists, which returns a severity-tagged findings list plus a readiness rating. That rating, not just the raw numbers, is how the agent judges whether an embedding is reportable.

### 3.7 Build order: modules first, calendar second

**Status as of Thu 9/24** (deadline Mon 9/28, 11:59pm EST, 4 days left including today): steps 0 to 4 below are done, though step 0's skill-drafting sub-item did not actually happen when it was written, `.claude/` does not exist on disk yet, that gap is now step 5 below, moved back onto the critical path where it belongs. `requirements.txt` is also done (direct dependency versions, captured 9/24, see §0). Everything from step 5 onward is still ahead: two skill files, `evaluate.py`, `visualize.py`, `run_plan.py`, a PDF library plus `report.py`, repeating the deterministic pipeline for pathmnist, `README.md`, two live single-prompt runs, and the manual `report.pdf`, which by nature can't start until both datasets' results exist. That's a full agenda for 4 days; worth re-checking pace after every couple of steps rather than assuming the original one-week estimate still holds.

Priority is a working end-to-end pipeline (profile, plan, reduce, evaluate, visualize, report) on real data, then extend to the second dataset, then refine. The critic and the synthetic smoke test are deliberately last and optional: both are designed to only read/write files that already exist in `outputs/<dataset>/`, so they're pure additions later, nothing built earlier has to change to accommodate them, and the pipeline is fully gradable without them.

**Step 0 (done, with one correction): drafted `CLAUDE.md`, hand-wrote both `DATA_DESCRIPTION.md` files, `git mv agent scripts`, built `scripts/doctor.py --check core`.** The skill stubs mentioned in the original version of this step were **not actually written**; that work is now explicit step 5 below instead of silently assumed done. `--check report` still waits on step 9, once the PDF library is picked (see §3.8).

Build order from here, each step named with how to test it standalone before moving to the next:

1. **`profiler.py` JSON+MD mode** (done). *Test (passed):* `python scripts/profiler.py --dataset pbmc --out outputs/pbmc/profile.json` and the same for pathmnist; sparsity high for scRNA-seq counts, near zero for images, no unexpected NaNs. (The dataset identifier is the folder name under `data/`, i.e. `pbmc`.)
2. **`plan.json` schema plus one hand-written plan** for pbmc (done, PCA plus UMAP).
3. **`validate_plan.py`** (done). *Test (passed):* valid plan passes; a deliberately broken copy (unknown method, missing reason, out-of-range hyperparameter) fails with specific errors.
4. **`reduce_dim.py`** (done, full method registry, not just the two in the hand-written plan). *Test (passed):* correct shapes, byte-identical reproducibility across repeated runs, the other 6 real methods smoke-tested on a throwaway array.
5. **Write `data-profiling/SKILL.md` and `method-selection/SKILL.md` in full** (not stubs). This is the piece step 0 was supposed to cover and didn't. Now is a good point to do it properly: `plan.json`'s real schema, `reduce_dim.py`'s actual preprocessing registry (`normalize_total`, `log1p`, `select_top_variance`, `standardize`, `scale_unit_range`) and method registry are both concrete now, so the skills can reference real contracts instead of guessing ahead of the code. *Test:* read them yourself and check that following them, by hand, for pbmc would produce something close to the existing hand-written `plan.json`; then do the same mentally for pathmnist (different modality, different preprocessing needs) as a sanity check that the reasoning generalizes, not just fits the one dataset already in hand.
6. **`evaluate.py`.** Include the method-to-metric mapping directly in its code (trustworthiness for any embedding; explained variance ratio for PCA; stress for MDS; etc., see §3.3's note on why this isn't a separate skill). *Test:* run against the pbmc embeddings from step 4; sanity-check ranges (PCA explained variance ratio in [0,1] and non-decreasing with `n_components`; trustworthiness in [0,1]).
7. **`visualize.py`.** *Test:* run against the same embeddings and actually open the PNGs; PBMC3k's UMAP should show visibly separated groups, its PCA should look blobbier.
8. **`run_plan.py`**, a thin composition of steps 4, 6, 7 over every method in a plan, plus `outputs/<dataset>/run_log.json`. *Test:* run it against the hand-written plan and diff its output against what steps 4/6/7 produced by hand, they must match. Deliberately break one method's params in a copy of the plan and confirm it logs the failure and falls back to PCA for that slot instead of aborting. Rerun unchanged and confirm every method gets skipped as "reused," not recomputed, with a progress line printed per method.
9. **Pick a PDF library, then `report.py`**, writing into `reports/<dataset>/generated_report_<N>.pdf`, and finish `scripts/doctor.py --check report` for whichever library gets picked (see §3.8). *Test:* assemble a report from the pbmc artifacts with `--report-number 1`; every quoted number traces to a real `metrics/*.json` value; the PDF renders with figures embedded, not just referenced by path.
10. **Repeat steps 1 to 9 for pathmnist** (`--report-number 2`), reusing the exact same scripts and skills plus a second hand-written `plan.json`. Real test of "no dataset-specific code downstream of `loaders.py`"; if any script needs an `if dataset == ...` branch, that's a design bug to fix, not a shortcut to take.
11. **`README.md`**: architecture, how the agent makes decisions, and a tutorial for running it, once the deterministic pipeline (steps 1 to 10) is stable for both datasets. Doesn't need the live single-prompt tests to be done first, but should get a final consistency pass after step 13, once everything's actually confirmed working, rather than being written once and never revisited.
12. **First real single-prompt Claude Code run**, on pbmc. Wire Claude Code to do steps 1 to 5's decisions itself (profile, consult the now-real skills, write its own `plan.json`) and call the already-hand-tested tools from steps 6 to 9. *Test:* one single-prompt session ("Do the data analysis on data/pbmc") with no hand-holding; confirm a valid `plan.json`, a `run_plan.py` execution, and a report landing in `reports/pbmc/` without you hand-writing any file.
13. **Same single-prompt test on pathmnist.** Once both pass unattended, the system is working end to end and gradable.
14. **Manual `report.pdf`.** Can't meaningfully start before step 13, since it requires "experimental results on all datasets." Architecture, decision-making process, tools used, results on both datasets, strengths and limitations, at most 4 pages. This is a separately graded, required deliverable in its own right, not a wrap-up afterthought, budget real time for it.

**Deferred / optional, built to be pluggable, not on the critical path:**
- **`dr-critic` subagent plus one revision loop.** It only reads `outputs/<dataset>/{plan.json, metrics/*.json, figures/*.png}` and writes `outputs/<dataset>/critique.json`; CLAUDE.md's corresponding step is written as "if this stage hasn't been built yet or produces nothing, proceed straight to reporting," so the core loop never depends on it existing. Add once the core loop (steps 1 to 13) is solid and only if time remains.
- **`make_synthetic.py`** (Swiss-roll smoke test) and the **third robustness dataset**. Both useful (the former for validating manifold methods against known ground truth, the latter for the "robustness on different datasets" grading criterion), neither required to meet the assignment's "at least two real datasets" minimum. Only attempt after step 14, if time remains.

### 3.8 Risks and open questions

- **Scaling:** PathMNIST has far more samples than PBMC3k; O(N squared) methods need a subsampling threshold decided and documented, not discovered mid-run.
- **Usage budget:** two full autonomous runs, a deferred critic pass on each, a synthetic smoke test, and a third robustness dataset could brush against Pro/Max weekly limits. Since the critic and synthetic set are deferred anyway (§3.7), the immediate budget risk is just the two required unattended runs (steps 12 to 13); re-check limits before adding the deferred pieces back in.
- **PDF generation:** no PDF library is confirmed installed yet; matplotlib can emit PDF figures, but `generated_report_<N>.pdf` needs a text-plus-figure layout tool (e.g. weasyprint, reportlab, or a markdown-to-pandoc path); pick and test this at step 9, before it's needed for real reports, not after. Whatever gets picked is also what `scripts/doctor.py --check report` verifies.
- **Why the single-prompt contract has to actually work, not just look like it does:** because you said to design for the possibility of a live rerun rather than assuming only the pre-made reports get graded, the CLAUDE.md workflow needs to survive real failures unattended (a method crashing, a disconnected neighbor graph, an odd dataset) without you there to fix things up afterward. That's the whole reason the fallback rules and final self-check checklist get real engineering effort instead of being a nice-to-have; if you were certain only the static reports mattered, you could just rerun by hand until they looked right and skip most of that robustness work.
- **GPLVM:** no lightweight scikit-learn implementation; likely skip with a documented reason rather than force it in.
- **PathMNIST dimensionality:** flattened images are 28x28x3 = 2352 features; a PCA pre-reduction before t-SNE/UMAP may be needed for speed; decide whether that's a `plan.json` preprocessing step or a chained method, and be consistent about it.
- **Version pinning:** t-SNE/UMAP results are somewhat version-sensitive; record the installed versions from §0 in `README.md`/`requirements.txt` for reproducibility.
- **Third dataset choice:** true robustness testing means picking it late, only after the core two-dataset loop works, and not tuning code around it in advance; worth confirming this is really what "unseen" should mean here versus picking it now for convenience.
- **Categorical (non-numeric) features are not supported, by decision, at lower priority.** The generic loader refuses any dataset with non-numeric feature columns and says which ones, rather than integer-coding them (which would treat category codes as continuous values, wrong for distance-based methods) or dropping them silently. Datasets with categorical columns therefore can't be analyzed yet, and the manual report should list this under limitations. If it's picked up later, the reference-repo research points to: one-hot for low-cardinality columns (about 20 levels or fewer) with rare levels grouped into "other"; an explicit "missing" level instead of mode-imputation; high-cardinality, id-like, free-text, and datetime columns excluded and recorded rather than encoded; no frequency or target encoding (they invent an ordering or need a target); and never flagging a numeric column as id-like from uniqueness alone. Even then, one-hot plus standardization over-weights many-level columns, and methods built for mixed data (Gower distance with MDS, MCA/FAMD) would be the better fit.
