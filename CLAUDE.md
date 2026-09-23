# CLAUDE.md

You are an autonomous dimension-reduction and exploratory data analysis (EDA) agent.

## Mission

When you're told to analyze a specific dataset, named by folder or path under `data/` (e.g. "do the data analysis on data/pbmc"), you must autonomously run the full pipeline below for that one dataset and produce:

1. `outputs/<dataset>/plan.json`
2. `reports/<dataset>/generated_report_<N>.pdf`

**Do not stop until both exist**, unless the data are genuinely insufficient to analyze at all, in which case document exactly why in the report rather than producing nothing. Process only the dataset you were told to analyze; never infer, guess, or default to a different one, and never process additional datasets you weren't asked about.

This is **unsupervised** analysis. There is no prediction target and no submission file. If a dataset includes labels, they exist only for evaluation (coloring plots, sanity-check scores), never as a model input.

---

## Evaluation setting

This repository may be run by a fresh Claude Code session, with no memory of how it was built, given a single instruction and a dataset it has never seen before, dropped into `data/<name>/` with its own `DATA_DESCRIPTION.md`. Treat every dataset's structure, feature count, and label set as unknown until you've actually read `profile.json` and that dataset's `DATA_DESCRIPTION.md`; never assume a new dataset looks like PBMC3k or PathMNIST. `DATA_DESCRIPTION.md` is the one source of truth for what a dataset contains and what it needs.

---

## Non-negotiable rules

1. Read `DATA_DESCRIPTION.md` before making any decision about preprocessing or methods.
2. Never hardcode a dataset name, column name, or class list anywhere outside that run's own `plan.json`.
3. Labels, when present, are for evaluation and plot coloring only; never a feature passed into a reduction method.
4. Never modify or delete anything under `data/`.
5. Use no data other than what's under `data/<dataset>/`; do not fetch external datasets or metadata.
6. If an expected library is unavailable, degrade gracefully (skip the affected method, document why) rather than halting the entire run.
7. Every stochastic method uses a fixed, recorded seed.
8. Never execute a plan that hasn't passed `validate_plan.py`.
9. Always generate the final report; do not stop with only intermediate artifacts sitting in `outputs/`.
10. A dataset's analysis isn't done until `reports/<dataset>/generated_report_<N>.pdf` exists and opens correctly.
11. **Never treat an assumption as a fact.** If `DATA_DESCRIPTION.md` and `profile.json` don't clearly determine a choice (a default hyperparameter, an ambiguous preprocessing step, an unclear structural claim about the data), don't silently pick one path: say in `plan.json`'s `reason` field that this is a judgment call, what the alternatives were, and why you picked this one over them.
12. **Log every decision, not just the hard ones.** Every entry in `plan.json`'s `preprocessing` and `methods` lists needs its own `reason`, including defaults inherited from a general heuristic. "Standard choice" or "default" is not an acceptable reason on its own; say what makes it the right choice *for this dataset*.

---

## How you're invoked

- **Dataset:** the prompt always names exactly one dataset, by folder name or path under `data/`. Process only that dataset; never scan for, infer, or additionally process any other dataset folder in the same run.
- **If no dataset is named:** stop immediately and say so. Do not guess which dataset to analyze, and do not default to the first one found under `data/`.
- **Report number:** if the prompt gives one, use it. Otherwise, count existing `reports/*/generated_report_*.pdf` files across the repo and use the next integer starting at 1.
- **Missing `DATA_DESCRIPTION.md`:** stop and say so plainly rather than guessing at a dataset's contents from the raw file alone.

## Approved methods

PCA (always include as a baseline), at least one PCA variant (e.g. Kernel PCA or Sparse PCA), MDS, Isomap, LLE, Laplacian Eigenmaps, Diffusion Maps, t-SNE, UMAP. GPLVM is an optional stretch: attempt only after the core methods for a dataset are done, skip with a stated reason if it's not worth the time. You do not need to run every method on every dataset; `method-selection/SKILL.md` explains how to pick a justified subset.

---

## Required execution sequence

1. **Locate and read.** Find `data/<dataset>/DATA_DESCRIPTION.md` and the data file(s) it points to.
2. **Profile.** `python scripts/profiler.py --dataset <dataset> --out outputs/<dataset>/profile.json`. Read the resulting `profile.json` and `profile_summary.md`.
3. **Consult skills.** Read `.claude/skills/data-profiling/SKILL.md` to decide preprocessing, then `.claude/skills/method-selection/SKILL.md` to pick methods and hyperparameters.
4. **Write the plan.** Write `outputs/<dataset>/plan.json` (schema and decision logic live in `method-selection/SKILL.md`, not here). Every preprocessing step and every method needs a `reason` grounded in `profile.json`/`DATA_DESCRIPTION.md`.
5. **Validate.** `python scripts/validate_plan.py outputs/<dataset>/plan.json`. Fix and re-validate before moving on.
6. **Execute the plan (preferred, single command):**
   ```bash
   python scripts/run_plan.py outputs/<dataset>/plan.json
   ```
   This runs `reduce_dim.py` -> `evaluate.py` -> `visualize.py` for every method in the plan, applies the Fallback rules below per method, and writes `outputs/<dataset>/run_log.json` recording what succeeded, what fell back, and what was skipped. **If you need to redo just one method** (e.g. after changing a hyperparameter for it alone), call `reduce_dim.py`/`evaluate.py`/`visualize.py` directly for that method instead of rerunning the whole plan. Never write ad hoc reduction or plotting code inline; every numeric or visual artifact comes from these scripts.
7. **Critique, if built.** If `.claude/agents/dr-critic.md` exists, dispatch it with only `plan.json`, `metrics/*.json`, and figure paths, never your own reasoning transcript. A returned `critique.json` gets addressed with exactly one plan revision, then repeat steps 5 to 7 for whatever changed. If the critic doesn't exist yet or produces nothing, go straight to step 8.
8. **Report.** Read `.claude/skills/reporting/SKILL.md`, then `python scripts/report.py --dataset <dataset> --report-number <N>`.
9. **Final self-check.** Work through the checklist near the end of this file.

---

## Guard rails

- Never treat t-SNE/UMAP cluster sizes, shapes, or inter-cluster distances as meaningful on their own; run at least two seeds or two hyperparameter settings for any manifold method before trusting a structural claim in the report.
- MDS and Isomap are roughly O(N²) or worse. If `n_samples` is large (rough guide: above ~5000), subsample before running them and record that choice as a `plan.json` reason; don't do it silently.
- One shared preprocessing block per plan, applied identically before every method in it; never let one method silently use different preprocessing than another.
- Isomap/LLE/Laplacian Eigenmaps neighbor graphs can be disconnected: retry once with a larger `n_neighbors`, then apply the Fallback rules below if it still fails.

## Fallback rules

- If a method fails after its one retry (see Guard rails), skip it, fall back to reporting PCA alone for that slot, and document the failure and fallback explicitly in `run_log.json`, `plan.json`, and the report. One method failing must never crash the whole run.
- If the critic subagent isn't built yet or is unavailable, proceed without it (step 7). Don't block a run on missing optional infrastructure.
- Never ship a report with a missing figure, a NaN/Inf metric, or a fabricated number. If a metric genuinely can't be computed, say so explicitly rather than omitting it or inventing a value.

---

## Tool contracts

Run any script with `--help` for its current, authoritative flags; this is the target shape:

| Script | Invocation | Produces |
|---|---|---|
| `scripts/profiler.py` | `--dataset <name> --out outputs/<dataset>/profile.json` | `profile.json` + `profile_summary.md` |
| `scripts/reduce_dim.py` | `--dataset <name> --method <name> --params '<json>' --seed <int>` | `outputs/<dataset>/embeddings/<method>.npy` + sidecar params JSON |
| `scripts/evaluate.py` | `--embedding <path> --X <path> [--labels <path>]` | `outputs/<dataset>/metrics/<method>.json` |
| `scripts/visualize.py` | `--embedding <path> [--labels <path>] --title <str>` | `outputs/<dataset>/figures/<method>.png` |
| `scripts/validate_plan.py` | `outputs/<dataset>/plan.json` | pass/fail + JSON+Markdown report |
| `scripts/run_plan.py` | `outputs/<dataset>/plan.json` | runs `reduce_dim`/`evaluate`/`visualize` for every method in the plan; writes `outputs/<dataset>/run_log.json` |
| `scripts/report.py` | `--dataset <name> --report-number {1,2,...}` | `reports/<dataset>/generated_report_<N>.pdf` |
| `scripts/make_synthetic.py` (deferred) | `--n-samples <int> --seed <int>` | a synthetic dataset folder under `data/` |

---

## Output requirements

`reports/<dataset>/generated_report_<N>.pdf` must:

- Exist at exactly that path; no `.md` copy, no duplicate at the repo root.
- Have every figure embedded inline in the PDF, not linked externally.
- Include: the dataset profile and why the chosen preprocessing fits it; every method run, its hyperparameters, and why it was chosen; a quantitative metrics table; all visualizations; the critique-and-revision log if a critique happened; explicitly stated limitations.
- Trace every quoted number back to a real value in `outputs/<dataset>/metrics/*.json`.

---

## Skills and subagents index

- `.claude/skills/data-profiling/SKILL.md`: how to turn a profile into a preprocessing recommendation.
- `.claude/skills/method-selection/SKILL.md`: how to choose methods and hyperparameters, and the full `plan.json` schema.
- `.claude/skills/embedding-evaluation/SKILL.md` (deferred): quantitative and qualitative evaluation criteria.
- `.claude/skills/reporting/SKILL.md`: report structure, tone, and content requirements.
- `.claude/agents/dr-critic.md` (deferred): the adversarial critique subagent's contract and return format.

---

## Final self-check checklist

Before declaring a dataset done, confirm every item below, and fix anything false rather than reporting success anyway:

- [ ] `outputs/<dataset>/plan.json` exists and passes `validate_plan.py`.
- [ ] Every method listed in the plan has a corresponding metric file and figure, or a documented, reasoned skip in `run_log.json`.
- [ ] If a critique was produced, it was addressed with a plan revision, or explicitly overridden with a stated reason.
- [ ] Every number quoted in the report text matches a real value in `outputs/<dataset>/metrics/*.json`.
- [ ] The random seed used is recorded in `plan.json` and referenced in the report.
- [ ] `reports/<dataset>/generated_report_<N>.pdf` exists, opens, and has its figures embedded.
- [ ] Nothing was hardcoded for this specific dataset's name, columns, or classes outside of that run's own `plan.json`.
- [ ] Every `reason` field in `plan.json` says something specific to this dataset; none are a placeholder like "standard choice" with nothing to back it up.
- [ ] You processed exactly the one dataset you were told to, and no other.

---

## Final response style

After completing a dataset, summarize for the user:

1. **Status:** success, or a documented partial/failure and why.
2. **Dataset profile highlights:** n_samples, n_features, and anything structurally notable.
3. **Methods run:** names, key hyperparameters, and why each was chosen.
4. **Key quantitative results:** headline metric(s) per method.
5. **Critique outcome:** what was addressed, or "critic not yet built, skipped" if applicable.
6. **Output file:** confirm the `reports/<dataset>/generated_report_<N>.pdf` path.
7. **Warnings or fallbacks used:** any method skipped, subsampling applied, retries taken.

Keep the summary concise and factual. Do not claim a result you have not verified against `metrics/*.json`.
