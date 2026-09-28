---
name: reporting
description: Write the findings section and produce the final PDF report for one dataset. Used in step 8 of CLAUDE.md's execution sequence, after run_plan.py has finished. Defines what the agent writes (findings.md), what report.py generates on its own, and how to verify the PDF afterward.
---

# Reporting

`scripts/report.py` builds `reports/<dataset>/generated_report_<N>.pdf` from artifacts. Most pages are generated mechanically, so there is nothing for you to write there. **The one part you author is the written interpretation**, saved as `outputs/<dataset>/findings.md`. Without it the report is numbers and figures with no reading of them, and `report.py` says so on the Limitations page.

## What report.py generates by itself

Title, dataset profile and preprocessing (with each step's `reason`), methods and hyperparameters (with each method's `reason`, role, and `run_log.json` status), a metrics table, a "How to read the metrics" page defining every metric shown, one page per figure (PCA-family methods also get a scree plot), your findings, an optional critique page, and a Limitations page. The Limitations page fills itself from `run_log.json` failures, subsampling, missing labels, and the absence of a critique.

Because the `reason` fields are printed verbatim, a vague reason in `plan.json` becomes a vague sentence in the report. If one reads badly, fix it in the plan and re-validate; don't paper over it in the findings.

## Steps

1. **Read before writing.** Open `outputs/<dataset>/run_log.json`, every `outputs/<dataset>/metrics/*.json`, `plan.json`, and **look at each figure** under `outputs/<dataset>/figures/` (view the PNGs; don't infer what they show from the metrics). Metrics are the only source of numbers you may quote.

2. **Write `outputs/<dataset>/findings.md`.** Plain paragraphs separated by blank lines; a line starting with `## ` starts a headed section. Aim for one to two pages. Cover, in roughly this order:
   - **What each method produced**, with its actual numbers from `metrics/*.json`, in the same rounding the table uses.
   - **What the figures show**, hedged appropriately (see rules below).
   - **How the methods relate.** A `general_purpose` and a `visualization_only` method are not competitors; say what each is for rather than declaring a winner.
   - **What can't be concluded**, and why (no labels, a subsampled metric, a method that fell back, etc.).

3. **Run the report:**
   ```bash
   python scripts/doctor.py --check report
   python scripts/report.py --dataset <dataset> --report-number <N>
   ```

4. **Verify the PDF, don't assume it.** Render pages to images (`pdftoppm -png -r 80 <pdf> <prefix>`) and look at them, or at minimum extract text (`pdftotext -layout <pdf> -`). Check: the page count is sensible; the metrics table matches `metrics/*.json`; every figure page shows a figure; no text is clipped at a page edge; your findings page is present. Fix what's wrong and regenerate; a report that ran without error is not the same as a correct one.

## Rules for the findings

- **Every number must appear in `metrics/*.json`** (or be a direct count from `profile.json`). Don't compute new statistics in prose, don't round differently from the table, and don't quote a number you can't point to.
- **Don't compare metrics across different output dimensionalities as if they measured the same thing.** Trustworthiness of a 50-dimensional PCA embedding and a 2-dimensional UMAP embedding are not a fair head-to-head; say so rather than calling the higher one "better".
- **UMAP and t-SNE geometry is a picture, not a measurement.** You may say "the plot shows N visually separated groups"; you may not treat cluster sizes, shapes, or the distances between clusters as findings (CLAUDE.md Guard rails). Never say a `visualization_only` embedding "confirms" a structure.
- **Labels are a sanity check, not a result to optimize.** Report the label silhouette plainly, including when it is negative or near zero, and say what that does and doesn't imply (poor separation *by that measure in that embedding*, not "the classes are indistinguishable").
- **State negative and null results plainly.** If PCA needed more components than were available in the diagnostic fit to reach 90% variance, say ">N", exactly as the table does. If a method fell back or was skipped, say so; don't leave it to the Limitations page alone.
- **Data-derived clusters are a coloring, not a result.** When the plan has a `clustering` block, the report's methods page states k, the scanned range, and either the silhouette values or how the density cross-check picked k (from `metrics/clustering.json`'s `k_selection`; see `method-selection/SKILL.md`). You may refer to "cluster 2" when describing a figure, and say how k was picked; you may not present the clusters as discovered groups, compare them to labels that don't exist, or infer meaning from their sizes.
- **No claims the data can't support.** For an unlabeled dataset, don't name cell types, tissue types, or classes; describe visible structure only, marked as exploratory.
- **Plain, direct language.** No promotional wording, no unexplained jargon, no hedging that hides a real finding.

## Final check

Before declaring the report done (this feeds CLAUDE.md's final self-check): `findings.md` exists and cites only real numbers; the PDF exists at `reports/<dataset>/generated_report_<N>.pdf`, opens, and has every figure embedded; every method in the plan appears in the table or is accounted for on the Limitations page.
