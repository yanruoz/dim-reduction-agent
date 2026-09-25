"""Assembles reports/<dataset>/generated_report_<N>.pdf from everything in
   outputs/<dataset>/: profile, plan (with reasons), metrics, and the
   figures visualize.py already produced. Built on matplotlib's PdfPages,
   the same approach two independent STAI-X repos (Sentinelle,
   staix-submission) both converged on for this exact "autonomously produce
   report.pdf" problem; zero new dependencies since matplotlib is already
   required elsewhere in this pipeline.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from profiler import _missing_text

PAGE_SIZE = (8.5, 11)
WRAP_WIDTH = 85
MIN_Y_BEFORE_TRUNCATE = 0.06


def _wrap(text, width=WRAP_WIDTH):
    return textwrap.wrap(text, width=width) or [""]


def text_page(title, blocks):
    """blocks: list of (heading_or_None, list_of_paragraph_strings). If
    content overflows the page, it's noted explicitly rather than silently
    dropped, since nothing here should disappear without a trace."""
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    y = 0.97
    ax.text(0.05, y, title, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    y -= 0.06

    truncated = False
    for heading, paragraphs in blocks:
        if y < MIN_Y_BEFORE_TRUNCATE:
            truncated = True
            break
        if heading:
            ax.text(0.05, y, heading, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
            y -= 0.035
        for para in paragraphs:
            for line in _wrap(para):
                if y < MIN_Y_BEFORE_TRUNCATE:
                    truncated = True
                    break
                ax.text(0.06, y, line, transform=ax.transAxes, fontsize=9, va="top", family="monospace")
                y -= 0.022
            if truncated:
                break
            y -= 0.008
        y -= 0.015

    if truncated:
        ax.text(
            0.05, MIN_Y_BEFORE_TRUNCATE - 0.02,
            "(content truncated to fit one page; see the JSON artifacts under outputs/ for the full detail)",
            transform=ax.transAxes, fontsize=8, style="italic", color="0.4",
        )
    return fig


def _block_height(heading, paragraphs):
    lines = sum(len(_wrap(p)) for p in paragraphs)
    return (0.035 if heading else 0.0) + lines * 0.022 + 0.008 * len(paragraphs) + 0.015


def text_pages(title, blocks):
    """Like text_page, but splits blocks across as many pages as needed (a
    block is never split mid-way), so a plan with many methods spills onto a
    continuation page instead of being truncated."""
    capacity = 0.97 - 0.06 - MIN_Y_BEFORE_TRUNCATE
    pages_blocks, current, used = [], [], 0.0
    for heading, paragraphs in blocks:
        h = _block_height(heading, paragraphs)
        if current and used + h > capacity:
            pages_blocks.append(current)
            current, used = [], 0.0
        current.append((heading, paragraphs))
        used += h
    if current or not pages_blocks:
        pages_blocks.append(current)
    return [
        text_page(title if i == 0 else f"{title} (cont.)", b) for i, b in enumerate(pages_blocks)
    ]


def table_page(title, headers, rows):
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    ax.text(0.05, 0.97, title, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")
    table = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="left", bbox=[0.02, 0.4, 0.96, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(len(headers))))
    return fig


def figure_page(png_path, title):
    # visualize.py's PNGs already carry their own title/subtitle; no outer
    # title here to avoid showing it twice. Sized close to the source image's
    # own aspect ratio rather than forcing it into a full letter-page axes,
    # so it doesn't sit in a mostly-blank page.
    img = plt.imread(png_path)
    h, w = img.shape[0], img.shape[1]
    fig_w = 7.5
    fig_h = fig_w * (h / w)
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_h, 10)))
    ax.axis("off")
    ax.imshow(img)
    fig.tight_layout()
    return fig


def parse_what_this_is(desc_path):
    text = Path(desc_path).read_text()
    if "## What this is" not in text:
        return "(no 'What this is' section found in DATA_DESCRIPTION.md)"
    section = text.split("## What this is", 1)[1]
    section = section.split("\n## ", 1)[0]
    return " ".join(section.strip().split())


def parse_findings(text):
    """findings.md -> [(heading_or_None, [paragraphs])]. Blank lines separate
    paragraphs; a line starting with '## ' starts a new headed block."""
    blocks, heading, paragraphs, buf = [], None, [], []

    def flush_paragraph():
        if buf:
            paragraphs.append(" ".join(buf))
            buf.clear()

    for line in text.splitlines():
        if line.startswith("## "):
            flush_paragraph()
            if heading is not None or paragraphs:
                blocks.append((heading, list(paragraphs)))
            heading, paragraphs = line[3:].strip(), []
        elif not line.strip():
            flush_paragraph()
        else:
            buf.append(line.strip())
    flush_paragraph()
    if heading is not None or paragraphs:
        blocks.append((heading, list(paragraphs)))
    return blocks


def fmt_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return f"[{len(value)} values]"
    return str(value) if value not in (None, "") else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the final PDF report from a dataset's outputs/.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report-number", required=True, type=int)
    args = parser.parse_args()

    out_dir = Path("outputs") / args.dataset
    plan = json.loads((out_dir / "plan.json").read_text())
    profile = json.loads((out_dir / "profile.json").read_text())
    desc_path = Path("data") / args.dataset / "DATA_DESCRIPTION.md"
    what_this_is = parse_what_this_is(desc_path) if desc_path.exists() else "(DATA_DESCRIPTION.md not found)"

    run_log_path = out_dir / "run_log.json"
    run_log = json.loads(run_log_path.read_text()) if run_log_path.exists() else None
    run_log_by_method = {m["name"]: m for m in run_log["methods"]} if run_log else {}

    critique_path = out_dir / "critique.json"
    critique = json.loads(critique_path.read_text()) if critique_path.exists() else None

    # Written interpretation is the one part of the report an agent authors
    # (see .claude/skills/reporting/SKILL.md); everything else is generated
    # mechanically from artifacts. Optional so report.py still runs without it.
    findings_path = out_dir / "findings.md"
    findings = findings_path.read_text() if findings_path.exists() else None

    pages = []

    # --- Title page ---
    pages.append(
        text_page(
            f"Dimension Reduction Analysis: {args.dataset}",
            [(None, [f"Generated report #{args.report_number}", f"Seed: {plan.get('seed')}  |  Plan revision: {plan.get('revision')}"])],
        )
    )

    # --- Profile + preprocessing page ---
    sparsity_text = "n/a" if profile["sparsity"] is None else f"{profile['sparsity'] * 100:.1f}%"
    summary_line = (
        f"Samples: {profile['n_samples']:,}   Features: {profile['n_features']:,}   Sparsity: {sparsity_text}"
    )
    # Missing-value detail (counts, worst feature) comes from the profile so the report never restates it by hand
    profile_lines = [what_this_is, summary_line, f"Missing values: {_missing_text(profile)}"]
    preprocessing_blocks = [(None, profile_lines)]
    for step in plan.get("preprocessing", []):
        preprocessing_blocks.append((f"Preprocessing: {step['step']}", [step.get("reason", "")]))
    pages.extend(text_pages("Dataset profile & preprocessing", preprocessing_blocks))

    # --- Methods page ---
    method_blocks = []
    for m in plan["methods"]:
        status = run_log_by_method.get(m["name"], {}).get("status", "unknown (no run_log.json)")
        heading = f"{m['name']}  (role: {m['role']}, status: {status})"
        hp_str = ", ".join(f"{k}={v}" for k, v in m.get("hyperparameters", {}).items())
        method_blocks.append((heading, [f"Hyperparameters: {hp_str}", m.get("reason", "")]))
    # Data-derived clusters (only colors unlabeled plots); every number comes from metrics/clustering.json
    clustering = None
    clustering_path = out_dir / "metrics" / "clustering.json"
    if plan.get("clustering") and clustering_path.exists():
        clustering = json.loads(clustering_path.read_text())
        block = plan["clustering"]
        scan = ", ".join(f"k={k}: {v:.3f}" for k, v in clustering["silhouette_by_k"].items() if v is not None)
        how_k = (
            f"k={clustering['k']} was fixed in the plan."
            if clustering["k_was_fixed"]
            else (
                f"k={clustering['k']} was chosen as the value with the highest silhouette among "
                f"k={clustering['params']['k_range'][0]} to {clustering['params']['k_range'][1]} ({scan})."
            )
        )
        scored = (
            f" Silhouette was scored on a fixed-seed subsample of {clustering['n_scored']:,} of {clustering['n_samples']:,} samples."
            if clustering["silhouette_subsampled"] else ""
        )
        method_blocks.append((
            f"Data-derived clusters (k-means on {block['source']}; colors only)",
            [
                f"{how_k}{scored} Cluster sizes: {', '.join(f'{c:,}' for c in clustering['cluster_sizes'])}. "
                "These clusters only color the plots of an unlabeled dataset; they are a description of the "
                "embedding, not a finding, and no biological or domain meaning is attached to them.",
                block.get("reason", ""),
            ],
        ))
    pages.extend(text_pages("Methods and hyperparameters", method_blocks))

    # --- Metrics table page ---
    metric_keys = [
        "trustworthiness",
        "variance_explained_by_used_components",
        "components_needed_for_90pct_variance",
        "stress",
        "label_silhouette_sanity_check",
    ]
    header_for = {
        "trustworthiness": "trustworthiness",
        "variance_explained_by_used_components": "variance explained\n(PCs used)",
        "components_needed_for_90pct_variance": "PCs for 90%\nvariance",
        "stress": "MDS stress",
        "label_silhouette_sanity_check": "label silhouette",
    }
    all_metrics = {}
    for m in plan["methods"]:
        metrics_path = out_dir / "metrics" / f"{m['name']}.json"
        if metrics_path.exists():
            all_metrics[m["name"]] = json.loads(metrics_path.read_text())
    # Only show a column if at least one method in this report has a value for it,
    # so every column shown is also one the definitions page explains.
    metric_keys = [k for k in metric_keys if any(mt.get(k) is not None for mt in all_metrics.values())]
    headers = ["method"] + [header_for[k] for k in metric_keys]
    rows = []
    for m in plan["methods"]:
        name = m["name"]
        if name not in all_metrics:
            rows.append([name, "(no metrics: method did not succeed, see Limitations)"] + [""] * (len(metric_keys) - 1))
            continue
        metrics = all_metrics[name]
        row = [name]
        for key in metric_keys:
            value = metrics.get(key)
            if key == "components_needed_for_90pct_variance" and value is not None:
                # A lower bound must be shown as one; a bare number would imply 90% was actually reached there.
                is_lb = metrics.get("components_needed_for_90pct_variance_is_lower_bound")
                row.append(f">{value}" if is_lb else str(value))
            else:
                row.append(fmt_metric(value))
        rows.append(row)
    pages.append(table_page("Quantitative metrics", headers, rows))

    # --- Metric definitions: only for the metrics that actually appear above ---
    definitions = {
        "trustworthiness": (
            "Trustworthiness (sklearn.manifold.trustworthiness, 10 neighbours)",
            "For each sample, takes its 10 nearest neighbours in the embedding and penalizes any that were not "
            "close neighbours in the original preprocessed data, weighting each penalty by how far down the "
            "original neighbour ranking that point sat. Ranges 0 to 1; 1 means the embedding introduces no false "
            "neighbours. It measures local neighbourhood fidelity only, not preservation of global distances. "
            "Computed against the data after the plan's preprocessing steps; on a random 5,000-sample subset "
            "when the dataset is larger than that (see Limitations).",
        ),
        "variance_explained_by_used_components": (
            "Variance explained (PCs used)",
            "The fraction of the total variance in the preprocessed data captured by the principal components "
            "actually kept in the embedding: the sum of those components' explained-variance ratios (each "
            "component's variance divided by the total). Ranges 0 to 1. PCA only; blank for methods it does "
            "not apply to.",
        ),
        "components_needed_for_90pct_variance": (
            "PCs for 90% variance",
            "The smallest number of principal components whose cumulative explained variance reaches 90%, read "
            "off a diagnostic PCA fit of up to min(4 x n_components, 200) components. A leading '>' means 90% "
            "was not reached within that fit, so the true number is larger than shown. The scree plots show "
            "the full curve.",
        ),
        "stress": (
            "MDS stress (Kruskal's Stress-1)",
            "The square root of the summed squared differences between the embedding's pairwise distances "
            "and the original pairwise distances, divided by the summed squared original distances "
            "(sklearn MDS with normalized_stress=True). It is scale-free, so it can be compared across "
            "datasets. Lower is better; 0 would be a perfect distance-preserving embedding. MDS only.",
        ),
        "label_silhouette_sanity_check": (
            "Label silhouette",
            "Mean silhouette coefficient (sklearn.metrics.silhouette_score) of the dataset's ground-truth class "
            "labels in the embedding space, using all embedding dimensions. Ranges -1 to 1: near 1 means points "
            "sit close to their own class and far from others, near 0 means classes overlap, negative means "
            "points are on average nearer another class than their own. A supervised sanity check only; "
            "labels were never used to build an embedding or to tune any hyperparameter.",
        ),
    }
    definition_blocks = [
        (heading, [text])
        for key, (heading, text) in definitions.items()
        if any(metrics.get(key) is not None for metrics in all_metrics.values())
    ]
    if definition_blocks:
        pages.extend(text_pages("How to read the metrics", definition_blocks))

    # --- Figure pages (scatter, then the scree plot right after a PCA-family scatter) ---
    for m in plan["methods"]:
        fig_path = out_dir / "figures" / f"{m['name']}.png"
        if fig_path.exists():
            pages.append(figure_page(fig_path, f"{args.dataset}: {m['name']}"))
        scree_path = out_dir / "figures" / f"{m['name']}_scree.png"
        if scree_path.exists():
            pages.append(figure_page(scree_path, f"{args.dataset}: {m['name']} scree"))

    # --- Findings and interpretation, if the agent wrote one ---
    if findings:
        pages.extend(text_pages("Findings and interpretation", parse_findings(findings)))

    # --- Critique page, if present ---
    if critique:
        pages.append(text_page("Critique and revision", [(None, [json.dumps(critique, indent=2)])]))

    # --- Limitations page ---
    limitation_lines = []
    for m in plan["methods"]:
        status = run_log_by_method.get(m["name"], {}).get("status")
        if status not in (None, "ok", "reused"):
            error = run_log_by_method.get(m["name"], {}).get("error", "")
            limitation_lines.append(f"{m['name']}: {status} ({error})")
    subsampled_methods = []
    for m in plan["methods"]:
        mp = out_dir / "metrics" / f"{m['name']}.json"
        if mp.exists():
            mj = json.loads(mp.read_text())
            if mj.get("trustworthiness_subsampled"):
                subsampled_methods.append((m["name"], mj.get("trustworthiness_n_samples_used")))
    if subsampled_methods:
        n_used = subsampled_methods[0][1]
        limitation_lines.append(
            f"Trustworthiness was computed on a random subset of {n_used:,} of {profile['n_samples']:,} samples "
            f"(it is O(N^2) and infeasible at full size) for: {', '.join(n for n, _ in subsampled_methods)}. "
            "Treat those scores as estimates, not exact full-dataset values."
        )
    if profile.get("has_labels") is True:
        limitation_lines.append(
            "The label silhouette column is a supervised sanity check only, computed on the embedding's own "
            "dimensions; it was never used to tune any hyperparameter."
        )
    if profile.get("has_labels") is False:
        limitation_lines.append(
            "No ground-truth labels available for this dataset; embedding quality assessed only via "
            "label-free trustworthiness, no supervised sanity check possible."
        )
    if clustering is not None:
        limitation_lines.append(
            f"The cluster colors on unlabeled plots come from k-means (k={clustering['k']}) on the "
            f"{plan['clustering']['source']} embedding, with k picked by silhouette rather than validated against any "
            "ground truth. Different k, a different algorithm, or a different source embedding would color the "
            "same points differently; treat the colors as a visual aid, not as groups that are known to exist."
        )
    if profile.get("has_missing"):
        steps = [p_["step"] for p_ in plan.get("preprocessing", [])]
        limitation_lines.append(
            f"{profile['missing_fraction'] * 100:.2f}% of the input cells were missing"
            + (" and were filled by the 'impute' step" if "impute" in steps else "")
            + (" after dropping mostly-missing features" if "drop_missing_features" in steps else "")
            + ". Single-value imputation understates uncertainty and pulls filled samples toward the feature centre, "
            "which can shrink apparent structure; if values are missing not at random, every method sees a biased fill."
        )
    if not findings:
        limitation_lines.append("No written interpretation (findings.md) was supplied for this run; the pages above are numbers and figures only.")
    if not critique:
        limitation_lines.append("No adversarial critique/revision pass was performed for this run (critic subagent not yet built).")
    if not limitation_lines:
        limitation_lines.append("No known limitations beyond what's noted above for each method.")
    pages.extend(text_pages("Limitations", [(None, limitation_lines)]))

    out_path = Path("reports") / args.dataset / f"generated_report_{args.report_number}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out_path} ({len(pages)} pages)")
