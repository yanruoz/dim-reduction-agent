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

    pages = []

    # --- Title page ---
    pages.append(
        text_page(
            f"Dimension Reduction Analysis: {args.dataset}",
            [(None, [f"Generated report #{args.report_number}", f"Seed: {plan.get('seed')}  |  Plan revision: {plan.get('revision')}"])],
        )
    )

    # --- Profile + preprocessing page ---
    summary_line = (
        f"Samples: {profile['n_samples']:,}   Features: {profile['n_features']:,}   "
        f"Sparsity: {profile['sparsity'] * 100:.1f}%   Missing values: {'Yes' if profile['has_missing'] else 'No'}"
    )
    preprocessing_blocks = [(None, [what_this_is, summary_line])]
    for step in plan.get("preprocessing", []):
        preprocessing_blocks.append((f"Preprocessing: {step['step']}", [step.get("reason", "")]))
    pages.append(text_page("Dataset profile & preprocessing", preprocessing_blocks))

    # --- Methods page ---
    method_blocks = []
    for m in plan["methods"]:
        status = run_log_by_method.get(m["name"], {}).get("status", "unknown (no run_log.json)")
        heading = f"{m['name']}  (role: {m['role']}, status: {status})"
        hp_str = ", ".join(f"{k}={v}" for k, v in m.get("hyperparameters", {}).items())
        method_blocks.append((heading, [f"Hyperparameters: {hp_str}", m.get("reason", "")]))
    pages.append(text_page("Methods and hyperparameters", method_blocks))

    # --- Metrics table page ---
    metric_keys = [
        "trustworthiness",
        "explained_variance_ratio_sum",
        "variance_explained_by_used_components",
        "components_needed_for_90pct_variance",
        "stress",
        "label_silhouette_sanity_check",
    ]
    rows = []
    for m in plan["methods"]:
        name = m["name"]
        metrics_path = out_dir / "metrics" / f"{name}.json"
        if not metrics_path.exists():
            rows.append([name, "(no metrics: method did not succeed, see Limitations)"] + [""] * (len(metric_keys) - 1))
            continue
        metrics = json.loads(metrics_path.read_text())
        row = [name]
        for key in metric_keys:
            if key == "explained_variance_ratio_sum" and "explained_variance_ratio" in metrics:
                row.append(fmt_metric(round(sum(metrics["explained_variance_ratio"]), 4)))
            else:
                row.append(fmt_metric(metrics.get(key)))
        rows.append(row)
    headers = ["method", "trustworthiness", "sum(EVR)", "var. used", "PCs for 90%", "MDS stress", "label silhouette"]
    pages.append(table_page("Quantitative metrics", headers, rows))

    # --- Figure pages ---
    for m in plan["methods"]:
        fig_path = out_dir / "figures" / f"{m['name']}.png"
        if fig_path.exists():
            pages.append(figure_page(fig_path, f"{args.dataset}: {m['name']}"))

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
    if profile.get("has_labels") is False:
        limitation_lines.append(
            "No ground-truth labels available for this dataset; embedding quality assessed only via "
            "label-free trustworthiness, no supervised sanity check possible."
        )
    if not critique:
        limitation_lines.append("No adversarial critique/revision pass was performed for this run (critic subagent not yet built).")
    if not limitation_lines:
        limitation_lines.append("No known limitations beyond what's noted above for each method.")
    pages.append(text_page("Limitations", [(None, limitation_lines)]))

    out_path = Path("reports") / args.dataset / f"generated_report_{args.report_number}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fig in pages:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {out_path} ({len(pages)} pages)")
