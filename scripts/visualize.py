"""Renders a 2D scatter plot of one method's embedding: colored by label if
   the dataset has labels, colored by density (overlap-alpha) otherwise.
   Only the first two dimensions are plotted; embeddings with more (e.g. PCA)
   are noted as such in the title rather than silently implying 2D is the
   whole picture.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, this only ever writes PNG files
import matplotlib.pyplot as plt
import numpy as np
from loaders import load_dataset
from plan_schema import METHOD_ROLES

# Density case (no labels): the dataviz skill's own sequential blue ramp,
# mid-step, used at low alpha so overlapping points read as denser regions.
DENSITY_COLOR = "#256abf"

# Categorical case: the dataviz skill's palette is only validated for 3 categories in a scatter (past that
# it recommends faceting or folding into "Other"). We need a single-embedding overview of up to ~20 classes,
# so we use matplotlib's tab10/tab20 (established categorical colormaps), and fold everything past the 19
# largest classes into one gray "other" (see categorical_colors). A deliberate tradeoff, not an oversight.
OTHER_COLOR = "#b0b0b0"
MAX_LEGEND_CLASSES = 20  # tab20's size; beyond this the smallest classes are folded into one gray "other"


def categorical_colors(y, label_fn=str):
    """Per-point colors (n, 4) and legend entries [(text, color)] for a categorical vector.
    Up to 10 classes use tab10, up to 20 use tab20; beyond that the 19 largest classes keep a color
    and every smaller one is folded into a single gray "other" so no two classes share a color."""
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_classes <= 10:
        cmap, keep = plt.get_cmap("tab10"), list(classes)
    elif n_classes <= MAX_LEGEND_CLASSES:
        cmap, keep = plt.get_cmap("tab20"), list(classes)
    else:
        cmap = plt.get_cmap("tab20")
        biggest = np.argsort(-counts, kind="stable")[: MAX_LEGEND_CLASSES - 1]
        keep = sorted(classes[biggest].tolist())
    color_of = {c: cmap(i) for i, c in enumerate(keep)}
    n_other = n_classes - len(keep)
    colors = np.array([color_of.get(c, matplotlib.colors.to_rgba(OTHER_COLOR)) for c in y])  # (n, 4)
    legend = [(label_fn(c), color_of[c]) for c in keep]
    if n_other:
        legend.append((f"other ({n_other} classes)", matplotlib.colors.to_rgba(OTHER_COLOR)))
    return colors, legend


def make_scatter(embedding, y, dataset, method, title=None, color_note=None, label_fn=str):
    """`y` is any categorical per-point vector (true labels or data-derived cluster ids) or None for
    density coloring; `color_note` is appended to the subtitle so the reader knows what the colors mean."""
    x, y_coord = embedding[:, 0], embedding[:, 1]
    n = embedding.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Marker size shrinks with N so a 90K-point cloud doesn't saturate into a
    # solid blob; a few thousand points keep the original size.
    point_size = float(np.clip(6.0 * (5000.0 / max(n, 1)) ** 0.5, 1.0, 6.0))

    if y is not None:
        colors, legend = categorical_colors(y, label_fn)
        # Random draw order: plotting class-by-class would let whichever
        # class is drawn last systematically cover the others.
        order = np.random.default_rng(0).permutation(n)
        ax.scatter(
            x[order], y_coord[order],
            s=point_size, alpha=0.6 if n <= 5000 else 0.4, linewidths=0,
            color=colors[order],
        )
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=c, label=t) for t, c in legend]
        ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    else:
        # No ground truth to color by: alpha-blended monochrome scatter, so
        # overlapping points visually darken denser regions.
        alpha = float(np.clip(200.0 / max(n, 1), 0.05, 0.6))
        ax.scatter(x, y_coord, s=point_size, alpha=alpha, linewidths=0, color=DENSITY_COLOR)

    if METHOD_ROLES.get(method) == "visualization_only":
        # UMAP/t-SNE axes carry no quantitative meaning (distances aren't
        # preserved), so drop ticks and spines entirely and draw the standard
        # small L-shaped axis indicator in the bottom-left corner instead.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        arrow = dict(arrowstyle="-|>", color="0.25", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=8)
        ax.annotate("", xy=(0.14, -0.02), xytext=(-0.02, -0.02), xycoords="axes fraction", arrowprops=arrow, annotation_clip=False)
        ax.annotate("", xy=(-0.02, 0.14), xytext=(-0.02, -0.02), xycoords="axes fraction", arrowprops=arrow, annotation_clip=False)
        ax.text(0.06, -0.045, f"{method.upper()} 1", transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.25")
        ax.text(-0.045, 0.06, f"{method.upper()} 2", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=8, color="0.25")
    else:
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    subtitle = f"n={n:,}"
    if color_note:
        subtitle += f", {color_note}"
    if embedding.shape[1] > 2:
        subtitle += f", showing dims 1-2 of {embedding.shape[1]}"
    # A cluster-colored subtitle can run long (e.g. "n=2,700, colored by data-derived clusters
    # (k-means on pca, k=3), showing dims 1-2 of 10"); wrap it instead of letting it clip at the
    # figure's right edge. Title left-aligned with the subtitle beneath it either way. Both are
    # placed by a point offset from the axes' top edge (not an axes-fraction one), so the spacing
    # stays correct regardless of the axes' pixel size or how many subtitle lines there are.
    subtitle_lines = textwrap.wrap(subtitle, width=58) or [subtitle]
    ax.set_title(title or f"{dataset}: {method}", fontsize=11, loc="left", pad=18 + 11 * (len(subtitle_lines) - 1))
    for i, line in enumerate(subtitle_lines):
        # Reading order top-to-bottom: line 0 sits highest (closest to the title), the last
        # line sits lowest (right above the axes); each line is one font-size step apart.
        offset_pt = 3 + 11 * (len(subtitle_lines) - 1 - i)
        ax.annotate(
            line, xy=(0, 1), xycoords="axes fraction", xytext=(0, offset_pt), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color="#898781", annotation_clip=False,
        )

    fig.tight_layout()
    return fig


def load_cluster_coloring(dataset, n):
    """Data-derived cluster ids to color an unlabeled dataset's plots, or (None, None) if the plan has no
    clustering block or the saved clusters don't match this embedding. Returns (labels (n,), note)."""
    out = Path("outputs") / dataset
    labels_path, metrics_path, plan_path = out / "clusters.npy", out / "metrics" / "clustering.json", out / "plan.json"
    if not (labels_path.exists() and metrics_path.exists() and plan_path.exists()):
        return None, None
    if "clustering" not in json.loads(plan_path.read_text()):
        return None, None  # a leftover file from an earlier plan revision must not color this plan's figures
    labels = np.load(labels_path)
    if labels.shape != (n,):
        return None, None
    info = json.loads(metrics_path.read_text())
    note = f"colored by data-derived clusters (k-means on {info['params']['source']}, k={info['k']})"
    return labels, note


def make_scree(evr, n_used, dataset, method):
    """Scree plot from the full diagnostic variance spectrum: individual
    variance per component (left) and cumulative variance (right), as two
    panels rather than one dual-axis chart. Marks how many components the
    embedding actually used and where the 90% line sits."""
    evr = np.asarray(evr)
    k = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

    ax1.bar(k, evr, width=0.8, color=DENSITY_COLOR, linewidth=0)
    ax1.set_title("Variance explained by each PC", fontsize=10, loc="left")
    ax1.set_ylabel("Fraction of total variance")

    ax2.plot(k, cum, color=DENSITY_COLOR, lw=2)
    ax2.axhline(0.9, color="0.4", lw=1)
    ax2.text(0.02, 0.9, "90%", transform=ax2.get_yaxis_transform(), va="bottom", ha="left", fontsize=8, color="0.3")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Cumulative variance explained", fontsize=10, loc="left")
    used_var = float(cum[min(n_used, len(cum)) - 1])
    ax2.annotate(
        f"{n_used} PCs used: {used_var * 100:.1f}%",
        xy=(min(n_used, len(cum)), used_var), xytext=(0.35, 0.25), textcoords="axes fraction",
        fontsize=8, color="0.25", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8),
    )
    if cum[-1] < 0.9:
        ax2.text(0.98, 0.04, f"90% not reached within {len(evr)} PCs", transform=ax2.transAxes,
                 ha="right", fontsize=8, style="italic", color="0.3")

    for ax in (ax1, ax2):
        ax.axvline(n_used, color="0.6", lw=0.8)
        ax.set_xlabel("Principal component")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(f"{dataset}: {method} scree plot", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 2D scatter plot for one method's embedding.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--embedding", default=None, help="Default: outputs/<dataset>/embeddings/<method>.npy")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", default=None, help="Default: outputs/<dataset>/figures/<method>.png")
    args = parser.parse_args()

    embedding_path = (
        Path(args.embedding) if args.embedding else Path("outputs") / args.dataset / "embeddings" / f"{args.method}.npy"
    )
    embedding = np.load(embedding_path)

    _, y, metadata = load_dataset(args.dataset)  # labels for coloring only, per CLAUDE.md

    # Labels win when present; otherwise data-derived clusters (if the plan asked for them); otherwise density
    color_note, label_fn = None, str
    if y is not None:
        color_note = "colored by ground-truth labels"
    else:
        y, color_note = load_cluster_coloring(args.dataset, embedding.shape[0])
        if y is not None:
            label_fn = lambda c: f"cluster {int(c) + 1}"  # noqa: E731

    fig = make_scatter(embedding, y, args.dataset, args.method, title=args.title, color_note=color_note, label_fn=label_fn)

    out_path = Path(args.out) if args.out else Path("outputs") / args.dataset / "figures" / f"{args.method}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote {out_path}")

    # PCA-family methods also save their full variance spectrum; if present,
    # draw the scree plot alongside the scatter.
    sidecar_path = embedding_path.with_suffix(".json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        spectrum = sidecar.get("explained_variance_ratio_diagnostic")
        if spectrum:
            scree_fig = make_scree(spectrum, embedding.shape[1], args.dataset, args.method)
            scree_path = out_path.with_name(f"{args.method}_scree.png")
            scree_fig.savefig(scree_path)
            plt.close(scree_fig)
            print(f"Wrote {scree_path}")
