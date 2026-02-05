"""
Experiment C: Top comment per method visualization.

Shows the #1 ranked comment from each method (PD, Pol.is, p-mean) as separate rows.
"""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, rankdata

# Add parent to path for style import
sys.path.insert(0, str(Path(__file__).parent.parent))

from style import (
    setup_style, despine,
    METRIC_COLORS, METRIC_LABELS, COLORS,
    RIDGELINE_CMAP, RIDGELINE_ALPHA,
    RIDGELINE_OUTLINE_COLOR, RIDGELINE_OUTLINE_WIDTH,
    RIDGELINE_BASELINE_COLOR, RIDGELINE_BASELINE_WIDTH,
)

setup_style()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def escape_latex(text):
    """Escape LaTeX special characters and remove non-ASCII characters."""
    text = text.encode('ascii', 'ignore').decode('ascii')
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('$', r'\$'),
        ('%', r'\%'),
        ('&', r'\&'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('^', r'\^{}'),
        ('~', r'\~{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def compute_unnormalized_density(x_values, x_grid, bandwidth=0.3):
    """Compute KDE density scaled by count (unnormalized)."""
    if len(x_values) < 2:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(x_values, bw_method=bandwidth)
        return kde(x_grid) * len(x_values)
    except Exception:
        return np.zeros_like(x_grid)


def plot_top_per_method(data: dict, output_path: Path):
    """Create ridgeline plot showing top comment from each method."""
    file_id = data["file_id"]
    positions = np.array(data["voter_positions"])

    pd_scores = np.array(data["pd_scores"])
    polis_scores = np.array(data["polis_scores"])
    pmean_scores = np.array(data["pmean_scores"])
    approval_fracs = np.array(data["approval_fracs"])
    n_voted = np.array(data["n_voted"])
    comment_texts = data.get("comment_texts", {})

    # Ranks
    pd_ranks = rankdata(-np.nan_to_num(pd_scores, nan=-999), method="min")
    polis_ranks = rankdata(-np.nan_to_num(polis_scores, nan=-999), method="min")
    pmean_ranks = rankdata(-np.nan_to_num(pmean_scores, nan=-999), method="min")

    # Find top comment for each method
    valid_pd = ~np.isnan(pd_scores)
    valid_polis = ~np.isnan(polis_scores)
    valid_pmean = ~np.isnan(pmean_scores)

    top_pd = np.where(valid_pd)[0][np.argmax(pd_scores[valid_pd])] if valid_pd.any() else None
    top_polis = np.where(valid_polis)[0][np.argmax(polis_scores[valid_polis])] if valid_polis.any() else None
    top_pmean = np.where(valid_pmean)[0][np.argmax(pmean_scores[valid_pmean])] if valid_pmean.any() else None

    # Build list of (method_name, item_idx, method_color) - order: PD, Polis, p-mean
    rows = []
    if top_pd is not None:
        rows.append(("PD", top_pd, METRIC_COLORS["pd"]))
    if top_polis is not None:
        rows.append(("Pol.is", top_polis, METRIC_COLORS["polis"]))
    if top_pmean is not None:
        rows.append((r"$p$-mean", top_pmean, METRIC_COLORS["pmean"]))

    if not rows:
        print(f"  Skipping {file_id}: no valid scores")
        return

    # Load matrix for per-item approver positions
    base_dir = Path(__file__).parent.parent.parent
    npz_path = base_dir / "data" / "processed" / "preflib" / f"{file_id}.npz"
    matrix = np.load(npz_path)["matrix"]
    observed = ~np.isnan(matrix)

    # Grid for KDE
    pos_min, pos_max = positions.min(), positions.max()
    x_margin = (pos_max - pos_min) * 0.1
    x_grid = np.linspace(pos_min - x_margin, pos_max + x_margin, 200)

    # Compute densities
    densities = []
    for method_name, idx, _ in rows:
        approvers = observed[idx, :] & (matrix[idx, :] == 1)
        x_approvers = positions[approvers]
        densities.append(compute_unnormalized_density(x_approvers, x_grid))

    max_density = max(d.max() for d in densities) if densities else 1.0

    # Pre-compute wrapped text and row heights
    wrap_width = 45
    min_row_height = 0.7
    line_height = 0.12
    wrapped_texts = []
    row_heights = []
    for method_name, idx, _ in rows:
        text = comment_texts.get(str(idx), f"Comment {idx + 1}")
        wrapped = textwrap.fill(text, width=wrap_width)
        wrapped = escape_latex(wrapped)
        wrapped_texts.append(wrapped)
        n_lines = wrapped.count('\n') + 1
        row_heights.append(max(min_row_height, 0.35 + n_lines * line_height))

    # Calculate cumulative y positions (bottom-up)
    y_bases = []
    y_cumulative = 0
    for rh in reversed(row_heights):
        y_bases.insert(0, y_cumulative)
        y_cumulative += rh

    total_height = y_cumulative
    scale_factor = (min_row_height * 0.5) / max_density if max_density > 0 else 1.0

    # Figure setup
    fig_width = 7.5
    fig_height = 0.6 + total_height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Colormap
    cmap = get_cmap(RIDGELINE_CMAP)
    norm = Normalize(vmin=pos_min, vmax=pos_max)

    for row_idx, (method_name, item_idx, method_color) in enumerate(rows):
        y_base = y_bases[row_idx]
        row_height = row_heights[row_idx]
        density = densities[row_idx] * scale_factor

        # Gradient fill
        for i in range(len(x_grid) - 1):
            x_left, x_right = x_grid[i], x_grid[i + 1]
            y_left, y_right = density[i], density[i + 1]
            if y_left < 0.001 and y_right < 0.001:
                continue
            color = cmap(norm((x_left + x_right) / 2))
            verts = [
                (x_left, y_base), (x_left, y_base + y_left),
                (x_right, y_base + y_right), (x_right, y_base),
            ]
            ax.add_patch(plt.Polygon(
                verts,
                facecolor=color,
                edgecolor="none",
                alpha=RIDGELINE_ALPHA,
            ))

        # Outline
        ax.plot(
            x_grid, y_base + density,
            color=RIDGELINE_OUTLINE_COLOR,
            linewidth=RIDGELINE_OUTLINE_WIDTH,
            alpha=0.6,
        )

        # Baseline
        ax.axhline(
            y=y_base,
            color=RIDGELINE_BASELINE_COLOR,
            linewidth=RIDGELINE_BASELINE_WIDTH,
        )

        # Comment text (left side, left-aligned)
        wrapped = wrapped_texts[row_idx]
        ax.text(
            pos_min - x_margin - 0.85, y_base + row_height * 0.4,
            wrapped,
            fontsize=6,
            ha="left",
            va="center",
            linespacing=1.1,
        )

        # Method label (far left, colored)
        ax.text(
            pos_min - x_margin - 1.55, y_base + row_height * 0.4,
            method_name,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            color=method_color,
        )

        # Right-side columns
        x_col1 = pos_max + x_margin + 0.03   # Approval % (n)
        x_col2 = x_col1 + 0.38               # PD
        x_col3 = x_col2 + 0.30               # Polis
        x_col4 = x_col3 + 0.30               # p-mean
        y_text = y_base + row_height * 0.35

        # Approval % (n)
        ax.text(x_col1, y_text, f"{approval_fracs[item_idx]*100:.0f}\\%",
                fontsize=7, ha="center", va="center", family="monospace")
        ax.text(x_col1 + 0.16, y_text, f"({n_voted[item_idx]})",
                fontsize=6, ha="left", va="center", family="monospace", color="#777777")

        # PD rank
        ax.text(x_col2, y_text, f"{int(pd_ranks[item_idx])}",
                fontsize=7, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pd"])

        # Polis rank
        ax.text(x_col3, y_text, f"{int(polis_ranks[item_idx])}",
                fontsize=7, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["polis"])

        # p-mean rank
        ax.text(x_col4, y_text, f"{int(pmean_ranks[item_idx])}",
                fontsize=7, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pmean"])

    # Axes limits
    ax.set_xlim(pos_min - x_margin - 1.6, pos_max + x_margin + 1.3)
    ax.set_ylim(-0.15, total_height + 0.25)
    ax.set_yticks([])
    ax.set_xlabel("Voter Position (MDS)", fontsize=9)

    despine(ax, left=True)

    # Column headers
    y_header = total_height + 0.08
    ax.text(pos_min - x_margin - 1.55, y_header, "Method",
            fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax.text(pos_min - x_margin - 0.85, y_header, "Comment",
            fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax.text((pos_min + pos_max) / 2, y_header, "Approver Distribution",
            fontsize=8, fontweight="bold", ha="center", va="bottom")

    x_col1 = pos_max + x_margin + 0.03
    x_col2 = x_col1 + 0.38
    x_col3 = x_col2 + 0.30
    x_col4 = x_col3 + 0.30

    ax.text(x_col1 + 0.1, y_header, "Appr. \\% ($n$)",
            fontsize=7, fontweight="bold", ha="center", va="bottom")
    ax.text(x_col2, y_header, "PD",
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["pd"])
    ax.text(x_col3, y_header, "Pol.is",
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["polis"])
    ax.text(x_col4, y_header, r"$p$-mean",
            fontsize=7, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["pmean"])

    # Save to png, svg, and pdf subfolders
    png_path = output_path.parent / "png" / output_path.name
    svg_path = output_path.parent / "svg" / output_path.with_suffix(".svg").name
    pdf_path = output_path.parent / "pdf" / output_path.with_suffix(".pdf").name
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png_path.name}")


def main():
    results_path = Path(__file__).parent.parent / "results" / "experiment_c.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    for file_id, data in all_results.items():
        print(f"Plotting {file_id}...")
        output_path = plots_dir / f"experiment_c_top_{file_id}.png"
        plot_top_per_method(data, output_path)


if __name__ == "__main__":
    main()
