"""
Experiment A: Ridgeline plots for French election qualitative rankings.

Loads JSON from run.py, produces ridgeline plots with 3 metric columns.
Top 5 candidates by PD score, rows show approval distribution along PC1.
"""

import json
import sys
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


def compute_unnormalized_density(x_values, x_grid, bandwidth=0.3):
    """Compute KDE density scaled by count (unnormalized)."""
    if len(x_values) < 2:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(x_values, bw_method=bandwidth)
    return kde(x_grid) * len(x_values)


def plot_ridgeline(data: dict, output_path: Path, top_n: int = 5):
    """Create ridgeline plot for one dataset with 3 metric columns."""
    candidate_names = data["candidate_names"]
    n_items = data["n_items"]
    pc1_scores = np.array(data["pc1_scores"])

    pd_scores = np.array(data["pd_scores"])
    polis_scores = np.array(data["polis_scores"])
    pmean_scores = np.array(data["pmean_scores"])
    approval_fracs = np.array(data["approval_fracs"])

    # Ranks (1 = highest score)
    pd_ranks = rankdata(-pd_scores, method="min")
    polis_ranks = rankdata(-polis_scores, method="min")
    pmean_ranks = rankdata(-pmean_scores, method="min")

    # Top N by PD
    top_indices = np.argsort(pd_scores)[::-1][:top_n]

    # Load matrix for per-candidate approval vectors
    base_dir = Path(__file__).parent.parent.parent
    dataset_id = data["dataset_id"]
    if "00026" in dataset_id:
        npz_path = base_dir / "data/processed/preflib/00026-combined.npz"
    else:
        npz_path = base_dir / "data/processed/preflib/00071-combined.npz"
    matrix = np.load(npz_path)["matrix"]

    # Setup grid for KDE
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    x_margin = (pc1_max - pc1_min) * 0.1
    x_grid = np.linspace(pc1_min - x_margin, pc1_max + x_margin, 200)

    # Compute densities
    densities = []
    for idx in top_indices:
        approvers = matrix[idx] == 1.0
        x_approvers = pc1_scores[approvers]
        densities.append(compute_unnormalized_density(x_approvers, x_grid))

    max_density = max(d.max() for d in densities) if densities else 1.0
    row_height = 1.0
    scale_factor = (row_height * 0.75) / max_density if max_density > 0 else 1.0

    # Figure setup
    n_display = len(top_indices)
    fig_width = 7.0  # Double column width
    fig_height = 0.8 + n_display * 0.9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Colormap (blue = left, red = right)
    cmap = get_cmap(RIDGELINE_CMAP)
    norm = Normalize(vmin=pc1_min, vmax=pc1_max)

    for row_idx, item_idx in enumerate(top_indices):
        y_base = n_display - 1 - row_idx
        density = densities[row_idx] * scale_factor

        # Gradient fill using trapezoids
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

        # Candidate name (left side)
        name = candidate_names[item_idx]
        ax.text(
            pc1_min - x_margin - 0.15, y_base + row_height * 0.35,
            name,
            fontsize=9,
            fontweight="medium",
            ha="right",
            va="center",
        )

        # Right-side columns (evenly spaced)
        x_col1 = pc1_max + x_margin + 0.15   # Approval %
        x_col2 = x_col1 + 0.45               # PD
        x_col3 = x_col2 + 0.40               # Polis
        x_col4 = x_col3 + 0.40               # p-mean
        y_text = y_base + row_height * 0.35

        # Approval %
        ax.text(x_col1, y_text, f"{approval_fracs[item_idx]*100:.0f}\\%",
                fontsize=8, ha="center", va="center", family="monospace")

        # PD rank
        ax.text(x_col2, y_text, f"{int(pd_ranks[item_idx])}",
                fontsize=8, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pd"])

        # Polis rank
        ax.text(x_col3, y_text, f"{int(polis_ranks[item_idx])}",
                fontsize=8, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["polis"])

        # p-mean rank
        ax.text(x_col4, y_text, f"{int(pmean_ranks[item_idx])}",
                fontsize=8, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pmean"])

    # Axes limits
    ax.set_xlim(pc1_min - x_margin - 1.2, pc1_max + x_margin + 1.8)
    ax.set_ylim(-0.3, n_display + 0.4)
    ax.set_yticks([])
    ax.set_xlabel("PC1 Score", fontsize=9)

    despine(ax, left=True)

    # Column headers
    y_header = n_display + 0.15
    ax.text(pc1_min - x_margin - 0.6, y_header, "Candidate",
            fontsize=8, fontweight="bold", ha="center", va="bottom")
    ax.text((pc1_min + pc1_max) / 2, y_header, "Approval Distribution",
            fontsize=8, fontweight="bold", ha="center", va="bottom")

    x_col1 = pc1_max + x_margin + 0.15
    x_col2 = x_col1 + 0.45
    x_col3 = x_col2 + 0.40
    x_col4 = x_col3 + 0.40

    ax.text(x_col1, y_header, "Appr. \\%",
            fontsize=8, fontweight="bold", ha="center", va="bottom")
    ax.text(x_col2, y_header, "PD",
            fontsize=8, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["pd"])
    ax.text(x_col3, y_header, "Pol.is",
            fontsize=8, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["polis"])
    ax.text(x_col4, y_header, r"$p$-mean",
            fontsize=8, fontweight="bold", ha="center", va="bottom",
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


def plot_ridgeline_on_ax(ax, data: dict, top_n: int = 5, fontscale: float = 1.0):
    """Draw ridgeline plot on a given axes (for combined figures)."""
    candidate_names = data["candidate_names"]
    pc1_scores = np.array(data["pc1_scores"])

    pd_scores = np.array(data["pd_scores"])
    polis_scores = np.array(data["polis_scores"])
    pmean_scores = np.array(data["pmean_scores"])
    approval_fracs = np.array(data["approval_fracs"])

    # Ranks (1 = highest score)
    pd_ranks = rankdata(-pd_scores, method="min")
    polis_ranks = rankdata(-polis_scores, method="min")
    pmean_ranks = rankdata(-pmean_scores, method="min")

    # Top N by PD
    top_indices = np.argsort(pd_scores)[::-1][:top_n]

    # Load matrix for per-candidate approval vectors
    base_dir = Path(__file__).parent.parent.parent
    dataset_id = data["dataset_id"]
    if "00026" in dataset_id:
        npz_path = base_dir / "data/processed/preflib/00026-combined.npz"
    else:
        npz_path = base_dir / "data/processed/preflib/00071-combined.npz"
    matrix = np.load(npz_path)["matrix"]

    # Setup grid for KDE
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    x_margin = (pc1_max - pc1_min) * 0.1
    x_grid = np.linspace(pc1_min - x_margin, pc1_max + x_margin, 200)

    # Compute densities
    densities = []
    for idx in top_indices:
        approvers = matrix[idx] == 1.0
        x_approvers = pc1_scores[approvers]
        densities.append(compute_unnormalized_density(x_approvers, x_grid))

    max_density = max(d.max() for d in densities) if densities else 1.0
    row_height = 1.0
    scale_factor = (row_height * 0.75) / max_density if max_density > 0 else 1.0
    n_display = len(top_indices)

    # Colormap (blue = left, red = right)
    cmap = get_cmap(RIDGELINE_CMAP)
    norm = Normalize(vmin=pc1_min, vmax=pc1_max)

    for row_idx, item_idx in enumerate(top_indices):
        y_base = n_display - 1 - row_idx
        density = densities[row_idx] * scale_factor

        # Gradient fill using trapezoids
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

        # Candidate name (left side)
        name = candidate_names[item_idx]
        ax.text(
            pc1_min - x_margin - 0.12, y_base + row_height * 0.35,
            name,
            fontsize=7 * fontscale,
            fontweight="medium",
            ha="right",
            va="center",
        )

        # Right-side columns (wider spacing for combined plot)
        x_col1 = pc1_max + x_margin + 0.15   # Approval %
        x_col2 = x_col1 + 0.50               # PD
        x_col3 = x_col2 + 0.38               # Polis
        x_col4 = x_col3 + 0.62               # p-mean
        y_text = y_base + row_height * 0.35

        # Approval %
        ax.text(x_col1, y_text, f"{approval_fracs[item_idx]*100:.0f}\\%",
                fontsize=6.5 * fontscale, ha="center", va="center", family="monospace")

        # PD rank
        ax.text(x_col2, y_text, f"{int(pd_ranks[item_idx])}",
                fontsize=6.5 * fontscale, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pd"])

        # Polis rank
        ax.text(x_col3, y_text, f"{int(polis_ranks[item_idx])}",
                fontsize=6.5 * fontscale, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["polis"])

        # p-mean rank
        ax.text(x_col4, y_text, f"{int(pmean_ranks[item_idx])}",
                fontsize=6.5 * fontscale, ha="center", va="center", family="monospace",
                color=METRIC_COLORS["pmean"])

    # Axes limits
    ax.set_xlim(pc1_min - x_margin - 0.95, pc1_max + x_margin + 2.1)
    ax.set_ylim(-0.3, n_display + 0.4)
    ax.set_yticks([])
    ax.set_xlabel("PC1 Score", fontsize=7 * fontscale)

    despine(ax, left=True)

    # Column headers
    y_header = n_display + 0.15
    ax.text(pc1_min - x_margin - 0.45, y_header, "Candidate",
            fontsize=6.5 * fontscale, fontweight="bold", ha="center", va="bottom")
    ax.text((pc1_min + pc1_max) / 2, y_header, "Approver Distribution",
            fontsize=6.5 * fontscale, fontweight="bold", ha="center", va="bottom")

    x_col1 = pc1_max + x_margin + 0.15
    x_col2 = x_col1 + 0.50
    x_col3 = x_col2 + 0.38
    x_col4 = x_col3 + 0.62

    ax.text(x_col1, y_header, "Appr. \\%",
            fontsize=6 * fontscale, fontweight="bold", ha="center", va="bottom")
    ax.text(x_col2, y_header, "PD",
            fontsize=6 * fontscale, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["pd"])
    ax.text(x_col3, y_header, "Pol.is",
            fontsize=6 * fontscale, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["polis"])
    ax.text(x_col4, y_header, r"$p$-mean",
            fontsize=6 * fontscale, fontweight="bold", ha="center", va="bottom",
            color=METRIC_COLORS["pmean"])


def plot_combined(all_results: dict, output_path: Path, top_n: int = 5):
    """Create combined side-by-side ridgeline plot for both elections."""
    # Figure setup - full page width, top half of page height
    fig_width = 7.0   # Full page width
    fig_height = 3.8  # Fits in top half of page
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Plot each dataset
    datasets = list(all_results.items())
    panel_labels = ["(a)", "(b)"]
    titles = ["French Election 2002", "French Election 2007"]

    for idx, (dataset_id, data) in enumerate(datasets):
        ax = axes[idx]
        plot_ridgeline_on_ax(ax, data, top_n=top_n, fontscale=1.0)

    plt.tight_layout()

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
    results_path = Path(__file__).parent.parent / "results" / "experiment_a.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    # Generate individual plots
    for dataset_id, data in all_results.items():
        print(f"Plotting {data['label']}...")
        output_path = plots_dir / f"experiment_a_{dataset_id}.png"
        plot_ridgeline(data, output_path, top_n=5)

    # Generate combined plot
    print("Plotting combined figure...")
    output_path = plots_dir / "experiment_a_combined.png"
    plot_combined(all_results, output_path, top_n=5)


if __name__ == "__main__":
    main()
