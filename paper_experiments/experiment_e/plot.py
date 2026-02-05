"""
Experiment E: p-mean Parameter Sensitivity Scatter Plot.

Creates scatter plots showing candidates positioned by:
- Y-axis: Approval rate
- X-axis: Approver heterogeneity (average pairwise Hamming distance)

Top candidates by each p value highlighted with colored markers,
showing how the "most bridging" candidate changes as p varies.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from style import setup_style, get_figure_size, despine, COLORS

setup_style()

import matplotlib.pyplot as plt


# P values and their display labels (using LaTeX for -inf)
P_VALUES = [1, 0, -1, -2, -5, -10, float('-inf')]
P_KEYS = ["1", "0", "-1", "-2", "-5", "-10", "-inf"]
P_DISPLAY_LABELS = [
    "$p=1$",
    "$p=0$",
    "$p=-1$",
    "$p=-2$",
    "$p=-5$",
    "$p=-10$",
    r"$p=-\infty$",
]

# Color gradient from warm (high p) to cool (low p) to black (min)
P_COLORS = {
    "1": "#E41A1C",       # Red (high p)
    "0": "#FF7F00",       # Orange
    "-1": "#FFFF33",      # Yellow
    "-2": "#4DAF4A",      # Green
    "-5": "#377EB8",      # Blue
    "-10": "#984EA3",     # Purple
    "-inf": "#000000",    # Black (min)
}


def plot_single_dataset(data: dict, ax, show_legend: bool = True, title: str = None):
    """Plot scatter for one dataset on given axes."""
    approval_rates = np.array(data["approval_rates"])
    heterogeneity = np.array(data["heterogeneity"])
    candidate_names = data["candidate_names"]
    top_by_p = data["top_by_p"]
    n_items = len(approval_rates)

    # Filter out NaN heterogeneity values for plotting
    valid_mask = ~np.isnan(heterogeneity)

    # Plot all candidates as small gray circles
    ax.scatter(
        heterogeneity[valid_mask],
        approval_rates[valid_mask],
        c=COLORS["gray"],
        s=25,
        alpha=0.5,
        zorder=2,
        label="_nolegend_",
    )

    # Highlight top candidates by each p value
    # Plot in order from p=1 to p=-inf
    # Track overlap count per candidate to progressively shrink sizes
    candidate_overlap_count = {}

    # Size settings
    MIN_SIZE = 25  # Same as grey background dots
    SIZE_STEP = 30  # Step between sizes

    # First pass: count how many p values select each candidate
    for p_key in P_KEYS:
        top_idx = top_by_p.get(p_key, -1)
        if top_idx >= 0 and valid_mask[top_idx]:
            candidate_overlap_count[top_idx] = candidate_overlap_count.get(top_idx, 0) + 1

    # Compute max size for each candidate (for label offset)
    candidate_max_size = {}
    for top_idx, count in candidate_overlap_count.items():
        candidate_max_size[top_idx] = MIN_SIZE + SIZE_STEP * (count - 1)

    # Track current overlap level for each candidate as we plot
    candidate_current_level = {}

    for p_key, p_label in zip(P_KEYS, P_DISPLAY_LABELS):
        top_idx = top_by_p.get(p_key, -1)
        if top_idx < 0 or not valid_mask[top_idx]:
            continue

        x = heterogeneity[top_idx]
        y = approval_rates[top_idx]

        # Get current level and total overlaps for this candidate
        current_level = candidate_current_level.get(top_idx, 0)
        total_overlaps = candidate_overlap_count[top_idx]

        # Compute size: largest first, smallest (MIN_SIZE) last
        # Each level decreases by SIZE_STEP
        max_size = MIN_SIZE + SIZE_STEP * (total_overlaps - 1)
        size = max_size - SIZE_STEP * current_level

        candidate_current_level[top_idx] = current_level + 1

        ax.scatter(
            x,
            y,
            c=P_COLORS[p_key],
            s=size,
            marker="o",
            edgecolors="none",
            zorder=10 + current_level,  # Higher zorder for smaller (later) dots
            label=p_label,
        )

    # Add candidate name labels (after computing sizes so we can offset properly)
    for i in range(n_items):
        if not valid_mask[i]:
            continue
        # Offset based on the largest dot size for this candidate
        max_size = candidate_max_size.get(i, 25)
        # Convert size to approximate radius in points, then to offset
        offset_x = 3 + np.sqrt(max_size) / 2
        ax.annotate(
            candidate_names[i],
            (heterogeneity[i], approval_rates[i]),
            fontsize=6,
            alpha=0.7,
            xytext=(offset_x, 3),
            textcoords="offset points",
            zorder=3,
        )

    # Axis labels
    ax.set_xlabel("Approver Heterogeneity (Hamming)")
    ax.set_ylabel("Approval Rate")

    if title:
        ax.set_title(title, fontsize=9)

    # Legend with uniform marker sizes
    if show_legend:
        leg = ax.legend(
            loc="upper right",
            fontsize=6,
            framealpha=0.9,
            ncol=1,
        )
        # Set all legend markers to the same size
        # Use legendHandles for older matplotlib, legend_handles for newer
        handles = getattr(leg, 'legend_handles', None) or leg.legendHandles
        for handle in handles:
            handle.set_sizes([40])

    # Grid
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)


def plot_combined(all_results: dict, output_path: Path):
    """Create 1x2 combined figure for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    dataset_order = ["00026-combined", "00071-combined"]

    for idx, dataset_id in enumerate(dataset_order):
        data = all_results[dataset_id]
        ax = axes[idx]
        plot_single_dataset(
            data,
            ax,
            show_legend=(idx == 0),
            title=data["label"],
        )

    plt.tight_layout()

    # Save to png, svg, and pdf subfolders
    plots_dir = output_path.parent
    png_path = plots_dir / "png" / output_path.name
    svg_path = plots_dir / "svg" / output_path.with_suffix(".svg").name
    pdf_path = plots_dir / "pdf" / output_path.with_suffix(".pdf").name

    for p in [png_path, svg_path, pdf_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path.name}")


def plot_individual(data: dict, dataset_id: str, output_path: Path):
    """Create individual plot for one dataset."""
    fig, ax = plt.subplots(figsize=get_figure_size("single", aspect=0.85))

    plot_single_dataset(data, ax, show_legend=True, title=data["label"])

    plt.tight_layout()

    # Save to png, svg, and pdf subfolders
    plots_dir = output_path.parent
    png_path = plots_dir / "png" / output_path.name
    svg_path = plots_dir / "svg" / output_path.with_suffix(".svg").name
    pdf_path = plots_dir / "pdf" / output_path.with_suffix(".pdf").name

    for p in [png_path, svg_path, pdf_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png_path.name}")


def main():
    results_path = Path(__file__).parent.parent / "results" / "experiment_e.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    # Individual plots per dataset
    for dataset_id, data in all_results.items():
        print(f"Plotting {dataset_id}...")
        plot_individual(
            data,
            dataset_id,
            plots_dir / f"experiment_e_{dataset_id}.png",
        )

    # Combined 1x2 figure
    print("Plotting combined figure...")
    plot_combined(all_results, plots_dir / "experiment_e_combined.png")


if __name__ == "__main__":
    main()
