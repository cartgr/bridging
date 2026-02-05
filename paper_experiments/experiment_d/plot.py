"""
Experiment D: Approval vs Approver Heterogeneity Scatter Plot.

Creates scatter plots showing candidates positioned by:
- Y-axis: Approval rate
- X-axis: Approver heterogeneity (average pairwise Hamming distance)

Top candidates by each method highlighted with colored markers.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from style import (
    setup_style, get_figure_size, despine,
    METRIC_COLORS, METRIC_MARKERS, METRIC_LABELS, COLORS
)

setup_style()

import matplotlib.pyplot as plt


def plot_single_dataset(data: dict, ax, show_legend: bool = True, title: str = None):
    """Plot scatter for one dataset on given axes."""
    approval_rates = np.array(data["approval_rates"])
    heterogeneity = np.array(data["heterogeneity"])
    candidate_names = data["candidate_names"]
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

    # Add candidate name labels
    for i in range(n_items):
        if not valid_mask[i]:
            continue
        ax.annotate(
            candidate_names[i],
            (heterogeneity[i], approval_rates[i]),
            fontsize=6,
            alpha=0.7,
            xytext=(3, 3),
            textcoords="offset points",
            zorder=3,
        )

    # Highlight top candidates by each method
    # Plot order: polis, pmean, pd (pd last so circle is on top)
    top_indices = [
        ("polis", data["top_polis"]),
        ("pmean", data["top_pmean"]),
        ("pd", data["top_pd"]),
    ]

    # Track which candidates have been plotted (for sizing - first one bigger)
    plotted_candidates = set()

    for metric_key, top_idx in top_indices:
        if top_idx < 0 or not valid_mask[top_idx]:
            continue

        x = heterogeneity[top_idx]
        y = approval_rates[top_idx]

        # First marker at this position is slightly bigger
        if top_idx not in plotted_candidates:
            size = 50
            plotted_candidates.add(top_idx)
        else:
            size = 35

        ax.scatter(
            x,
            y,
            c=METRIC_COLORS[metric_key],
            s=size,
            marker=METRIC_MARKERS[metric_key],
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
            label=METRIC_LABELS[metric_key],
        )

    # Axis labels
    ax.set_xlabel("Approver Heterogeneity (Hamming)")
    ax.set_ylabel("Approval Rate")

    if title:
        ax.set_title(title, fontsize=9)

    # Legend (reorder to show PD, Pol.is GIC, p-mean)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Reorder: PD first, then Pol.is GIC, then p-mean
        order_map = {METRIC_LABELS["pd"]: 0, METRIC_LABELS["polis"]: 1, METRIC_LABELS["pmean"]: 2}
        sorted_pairs = sorted(zip(handles, labels), key=lambda x: order_map.get(x[1], 99))
        handles, labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.9, markerscale=0.7)

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
    results_path = Path(__file__).parent.parent / "results" / "experiment_d.json"
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
            plots_dir / f"experiment_d_{dataset_id}.png",
        )

    # Combined 1x2 figure
    print("Plotting combined figure...")
    plot_combined(all_results, plots_dir / "experiment_d_combined.png")


if __name__ == "__main__":
    main()
