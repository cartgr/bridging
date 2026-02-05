"""
Experiment F: p-mean Robustness Plots.

Line plots showing how different p values perform under MCAR masking:
- X-axis: Observation rate
- Y-axis: Spearman correlation (or Kendall tau, or Top-1 accuracy)
- Lines: One per p value, with standard deviation bands
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from style import setup_style, get_figure_size, despine

setup_style()

import matplotlib.pyplot as plt


# P values and their display labels
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

# Markers for each p value
P_MARKERS = {
    "1": "o",
    "0": "s",
    "-1": "^",
    "-2": "D",
    "-5": "v",
    "-10": "p",
    "-inf": "*",
}

MEASURE_CONFIG = {
    "spearman": {"ylabel": r"Spearman $\rho$", "ylim": (-0.05, 1.08)},
    "kendall": {"ylabel": r"Kendall $\tau$", "ylim": (-0.05, 1.08)},
    "top1": {"ylabel": "Top-1 Accuracy", "ylim": (-0.05, 1.08)},
}

DATASET_LABELS = {
    "00026-combined": "French Election 2002",
    "00071-combined": "French Election 2007",
}


def plot_measure_on_ax(ax, results_by_rate, measure_key, show_legend=False,
                       show_xlabel=True, show_ylabel=True):
    """Plot a measure on a given axes (for combined figures)."""
    config = MEASURE_CONFIG[measure_key]

    for p_key, p_label in zip(P_KEYS, P_DISPLAY_LABELS):
        rates = sorted(results_by_rate.keys())
        means = []
        stds = []

        for rate in rates:
            vals = [r[f"{measure_key}_{p_key}"] for r in results_by_rate[rate]
                    if not np.isnan(r.get(f"{measure_key}_{p_key}", np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = P_COLORS[p_key]
        marker = P_MARKERS[p_key]

        # Plot line with markers
        ax.plot(
            rates, means,
            marker=marker,
            color=color,
            label=p_label,
            linewidth=1.2,
            markersize=4,
            markeredgecolor=color,
            markeredgewidth=0,
        )

        # Confidence band (+/- 1 std)
        ax.fill_between(
            rates,
            means - stds,
            means + stds,
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )

    if show_xlabel:
        ax.set_xlabel("Observation Rate", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(config["ylabel"], fontsize=8)
    ax.set_ylim(config["ylim"])

    # X-axis formatting
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.05, 0.25, 0.50, 0.75, 0.95])
    ax.tick_params(labelsize=7)

    # Legend
    if show_legend:
        ax.legend(loc="lower right", fontsize=6, ncol=2)

    # Light grid on y-axis only
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)


def plot_single_measure(results_by_rate, measure_key, output_path, title=None):
    """Plot a single measure for one dataset."""
    config = MEASURE_CONFIG[measure_key]

    fig, ax = plt.subplots(figsize=get_figure_size("single", aspect=0.75))

    plot_measure_on_ax(ax, results_by_rate, measure_key,
                       show_legend=True, show_xlabel=True, show_ylabel=True)

    if title:
        ax.set_title(title, fontsize=9)

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

    print(f"  Saved: {png_path.name}")


def plot_combined_2x2(all_results, output_path):
    """Create 2x2 combined figure: rows=Kendall/Top-1, cols=2002/2007."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.5))

    dataset_ids = ["00026-combined", "00071-combined"]
    measure_keys = ["kendall", "top1"]

    for col_idx, dataset_id in enumerate(dataset_ids):
        data = all_results[dataset_id]

        # Group results by observation rate
        results_by_rate = defaultdict(list)
        for r in data["mcar_results"]:
            obs_rate = round(1 - r["mask_rate"], 2)
            results_by_rate[obs_rate].append(r)

        for row_idx, measure_key in enumerate(measure_keys):
            ax = axes[row_idx, col_idx]

            plot_measure_on_ax(
                ax, results_by_rate, measure_key,
                show_legend=(col_idx == 1 and row_idx == 0),  # Legend on top-right
                show_xlabel=(row_idx == 1),  # X-label only on bottom row
                show_ylabel=(col_idx == 0),  # Y-label only on left
            )

        # Column titles
        axes[0, col_idx].set_title(DATASET_LABELS[dataset_id], fontsize=9)

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

    print(f"  Saved: {png_path.name}")


def main():
    results_path = Path(__file__).parent.parent / "results" / "experiment_f.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    # Individual plots per dataset
    for dataset_id, data in all_results.items():
        print(f"Plotting {dataset_id}...")

        # Group results by observation rate
        results_by_rate = defaultdict(list)
        for r in data["mcar_results"]:
            obs_rate = round(1 - r["mask_rate"], 2)
            results_by_rate[obs_rate].append(r)

        # Generate plots for each measure
        for measure_key in MEASURE_CONFIG.keys():
            plot_single_measure(
                results_by_rate,
                measure_key,
                plots_dir / f"experiment_f_{measure_key}_{dataset_id}.png",
                title=DATASET_LABELS[dataset_id],
            )

    # Combined 2x2 figure
    print("Plotting combined figure...")
    plot_combined_2x2(all_results, plots_dir / "experiment_f_combined.png")


if __name__ == "__main__":
    main()
