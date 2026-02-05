"""
Appendix Experiment: Plots for naive PD vs IPW-corrected PD.

Produces Spearman/Kendall correlation plots comparing:
- PD (naive) - standard naive estimator
- PD (IPW) - IPW-corrected estimator with known q
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add parent to path for style import
sys.path.insert(0, str(Path(__file__).parent.parent))

from style import (
    setup_style, get_figure_size, despine,
    COLORS,
)

setup_style()

import matplotlib.pyplot as plt


# Colors for this experiment
ESTIMATOR_COLORS = {
    "naive": COLORS["blue"],
    "ipw": COLORS["orange"],
}

ESTIMATOR_MARKERS = {
    "naive": "o",
    "ipw": "s",
}

ESTIMATOR_LABELS = {
    "naive": "PD (naive)",
    "ipw": "PD (IPW)",
}

MEASURE_CONFIG = {
    "spearman": {"ylabel": r"Spearman $\rho$", "ylim": (0, 1.08)},
    "kendall": {"ylabel": r"Kendall $\tau$", "ylim": (0, 1.08)},
    "top1": {"ylabel": "Top-1 Accuracy", "ylim": (0, 1.08)},
}


def plot_measure(results_by_rate, measure_key, output_path, show_legend=True):
    """Plot naive vs IPW with std bands."""
    config = MEASURE_CONFIG[measure_key]

    fig, ax = plt.subplots(figsize=get_figure_size("single", aspect=0.75))

    for est_key in ["naive", "ipw"]:
        rates = sorted(results_by_rate.keys())
        means = []
        stds = []
        for rate in rates:
            vals = [r[f"{measure_key}_{est_key}"] for r in results_by_rate[rate]
                    if not np.isnan(r.get(f"{measure_key}_{est_key}", np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = ESTIMATOR_COLORS[est_key]
        marker = ESTIMATOR_MARKERS[est_key]
        label = ESTIMATOR_LABELS[est_key]

        # Plot line with markers
        ax.plot(
            rates, means,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=5,
            markeredgecolor=color,
            markeredgewidth=0,
        )

        # Confidence band
        ax.fill_between(
            rates,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

    ax.set_xlabel("Observation Rate")
    ax.set_ylabel(config["ylabel"])
    ax.set_ylim(config["ylim"])
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])

    if show_legend:
        ax.legend(loc="lower right")

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)

    # Save to png and svg subfolders
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


def plot_combined(all_results, measure_key, output_path):
    """Create combined 1x2 figure for both datasets."""
    config = MEASURE_CONFIG[measure_key]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))

    dataset_ids = ["00026-combined", "00071-combined"]
    dataset_labels = ["2002", "2007"]

    for col_idx, dataset_id in enumerate(dataset_ids):
        data = all_results[dataset_id]
        ax = axes[col_idx]

        # Group results by rate
        results_by_rate = defaultdict(list)
        for r in data["results"]:
            results_by_rate[r["target_obs_rate"]].append(r)

        for est_key in ["naive", "ipw"]:
            rates = sorted(results_by_rate.keys())
            means = []
            stds = []
            for rate in rates:
                vals = [r[f"{measure_key}_{est_key}"] for r in results_by_rate[rate]
                        if not np.isnan(r.get(f"{measure_key}_{est_key}", np.nan))]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)

            color = ESTIMATOR_COLORS[est_key]
            marker = ESTIMATOR_MARKERS[est_key]
            label = ESTIMATOR_LABELS[est_key]

            ax.plot(
                rates, means,
                marker=marker,
                color=color,
                label=label,
                linewidth=1.2,
                markersize=4,
                markeredgecolor=color,
                markeredgewidth=0,
            )

            ax.fill_between(
                rates,
                means - stds,
                means + stds,
                color=color,
                alpha=0.15,
                linewidth=0,
                zorder=1,
            )

        ax.set_xlabel("Observation Rate", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(config["ylabel"], fontsize=8)
        ax.set_ylim(config["ylim"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(labelsize=7)
        ax.set_title(dataset_labels[col_idx], fontsize=9)

        if col_idx == 1:
            ax.legend(loc="lower right", fontsize=7)

        ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
        ax.set_axisbelow(True)
        despine(ax)

    plt.tight_layout()

    # Save
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
    results_path = Path(__file__).parent.parent / "results" / "appendix.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    # Individual plots per dataset
    for dataset_id, data in all_results.items():
        print(f"Plotting {dataset_id}...")

        results_by_rate = defaultdict(list)
        for r in data["results"]:
            results_by_rate[r["target_obs_rate"]].append(r)

        for measure_key in MEASURE_CONFIG.keys():
            plot_measure(
                results_by_rate,
                measure_key,
                plots_dir / f"appendix_{measure_key}_{dataset_id}.png",
            )

    # Combined figures
    print("Plotting combined figures...")
    for measure_key in MEASURE_CONFIG.keys():
        plot_combined(
            all_results,
            measure_key,
            plots_dir / f"appendix_{measure_key}_combined.png",
        )


if __name__ == "__main__":
    main()
