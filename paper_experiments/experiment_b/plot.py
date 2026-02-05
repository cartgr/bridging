"""
Experiment B: Robustness plots.

Produces three plot types per experiment (MCAR / routing) per dataset:
1. Spearman rho vs observation rate
2. Kendall tau vs observation rate
3. Top-1 frequency vs observation rate

Each with +/-1 std bands for PD, Pol.is GIC, and p-mean.
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
    METRIC_COLORS, METRIC_MARKERS, METRIC_LABELS, COLORS
)

setup_style()

import matplotlib.pyplot as plt


MEASURE_CONFIG = {
    "spearman": {"ylabel": r"Spearman's $\rho$", "ylim": (0, 1.08)},
    "kendall": {"ylabel": r"Kendall's $\tau$", "ylim": (0, 1.08)},
    "top1": {"ylabel": "Top-1 Accuracy", "ylim": (0, 1.08)},
}


def plot_measure(results_by_rate, measure_key, output_path, show_legend=True):
    """Plot a measure vs observation rate with std bands."""
    config = MEASURE_CONFIG[measure_key]

    fig, ax = plt.subplots(figsize=get_figure_size("single", aspect=0.75))

    # Plot order and marker sizes - p-mean larger, PD plotted last (on top)
    metric_sizes = {"pd": 5, "polis": 5, "pmean": 6}

    for metric_key in ["pmean", "polis", "pd"]:
        rates = sorted(results_by_rate.keys())
        means = []
        stds = []
        for rate in rates:
            vals = [r[f"{measure_key}_{metric_key}"] for r in results_by_rate[rate]
                    if not np.isnan(r.get(f"{measure_key}_{metric_key}", np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = METRIC_COLORS[metric_key]
        marker = METRIC_MARKERS[metric_key]
        label = METRIC_LABELS[metric_key]
        size = metric_sizes[metric_key]

        # Plot line with markers
        ax.plot(
            rates, means,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=size,
            markeredgecolor=color,
            markeredgewidth=0,
        )

        # Confidence band (+/- 1 std)
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

    # X-axis formatting
    ax.set_xlim(0.25, 0.95)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])

    # Legend
    if show_legend:
        ax.legend(loc="lower right")

    # Light grid on y-axis only
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)

    # Save to png and svg subfolders
    png_path = output_path.parent / "png" / output_path.name
    svg_path = output_path.parent / "svg" / output_path.with_suffix(".svg").name
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {png_path.name}")


def plot_measure_on_ax(ax, results_by_rate, measure_key, show_legend=False, show_xlabel=True, show_ylabel=True):
    """Plot a measure on a given axes (for combined figures)."""
    config = MEASURE_CONFIG[measure_key]

    # Plot order and marker sizes - p-mean larger, PD plotted last (on top)
    metric_sizes = {"pd": 4, "polis": 4, "pmean": 5}

    for metric_key in ["pmean", "polis", "pd"]:
        rates = sorted(results_by_rate.keys())
        means = []
        stds = []
        for rate in rates:
            vals = [r[f"{measure_key}_{metric_key}"] for r in results_by_rate[rate]
                    if not np.isnan(r.get(f"{measure_key}_{metric_key}", np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = METRIC_COLORS[metric_key]
        marker = METRIC_MARKERS[metric_key]
        label = METRIC_LABELS[metric_key]
        size = metric_sizes[metric_key]

        # Plot line with markers
        ax.plot(
            rates, means,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.2,
            markersize=size,
            markeredgecolor=color,
            markeredgewidth=0,
        )

        # Confidence band (+/- 1 std)
        ax.fill_between(
            rates,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

    if show_xlabel:
        ax.set_xlabel("Observation Rate", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(config["ylabel"], fontsize=8)
    ax.set_ylim(config["ylim"])

    # X-axis formatting
    ax.set_xlim(0.25, 0.95)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.tick_params(labelsize=7)

    # Legend
    if show_legend:
        ax.legend(loc="lower right", fontsize=7)

    # Light grid on y-axis only
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)


def plot_combined_2x2_by_experiment(all_results, experiment_type, output_path):
    """Create 2x2 combined figure: rows=Kendall/Top-1, cols=2002/2007."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.2))

    dataset_ids = ["00026-combined", "00071-combined"]
    dataset_labels = ["French Election 2002", "French Election 2007"]
    measure_keys = ["kendall", "top1"]

    for col_idx, dataset_id in enumerate(dataset_ids):
        data = all_results[dataset_id]

        # Group results by rate
        if experiment_type == "mcar":
            results_by_rate = defaultdict(list)
            for r in data["mcar_results"]:
                obs_rate = round(1 - r["mask_rate"], 2)
                results_by_rate[obs_rate].append(r)
        else:  # routing
            results_by_rate = defaultdict(list)
            for r in data["routing_results"]:
                results_by_rate[r["target_obs_rate"]].append(r)

        for row_idx, measure_key in enumerate(measure_keys):
            config = MEASURE_CONFIG[measure_key]
            ax = axes[row_idx, col_idx]

            plot_measure_on_ax(
                ax, results_by_rate, measure_key,
                show_legend=(col_idx == 1 and row_idx == 0),  # Legend only on top-right
                show_xlabel=(row_idx == 1),  # X-label only on bottom row
                show_ylabel=(col_idx == 0),  # Y-label only on left
            )

        # Column titles
        axes[0, col_idx].set_title(dataset_labels[col_idx], fontsize=9)

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


def plot_measure_on_ax_with_ylim(ax, results_by_rate, measure_key, ylim, show_legend=False, show_xlabel=True, show_ylabel=True):
    """Plot a measure on a given axes with custom ylim."""
    config = MEASURE_CONFIG[measure_key]

    # Plot order and marker sizes - p-mean larger, PD plotted last (on top)
    metric_sizes = {"pd": 4, "polis": 4, "pmean": 5}

    for metric_key in ["pmean", "polis", "pd"]:
        rates = sorted(results_by_rate.keys())
        means = []
        stds = []
        for rate in rates:
            vals = [r[f"{measure_key}_{metric_key}"] for r in results_by_rate[rate]
                    if not np.isnan(r.get(f"{measure_key}_{metric_key}", np.nan))]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(np.nan)
                stds.append(0)

        means = np.array(means)
        stds = np.array(stds)

        color = METRIC_COLORS[metric_key]
        marker = METRIC_MARKERS[metric_key]
        label = METRIC_LABELS[metric_key]
        size = metric_sizes[metric_key]

        # Plot line with markers
        ax.plot(
            rates, means,
            marker=marker,
            color=color,
            label=label,
            linewidth=1.2,
            markersize=size,
            markeredgecolor=color,
            markeredgewidth=0,
        )

        # Confidence band (+/- 1 std)
        ax.fill_between(
            rates,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

    if show_xlabel:
        ax.set_xlabel("Observation Rate", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(config["ylabel"], fontsize=8)
    ax.set_ylim(ylim)

    # X-axis formatting
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.05, 0.25, 0.50, 0.75, 0.95])
    ax.tick_params(labelsize=7)

    # Legend
    if show_legend:
        ax.legend(loc="lower right", fontsize=7)

    # Light grid on y-axis only
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    despine(ax)


def plot_combined_2x2_by_missingness(all_results, output_path):
    """Create 2x2 combined figure: rows=MCAR/Routing, cols=2002/2007, Kendall tau only."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.5))

    dataset_ids = ["00026-combined", "00071-combined"]
    dataset_labels = ["French Election 2002", "French Election 2007"]
    experiment_types = ["mcar", "routing"]
    row_labels = ["Kendall's $\\tau$\nunder MCAR", "Kendall's $\\tau$\nunder Pol.is Routing"]

    # Use different y-limits for MCAR vs routing (routing can go negative)
    ylims = {
        "mcar": (-0.1, 1.08),
        "routing": (-0.5, 1.08),
    }

    for col_idx, dataset_id in enumerate(dataset_ids):
        data = all_results[dataset_id]

        for row_idx, experiment_type in enumerate(experiment_types):
            ax = axes[row_idx, col_idx]

            # Group results by rate
            if experiment_type == "mcar":
                results_by_rate = defaultdict(list)
                for r in data["mcar_results"]:
                    obs_rate = round(1 - r["mask_rate"], 2)
                    results_by_rate[obs_rate].append(r)
            else:  # routing
                results_by_rate = defaultdict(list)
                for r in data["routing_results"]:
                    results_by_rate[r["target_obs_rate"]].append(r)

            plot_measure_on_ax_with_ylim(
                ax, results_by_rate, "kendall", ylims[experiment_type],
                show_legend=(col_idx == 1 and row_idx == 0),  # Legend only on top-right
                show_xlabel=(row_idx == 1),  # X-label only on bottom row
                show_ylabel=False,  # We'll add a shared ylabel
            )

        # Column titles
        axes[0, col_idx].set_title(dataset_labels[col_idx], fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)

    # Add row labels using fig.text for consistent horizontal alignment
    for row_idx, row_label in enumerate(row_labels):
        # Get the vertical center of each row
        ax = axes[row_idx, 0]
        bbox = ax.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.05, y_center, row_label, fontsize=8, ha='center', va='center', rotation=90)

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
    results_path = Path(__file__).parent.parent / "results" / "experiment_b.json"
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        all_results = json.load(f)

    for dataset_id, data in all_results.items():
        print(f"Plotting {dataset_id}...")

        # Group results by rate
        mcar_by_rate = defaultdict(list)
        for r in data["mcar_results"]:
            obs_rate = round(1 - r["mask_rate"], 2)
            mcar_by_rate[obs_rate].append(r)

        routing_by_rate = defaultdict(list)
        for r in data["routing_results"]:
            routing_by_rate[r["target_obs_rate"]].append(r)

        # Generate all plot types
        for measure_key in MEASURE_CONFIG.keys():
            # MCAR
            plot_measure(
                mcar_by_rate,
                measure_key,
                plots_dir / f"experiment_b_mcar_{measure_key}_{dataset_id}.png",
            )
            # Routing
            plot_measure(
                routing_by_rate,
                measure_key,
                plots_dir / f"experiment_b_routing_{measure_key}_{dataset_id}.png",
            )

    # Generate combined 2x2 figures by experiment type
    print("Plotting combined figures...")
    plot_combined_2x2_by_experiment(all_results, "mcar", plots_dir / "experiment_b_mcar_combined.png")
    plot_combined_2x2_by_experiment(all_results, "routing", plots_dir / "experiment_b_routing_combined.png")

    # Generate combined 2x2 figure with rows=missingness model, Kendall tau only
    print("Plotting combined Kendall tau figure...")
    plot_combined_2x2_by_missingness(all_results, plots_dir / "experiment_b_combined.png")


if __name__ == "__main__":
    main()
