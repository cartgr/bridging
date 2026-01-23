"""
Visualization functions for experiment 2 (IPW vs Naive estimation on Pol.is data).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_spearman_comparison(
    results: Dict,
    output_path: Path,
    figsize: tuple = (14, 6),
) -> None:
    """
    Create bar chart comparing IPW vs Naive Spearman correlation across datasets.

    Args:
        results: loaded results JSON
        output_path: path to save the plot
        figsize: figure size
    """
    per_dataset = results["per_dataset"]

    # Extract data
    names = [d["dataset_name"].replace("00069-", "") for d in per_dataset]
    ipw_spearman = [d["ipw_metrics"]["spearman_correlation"] for d in per_dataset]
    naive_spearman = [d["naive_metrics"]["spearman_correlation"] for d in per_dataset]

    # Sort by naive spearman (descending)
    sorted_indices = np.argsort(naive_spearman)[::-1]
    names = [names[i] for i in sorted_indices]
    ipw_spearman = [ipw_spearman[i] for i in sorted_indices]
    naive_spearman = [naive_spearman[i] for i in sorted_indices]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width/2, naive_spearman, width, label="Naive", color="steelblue")
    bars2 = ax.bar(x + width/2, ipw_spearman, width, label="IPW", color="coral")

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Spearman Correlation", fontsize=12)
    ax.set_title("Bridging Score Estimation: Naive vs IPW", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add aggregate means as horizontal lines
    agg = results["aggregate"]
    ax.axhline(y=agg["naive"]["spearman_mean"], color="steelblue", linestyle="--",
               alpha=0.7, label=f"Naive mean: {agg['naive']['spearman_mean']:.3f}")
    ax.axhline(y=agg["ipw"]["spearman_mean"], color="coral", linestyle="--",
               alpha=0.7, label=f"IPW mean: {agg['ipw']['spearman_mean']:.3f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_obs_rate_vs_spearman(
    results: Dict,
    output_path: Path,
    figsize: tuple = (10, 8),
) -> None:
    """
    Create scatter plot of observation rate vs Spearman correlation.

    Args:
        results: loaded results JSON
        output_path: path to save the plot
        figsize: figure size
    """
    per_dataset = results["per_dataset"]

    obs_rates = [d["observation_stats"]["observation_rate"] for d in per_dataset]
    ipw_spearman = [d["ipw_metrics"]["spearman_correlation"] for d in per_dataset]
    naive_spearman = [d["naive_metrics"]["spearman_correlation"] for d in per_dataset]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(obs_rates, naive_spearman, s=80, alpha=0.7, label="Naive", color="steelblue", edgecolors="white")
    ax.scatter(obs_rates, ipw_spearman, s=80, alpha=0.7, label="IPW", color="coral", edgecolors="white")

    # Connect same-dataset points with lines
    for obs, naive, ipw in zip(obs_rates, naive_spearman, ipw_spearman):
        ax.plot([obs, obs], [naive, ipw], "k-", alpha=0.2, linewidth=0.5)

    ax.set_xlabel("Observation Rate", fontsize=12)
    ax.set_ylabel("Spearman Correlation with Ground Truth", fontsize=12)
    ax.set_title("Estimation Quality vs Observation Rate", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_improvement_distribution(
    results: Dict,
    output_path: Path,
    figsize: tuple = (10, 6),
) -> None:
    """
    Create histogram showing distribution of (Naive - IPW) improvement.

    Positive means naive is better.

    Args:
        results: loaded results JSON
        output_path: path to save the plot
        figsize: figure size
    """
    per_dataset = results["per_dataset"]

    improvements = []
    for d in per_dataset:
        naive_sp = d["naive_metrics"]["spearman_correlation"]
        ipw_sp = d["ipw_metrics"]["spearman_correlation"]
        improvements.append(naive_sp - ipw_sp)

    fig, ax = plt.subplots(figsize=figsize)

    # Color bars by sign
    colors = ["steelblue" if imp >= 0 else "coral" for imp in improvements]
    sorted_indices = np.argsort(improvements)[::-1]
    improvements_sorted = [improvements[i] for i in sorted_indices]
    names = [per_dataset[i]["dataset_name"].replace("00069-", "") for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    ax.bar(range(len(improvements_sorted)), improvements_sorted, color=colors_sorted, alpha=0.8)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Spearman Improvement (Naive - IPW)", fontsize=12)
    ax.set_title("Naive vs IPW: Improvement per Dataset\n(Positive = Naive better)", fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Count wins
    naive_wins = sum(1 for imp in improvements if imp > 0)
    ipw_wins = sum(1 for imp in improvements if imp < 0)
    ties = sum(1 for imp in improvements if imp == 0)
    ax.text(0.02, 0.98, f"Naive wins: {naive_wins}\nIPW wins: {ipw_wins}\nTies: {ties}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_metric_comparison(
    results: Dict,
    output_path: Path,
    figsize: tuple = (16, 5),
) -> None:
    """
    Create side-by-side plots comparing Spearman, Kendall, and RMSE.

    Args:
        results: loaded results JSON
        output_path: path to save the plot
        figsize: figure size
    """
    per_dataset = results["per_dataset"]
    agg = results["aggregate"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Sort by naive spearman
    naive_spearman = [d["naive_metrics"]["spearman_correlation"] for d in per_dataset]
    sorted_indices = np.argsort(naive_spearman)[::-1]
    names = [per_dataset[i]["dataset_name"].replace("00069-", "") for i in sorted_indices]

    x = np.arange(len(names))
    width = 0.35

    # Plot 1: Spearman
    ax = axes[0]
    naive_vals = [per_dataset[i]["naive_metrics"]["spearman_correlation"] for i in sorted_indices]
    ipw_vals = [per_dataset[i]["ipw_metrics"]["spearman_correlation"] for i in sorted_indices]
    ax.bar(x - width/2, naive_vals, width, label="Naive", color="steelblue")
    ax.bar(x + width/2, ipw_vals, width, label="IPW", color="coral")
    ax.set_ylabel("Spearman", fontsize=11)
    ax.set_title(f"Spearman Correlation\n(Naive: {agg['naive']['spearman_mean']:.3f}, IPW: {agg['ipw']['spearman_mean']:.3f})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Plot 2: Kendall
    ax = axes[1]
    naive_vals = [per_dataset[i]["naive_metrics"]["kendall_correlation"] for i in sorted_indices]
    ipw_vals = [per_dataset[i]["ipw_metrics"]["kendall_correlation"] for i in sorted_indices]
    ax.bar(x - width/2, naive_vals, width, label="Naive", color="steelblue")
    ax.bar(x + width/2, ipw_vals, width, label="IPW", color="coral")
    ax.set_ylabel("Kendall", fontsize=11)
    ax.set_title(f"Kendall Tau\n(Naive: {agg['naive']['kendall_mean']:.3f}, IPW: {agg['ipw']['kendall_mean']:.3f})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Plot 3: RMSE (log scale, exclude outliers)
    ax = axes[2]
    naive_vals = [per_dataset[i]["naive_metrics"]["rmse"] for i in sorted_indices]
    ipw_vals = [per_dataset[i]["ipw_metrics"]["rmse"] for i in sorted_indices]

    # Use log scale since RMSE varies widely
    ax.bar(x - width/2, naive_vals, width, label="Naive", color="steelblue")
    ax.bar(x + width/2, ipw_vals, width, label="IPW", color="coral")
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(f"RMSE\n(Naive: {agg['naive']['rmse_mean']:.3f}, IPW: {agg['ipw']['rmse_mean']:.3f})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dataset_sizes(
    results: Dict,
    output_path: Path,
    figsize: tuple = (12, 5),
) -> None:
    """
    Create scatter plot showing dataset sizes and observation rates.

    Args:
        results: loaded results JSON
        output_path: path to save the plot
        figsize: figure size
    """
    per_dataset = results["per_dataset"]

    n_items = [d["n_items"] for d in per_dataset]
    n_voters = [d["n_voters"] for d in per_dataset]
    obs_rates = [d["observation_stats"]["observation_rate"] for d in per_dataset]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: n_items vs n_voters
    ax = axes[0]
    scatter = ax.scatter(n_items, n_voters, c=obs_rates, cmap="viridis", s=80, alpha=0.8, edgecolors="white")
    ax.set_xlabel("Number of Items (Comments)", fontsize=12)
    ax.set_ylabel("Number of Voters", fontsize=12)
    ax.set_title("Dataset Sizes", fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Observation Rate", fontsize=10)

    # Plot 2: Total cells vs observation rate
    ax = axes[1]
    total_cells = [d["observation_stats"]["total_cells"] for d in per_dataset]
    ax.scatter(total_cells, obs_rates, s=80, alpha=0.8, color="steelblue", edgecolors="white")
    ax.set_xlabel("Total Cells (Items × Voters)", fontsize=12)
    ax.set_ylabel("Observation Rate", fontsize=12)
    ax.set_title("Sparsity vs Dataset Size", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_table(
    results: Dict,
    output_path: Path,
) -> str:
    """
    Create a markdown table summarizing results.

    Args:
        results: loaded results JSON
        output_path: path to save the table

    Returns:
        Markdown table string
    """
    per_dataset = results["per_dataset"]
    agg = results["aggregate"]

    lines = [
        "# Experiment 2: IPW vs Naive Estimation on Pol.is Data",
        "",
        "## Aggregate Results",
        "",
        "| Metric | Naive | IPW |",
        "|--------|-------|-----|",
        f"| Spearman (mean ± std) | {agg['naive']['spearman_mean']:.3f} ± {agg['naive']['spearman_std']:.3f} | {agg['ipw']['spearman_mean']:.3f} ± {agg['ipw']['spearman_std']:.3f} |",
        f"| Kendall (mean ± std) | {agg['naive']['kendall_mean']:.3f} ± {agg['naive']['kendall_std']:.3f} | {agg['ipw']['kendall_mean']:.3f} ± {agg['ipw']['kendall_std']:.3f} |",
        f"| RMSE (mean ± std) | {agg['naive']['rmse_mean']:.3f} ± {agg['naive']['rmse_std']:.3f} | {agg['ipw']['rmse_mean']:.3f} ± {agg['ipw']['rmse_std']:.3f} |",
        "",
        f"**Observation Rate**: {agg['observation_rate']['mean']:.1%} (range: {agg['observation_rate']['min']:.1%} - {agg['observation_rate']['max']:.1%})",
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Items | Voters | Obs Rate | Naive Spearman | IPW Spearman | Winner |",
        "|---------|-------|--------|----------|----------------|--------------|--------|",
    ]

    # Sort by naive spearman
    sorted_data = sorted(per_dataset, key=lambda d: -d["naive_metrics"]["spearman_correlation"])

    for d in sorted_data:
        name = d["dataset_name"].replace("00069-", "")
        n_items = d["n_items"]
        n_voters = d["n_voters"]
        obs_rate = d["observation_stats"]["observation_rate"]
        naive_sp = d["naive_metrics"]["spearman_correlation"]
        ipw_sp = d["ipw_metrics"]["spearman_correlation"]
        winner = "Naive" if naive_sp > ipw_sp else ("IPW" if ipw_sp > naive_sp else "Tie")

        lines.append(
            f"| {name} | {n_items} | {n_voters} | {obs_rate:.1%} | {naive_sp:.3f} | {ipw_sp:.3f} | {winner} |"
        )

    # Summary
    naive_wins = sum(1 for d in per_dataset if d["naive_metrics"]["spearman_correlation"] > d["ipw_metrics"]["spearman_correlation"])
    ipw_wins = sum(1 for d in per_dataset if d["ipw_metrics"]["spearman_correlation"] > d["naive_metrics"]["spearman_correlation"])
    ties = len(per_dataset) - naive_wins - ipw_wins

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Naive wins**: {naive_wins} datasets",
        f"- **IPW wins**: {ipw_wins} datasets",
        f"- **Ties**: {ties} datasets",
        "",
        "**Conclusion**: The naive estimator outperforms IPW on real Pol.is data, likely because:",
        "1. The missing data mechanism is not purely MAR (Missing At Random)",
        "2. IPW can have high variance with estimated propensity scores",
        "3. The observation patterns in Pol.is may be closer to MCAR than expected",
        "",
    ])

    table_str = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def generate_all_plots(results_path: Path, output_dir: Path) -> None:
    """
    Generate all plots for experiment 2 results.

    Args:
        results_path: path to results JSON file
        output_dir: directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    print(f"Generating plots for {len(results['per_dataset'])} datasets...")

    # Generate all plots
    plot_spearman_comparison(results, output_dir / "spearman_comparison.png")
    print("  - spearman_comparison.png")

    plot_obs_rate_vs_spearman(results, output_dir / "obs_rate_vs_spearman.png")
    print("  - obs_rate_vs_spearman.png")

    plot_improvement_distribution(results, output_dir / "improvement_distribution.png")
    print("  - improvement_distribution.png")

    plot_multi_metric_comparison(results, output_dir / "multi_metric_comparison.png")
    print("  - multi_metric_comparison.png")

    plot_dataset_sizes(results, output_dir / "dataset_sizes.png")
    print("  - dataset_sizes.png")

    create_summary_table(results, output_dir / "summary.md")
    print("  - summary.md")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    import sys

    # Default paths
    results_dir = Path(__file__).parent / "results"
    plots_dir = Path(__file__).parent / "plots"

    # Find most recent results file
    results_files = sorted(results_dir.glob("results_*.json"))
    if not results_files:
        print("No results files found in", results_dir)
        sys.exit(1)

    results_path = results_files[-1]
    print(f"Using results file: {results_path}")

    generate_all_plots(results_path, plots_dir)
