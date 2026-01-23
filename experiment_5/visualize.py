"""
Visualization functions for robustness experiment results.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_robustness_comparison(
    aggregated: Dict,
    output_path: Path,
    metric: str = "spearman",
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot metric vs mask rate for all methods with error bands.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the plot
        metric: which metric to plot ("spearman", "kendall", "pearson", "rmse", "mae")
        title: optional custom title
        figsize: figure size
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]

    # Extract metric values
    bridging_means = [bridging[r][f"{metric}_mean"] for r in mask_rates]
    bridging_stds = [bridging[r][f"{metric}_std"] for r in mask_rates]
    polis_means = [polis[r][f"{metric}_mean"] for r in mask_rates]
    polis_stds = [polis[r][f"{metric}_std"] for r in mask_rates]

    # Convert mask rate to observation rate for x-axis
    obs_rates = [1 - r for r in mask_rates]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bridging (PD)
    ax.plot(obs_rates, bridging_means, "b-", label="PD Bridging", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(bridging_means) - np.array(bridging_stds),
        np.array(bridging_means) + np.array(bridging_stds),
        color="blue",
        alpha=0.15,
    )

    # Plot p-norm if available
    if pnorm:
        pnorm_means = [pnorm[r].get(f"{metric}_mean", np.nan) for r in mask_rates]
        pnorm_stds = [pnorm[r].get(f"{metric}_std", 0) for r in mask_rates]
        if not all(np.isnan(pnorm_means)):
            ax.plot(obs_rates, pnorm_means, "g-", label="p-norm (p=-10)", linewidth=2)
            ax.fill_between(
                obs_rates,
                np.array(pnorm_means) - np.array(pnorm_stds),
                np.array(pnorm_means) + np.array(pnorm_stds),
                color="green",
                alpha=0.15,
            )

    # Plot polis
    ax.plot(obs_rates, polis_means, "r-", label="Polis Consensus", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(polis_means) - np.array(polis_stds),
        np.array(polis_means) + np.array(polis_stds),
        color="red",
        alpha=0.15,
    )

    # Labels and formatting
    ax.set_xlabel("Observation Rate", fontsize=12)
    metric_labels = {
        "spearman": "Spearman Correlation",
        "kendall": "Kendall Tau",
        "pearson": "Pearson Correlation",
        "rmse": "RMSE",
        "mae": "MAE",
    }
    ax.set_ylabel(metric_labels.get(metric, metric.title()), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{metric_labels.get(metric, metric)} vs Observation Rate", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(obs_rates) - 0.05, max(obs_rates) + 0.05)

    # Invert x-axis so higher observation rate is on left
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_variance_comparison(
    aggregated: Dict,
    output_path: Path,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot estimate variance vs mask rate for all methods.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the plot
        title: optional custom title
        figsize: figure size
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]

    # Extract variance values
    bridging_vars = [bridging[r]["mean_estimate_variance"] for r in mask_rates]
    polis_vars = [polis[r]["mean_estimate_variance"] for r in mask_rates]

    obs_rates = [1 - r for r in mask_rates]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(obs_rates, bridging_vars, "b-o", label="PD Bridging", linewidth=2)

    # Plot p-norm if available
    if pnorm:
        pnorm_vars = [pnorm[r].get("mean_estimate_variance", np.nan) for r in mask_rates]
        if not all(np.isnan(pnorm_vars)):
            ax.plot(obs_rates, pnorm_vars, "g-o", label="p-norm (p=-10)", linewidth=2)

    ax.plot(obs_rates, polis_vars, "r-o", label="Polis Consensus", linewidth=2)

    ax.set_xlabel("Observation Rate", fontsize=12)
    ax.set_ylabel("Mean Estimate Variance", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Estimate Variance vs Observation Rate", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_scatter(
    gt_scores: np.ndarray,
    est_scores: np.ndarray,
    output_path: Path,
    title: str = "Estimated vs Ground Truth",
    xlabel: str = "Ground Truth",
    ylabel: str = "Estimated",
    figsize: tuple = (8, 8),
) -> None:
    """
    Create scatter plot of estimated vs ground truth scores.

    Args:
        gt_scores: (n_items,) ground truth scores
        est_scores: (n_items,) estimated scores
        output_path: path to save the plot
        title: plot title
        xlabel: x-axis label
        ylabel: y-axis label
        figsize: figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(gt_scores, est_scores, alpha=0.6, edgecolors="none")

    # Add diagonal line
    min_val = min(gt_scores.min(), est_scores.min())
    max_val = max(gt_scores.max(), est_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="y = x")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Make axes equal
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_metric(
    aggregated: Dict,
    output_path: Path,
    metrics: List[str] = ["spearman", "rmse"],
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Create side-by-side plots of multiple metrics.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the plot
        metrics: list of metrics to plot
        title: optional overall title
        figsize: figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]
    obs_rates = [1 - r for r in mask_rates]

    metric_labels = {
        "spearman": "Spearman Correlation",
        "kendall": "Kendall Tau",
        "pearson": "Pearson Correlation",
        "rmse": "RMSE",
        "mae": "MAE",
    }

    for ax, metric in zip(axes, metrics):
        bridging_means = [bridging[r][f"{metric}_mean"] for r in mask_rates]
        bridging_stds = [bridging[r][f"{metric}_std"] for r in mask_rates]
        polis_means = [polis[r][f"{metric}_mean"] for r in mask_rates]
        polis_stds = [polis[r][f"{metric}_std"] for r in mask_rates]

        ax.plot(obs_rates, bridging_means, "b-", label="PD Bridging", linewidth=2)
        ax.fill_between(
            obs_rates,
            np.array(bridging_means) - np.array(bridging_stds),
            np.array(bridging_means) + np.array(bridging_stds),
            color="blue",
            alpha=0.15,
        )

        # Plot p-norm if available
        if pnorm:
            pnorm_means = [pnorm[r].get(f"{metric}_mean", np.nan) for r in mask_rates]
            pnorm_stds = [pnorm[r].get(f"{metric}_std", 0) for r in mask_rates]
            if not all(np.isnan(pnorm_means)):
                ax.plot(obs_rates, pnorm_means, "g-", label="p-norm (p=-10)", linewidth=2)
                ax.fill_between(
                    obs_rates,
                    np.array(pnorm_means) - np.array(pnorm_stds),
                    np.array(pnorm_means) + np.array(pnorm_stds),
                    color="green",
                    alpha=0.15,
                )

        ax.plot(obs_rates, polis_means, "r-", label="Polis Consensus", linewidth=2)
        ax.fill_between(
            obs_rates,
            np.array(polis_means) - np.array(polis_stds),
            np.array(polis_means) + np.array(polis_stds),
            color="red",
            alpha=0.15,
        )

        ax.set_xlabel("Observation Rate", fontsize=11)
        ax.set_ylabel(metric_labels.get(metric, metric.title()), fontsize=11)
        ax.set_title(metric_labels.get(metric, metric.title()), fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_table(
    aggregated: Dict,
    output_path: Path,
) -> str:
    """
    Create a markdown table summarizing results.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the table (as .md file)

    Returns:
        Markdown table string
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]

    has_pnorm = bool(pnorm) and any(
        not np.isnan(pnorm[r].get("spearman_mean", np.nan)) for r in mask_rates
    )

    if has_pnorm:
        lines = [
            "# Robustness Comparison: PD Bridging vs p-norm vs Polis Consensus",
            "",
            "## Results by Observation Rate",
            "",
            "| Obs Rate | PD Spearman | p-norm Spearman | Polis Spearman | PD RMSE | p-norm RMSE | Polis RMSE | Polis k |",
            "|----------|-------------|-----------------|----------------|---------|-------------|------------|---------|",
        ]
    else:
        lines = [
            "# Robustness Comparison: PD Bridging vs Polis Consensus",
            "",
            "## Results by Observation Rate",
            "",
            "| Obs Rate | PD Spearman | Polis Spearman | PD RMSE | Polis RMSE | Polis k |",
            "|----------|-------------|----------------|---------|------------|---------|",
        ]

    for mask_rate in mask_rates:
        obs_rate = 1 - mask_rate
        b_spearman = bridging[mask_rate]["spearman_mean"]
        b_spearman_std = bridging[mask_rate]["spearman_std"]
        p_spearman = polis[mask_rate]["spearman_mean"]
        p_spearman_std = polis[mask_rate]["spearman_std"]
        b_rmse = bridging[mask_rate]["rmse_mean"]
        p_rmse = polis[mask_rate]["rmse_mean"]
        p_k = polis[mask_rate]["k_mean"]

        if has_pnorm:
            pn_spearman = pnorm[mask_rate].get("spearman_mean", np.nan)
            pn_spearman_std = pnorm[mask_rate].get("spearman_std", np.nan)
            pn_rmse = pnorm[mask_rate].get("rmse_mean", np.nan)
            lines.append(
                f"| {obs_rate:.0%} | {b_spearman:.3f} ({b_spearman_std:.3f}) | "
                f"{pn_spearman:.3f} ({pn_spearman_std:.3f}) | "
                f"{p_spearman:.3f} ({p_spearman_std:.3f}) | {b_rmse:.4f} | "
                f"{pn_rmse:.4f} | {p_rmse:.4f} | {p_k:.1f} |"
            )
        else:
            lines.append(
                f"| {obs_rate:.0%} | {b_spearman:.3f} ({b_spearman_std:.3f}) | "
                f"{p_spearman:.3f} ({p_spearman_std:.3f}) | {b_rmse:.4f} | "
                f"{p_rmse:.4f} | {p_k:.1f} |"
            )

    lines.extend([
        "",
        "## Legend",
        "- **Obs Rate**: Fraction of votes observed (1 - mask_rate)",
        "- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)",
        "- **RMSE**: Root mean squared error from ground truth",
        "- **Polis k**: Average number of clusters selected by Polis",
        "",
    ])

    table_str = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def plot_top_k_precision(
    aggregated: Dict,
    output_path: Path,
    k: int = 3,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot top-k precision vs observation rate.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the plot
        k: k value for top-k precision
        title: optional custom title
        figsize: figure size
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]

    bridging_means = [bridging[r][f"top_{k}_precision_mean"] for r in mask_rates]
    bridging_stds = [bridging[r][f"top_{k}_precision_std"] for r in mask_rates]
    polis_means = [polis[r][f"top_{k}_precision_mean"] for r in mask_rates]
    polis_stds = [polis[r][f"top_{k}_precision_std"] for r in mask_rates]

    obs_rates = [1 - r for r in mask_rates]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(obs_rates, bridging_means, "b-o", label="PD Bridging", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(bridging_means) - np.array(bridging_stds),
        np.array(bridging_means) + np.array(bridging_stds),
        color="blue",
        alpha=0.15,
    )

    # Plot p-norm if available
    if pnorm:
        pnorm_means = [pnorm[r].get(f"top_{k}_precision_mean", np.nan) for r in mask_rates]
        pnorm_stds = [pnorm[r].get(f"top_{k}_precision_std", 0) for r in mask_rates]
        if not all(np.isnan(pnorm_means)):
            ax.plot(obs_rates, pnorm_means, "g-o", label="p-norm (p=-10)", linewidth=2)
            ax.fill_between(
                obs_rates,
                np.array(pnorm_means) - np.array(pnorm_stds),
                np.array(pnorm_means) + np.array(pnorm_stds),
                color="green",
                alpha=0.15,
            )

    ax.plot(obs_rates, polis_means, "r-o", label="Polis Consensus", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(polis_means) - np.array(polis_stds),
        np.array(polis_means) + np.array(polis_stds),
        color="red",
        alpha=0.15,
    )

    ax.set_xlabel("Observation Rate", fontsize=12)
    ax.set_ylabel(f"Top-{k} Precision", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Top-{k} Precision vs Observation Rate", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_simulation_comparison(
    aggregated: Dict,
    output_path: Path,
    metric: str = "spearman",
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Plot metric vs observation rate for all three methods (simulation experiment).

    Args:
        aggregated: output from aggregate_simulation_results
        output_path: path to save the plot
        metric: which metric to plot
        title: optional custom title
        figsize: figure size
    """
    distribution_names = aggregated["distribution_names"]
    bridging_naive = aggregated["bridging_naive"]
    bridging_ipw = aggregated["bridging_ipw"]
    polis = aggregated["polis"]

    # Get observation rates for x-axis
    obs_rates = [bridging_naive[d]["observation_rate_mean"] for d in distribution_names]

    # Extract metric values
    naive_means = [bridging_naive[d][f"{metric}_mean"] for d in distribution_names]
    naive_stds = [bridging_naive[d][f"{metric}_std"] for d in distribution_names]
    ipw_means = [bridging_ipw[d][f"{metric}_mean"] for d in distribution_names]
    ipw_stds = [bridging_ipw[d][f"{metric}_std"] for d in distribution_names]
    polis_means = [polis[d][f"{metric}_mean"] for d in distribution_names]
    polis_stds = [polis[d][f"{metric}_std"] for d in distribution_names]

    # Sort by observation rate
    sorted_indices = np.argsort(obs_rates)[::-1]
    obs_rates = [obs_rates[i] for i in sorted_indices]
    naive_means = [naive_means[i] for i in sorted_indices]
    naive_stds = [naive_stds[i] for i in sorted_indices]
    ipw_means = [ipw_means[i] for i in sorted_indices]
    ipw_stds = [ipw_stds[i] for i in sorted_indices]
    polis_means = [polis_means[i] for i in sorted_indices]
    polis_stds = [polis_stds[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bridging naive
    ax.plot(obs_rates, naive_means, "b-", label="Bridging (Naive)", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(naive_means) - np.array(naive_stds),
        np.array(naive_means) + np.array(naive_stds),
        color="blue",
        alpha=0.15,
    )

    # Plot bridging IPW
    ax.plot(obs_rates, ipw_means, "g-", label="Bridging (IPW)", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(ipw_means) - np.array(ipw_stds),
        np.array(ipw_means) + np.array(ipw_stds),
        color="green",
        alpha=0.15,
    )

    # Plot polis
    ax.plot(obs_rates, polis_means, "r-", label="Polis Consensus", linewidth=2)
    ax.fill_between(
        obs_rates,
        np.array(polis_means) - np.array(polis_stds),
        np.array(polis_means) + np.array(polis_stds),
        color="red",
        alpha=0.15,
    )

    metric_labels = {
        "spearman": "Spearman Correlation",
        "kendall": "Kendall Tau",
        "pearson": "Pearson Correlation",
        "rmse": "RMSE",
        "mae": "MAE",
    }

    ax.set_xlabel("Observation Rate", fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric.title()), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{metric_labels.get(metric, metric)} vs Observation Rate (Simulated)", fontsize=14)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_simulation_multi_metric(
    aggregated: Dict,
    output_path: Path,
    metrics: List[str] = ["spearman", "rmse"],
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Create side-by-side plots of multiple metrics for simulation experiment.

    Args:
        aggregated: output from aggregate_simulation_results
        output_path: path to save the plot
        metrics: list of metrics to plot
        title: optional overall title
        figsize: figure size
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    distribution_names = aggregated["distribution_names"]
    bridging_naive = aggregated["bridging_naive"]
    bridging_ipw = aggregated["bridging_ipw"]
    polis = aggregated["polis"]

    obs_rates = [bridging_naive[d]["observation_rate_mean"] for d in distribution_names]
    sorted_indices = np.argsort(obs_rates)[::-1]
    obs_rates = [obs_rates[i] for i in sorted_indices]

    metric_labels = {
        "spearman": "Spearman Correlation",
        "kendall": "Kendall Tau",
        "pearson": "Pearson Correlation",
        "rmse": "RMSE",
        "mae": "MAE",
    }

    for ax, metric in zip(axes, metrics):
        naive_means = [bridging_naive[distribution_names[i]][f"{metric}_mean"] for i in sorted_indices]
        naive_stds = [bridging_naive[distribution_names[i]][f"{metric}_std"] for i in sorted_indices]
        ipw_means = [bridging_ipw[distribution_names[i]][f"{metric}_mean"] for i in sorted_indices]
        ipw_stds = [bridging_ipw[distribution_names[i]][f"{metric}_std"] for i in sorted_indices]
        polis_means = [polis[distribution_names[i]][f"{metric}_mean"] for i in sorted_indices]
        polis_stds = [polis[distribution_names[i]][f"{metric}_std"] for i in sorted_indices]

        ax.plot(obs_rates, naive_means, "b-", label="Bridging (Naive)", linewidth=2)
        ax.fill_between(
            obs_rates,
            np.array(naive_means) - np.array(naive_stds),
            np.array(naive_means) + np.array(naive_stds),
            color="blue",
            alpha=0.15,
        )

        ax.plot(obs_rates, ipw_means, "g-", label="Bridging (IPW)", linewidth=2)
        ax.fill_between(
            obs_rates,
            np.array(ipw_means) - np.array(ipw_stds),
            np.array(ipw_means) + np.array(ipw_stds),
            color="green",
            alpha=0.15,
        )

        ax.plot(obs_rates, polis_means, "r-", label="Polis Consensus", linewidth=2)
        ax.fill_between(
            obs_rates,
            np.array(polis_means) - np.array(polis_stds),
            np.array(polis_means) + np.array(polis_stds),
            color="red",
            alpha=0.15,
        )

        ax.set_xlabel("Observation Rate", fontsize=11)
        ax.set_ylabel(metric_labels.get(metric, metric.title()), fontsize=11)
        ax.set_title(metric_labels.get(metric, metric.title()), fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranking_stability(
    aggregated: Dict,
    output_path: Path,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Plot ranking stability metrics vs observation rate (MCAR experiment).

    Creates side-by-side plots of top-1 frequency and rank correlation.

    Args:
        aggregated: output from aggregate_by_mask_rate
        output_path: path to save the plot
        title: optional overall title
        figsize: figure size
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    pnorm = aggregated.get("pnorm", {})
    polis = aggregated["polis"]
    obs_rates = [1 - r for r in mask_rates]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Top-1 Frequency
    ax = axes[0]
    bridging_top1 = [bridging[r].get("stability_top_1_frequency", np.nan) for r in mask_rates]
    polis_top1 = [polis[r].get("stability_top_1_frequency", np.nan) for r in mask_rates]

    ax.plot(obs_rates, bridging_top1, "b-o", label="PD Bridging", linewidth=2)

    # Plot p-norm if available
    if pnorm:
        pnorm_top1 = [pnorm[r].get("stability_top_1_frequency", np.nan) for r in mask_rates]
        # Only skip if all values are NaN (use np.all for proper array handling)
        pnorm_top1_arr = np.array(pnorm_top1)
        if not np.all(np.isnan(pnorm_top1_arr)):
            ax.plot(obs_rates, pnorm_top1, "g-o", label="p-norm (p=-10)", linewidth=2)

    ax.plot(obs_rates, polis_top1, "r-o", label="Polis Consensus", linewidth=2)

    ax.set_xlabel("Observation Rate", fontsize=11)
    ax.set_ylabel("Top-1 Frequency", fontsize=11)
    ax.set_title("Top-1 Stability (higher = more stable)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)

    # Plot 2: Rank Correlation Between Trials
    ax = axes[1]
    bridging_corr = [bridging[r].get("stability_rank_correlation_mean", np.nan) for r in mask_rates]
    polis_corr = [polis[r].get("stability_rank_correlation_mean", np.nan) for r in mask_rates]

    ax.plot(obs_rates, bridging_corr, "b-o", label="PD Bridging", linewidth=2)

    # Plot p-norm if available
    if pnorm:
        pnorm_corr = [pnorm[r].get("stability_rank_correlation_mean", np.nan) for r in mask_rates]
        # Only skip if all values are NaN (use np.all for proper array handling)
        pnorm_corr_arr = np.array(pnorm_corr)
        if not np.all(np.isnan(pnorm_corr_arr)):
            ax.plot(obs_rates, pnorm_corr, "g-o", label="p-norm (p=-10)", linewidth=2)

    ax.plot(obs_rates, polis_corr, "r-o", label="Polis Consensus", linewidth=2)

    ax.set_xlabel("Observation Rate", fontsize=11)
    ax.set_ylabel("Rank Correlation", fontsize=11)
    ax.set_title("Inter-Trial Rank Correlation (higher = more stable)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_simulation_ranking_stability(
    aggregated: Dict,
    output_path: Path,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> None:
    """
    Plot ranking stability metrics vs observation rate (simulation experiment).

    Creates side-by-side plots of top-1 frequency and rank correlation for all 3 methods.

    Args:
        aggregated: output from aggregate_simulation_results
        output_path: path to save the plot
        title: optional overall title
        figsize: figure size
    """
    distribution_names = aggregated["distribution_names"]
    bridging_naive = aggregated["bridging_naive"]
    bridging_ipw = aggregated["bridging_ipw"]
    polis = aggregated["polis"]

    # Get observation rates and sort
    obs_rates = [bridging_naive[d]["observation_rate_mean"] for d in distribution_names]
    sorted_indices = np.argsort(obs_rates)[::-1]
    obs_rates_sorted = [obs_rates[i] for i in sorted_indices]
    dist_sorted = [distribution_names[i] for i in sorted_indices]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Top-1 Frequency
    ax = axes[0]
    naive_top1 = [bridging_naive[d].get("stability_top_1_frequency", np.nan) for d in dist_sorted]
    ipw_top1 = [bridging_ipw[d].get("stability_top_1_frequency", np.nan) for d in dist_sorted]
    polis_top1 = [polis[d].get("stability_top_1_frequency", np.nan) for d in dist_sorted]

    ax.plot(obs_rates_sorted, naive_top1, "b-o", label="Bridging (Naive)", linewidth=2)
    ax.plot(obs_rates_sorted, ipw_top1, "g-o", label="Bridging (IPW)", linewidth=2)
    ax.plot(obs_rates_sorted, polis_top1, "r-o", label="Polis Consensus", linewidth=2)

    ax.set_xlabel("Observation Rate", fontsize=11)
    ax.set_ylabel("Top-1 Frequency", fontsize=11)
    ax.set_title("Top-1 Stability (higher = more stable)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)

    # Plot 2: Rank Correlation Between Trials
    ax = axes[1]
    naive_corr = [bridging_naive[d].get("stability_rank_correlation_mean", np.nan) for d in dist_sorted]
    ipw_corr = [bridging_ipw[d].get("stability_rank_correlation_mean", np.nan) for d in dist_sorted]
    polis_corr = [polis[d].get("stability_rank_correlation_mean", np.nan) for d in dist_sorted]

    ax.plot(obs_rates_sorted, naive_corr, "b-o", label="Bridging (Naive)", linewidth=2)
    ax.plot(obs_rates_sorted, ipw_corr, "g-o", label="Bridging (IPW)", linewidth=2)
    ax.plot(obs_rates_sorted, polis_corr, "r-o", label="Polis Consensus", linewidth=2)

    ax.set_xlabel("Observation Rate", fontsize=11)
    ax.set_ylabel("Rank Correlation", fontsize=11)
    ax.set_title("Inter-Trial Rank Correlation (higher = more stable)", fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_simulation_summary_table(
    aggregated: Dict,
    output_path: Path,
) -> str:
    """
    Create a markdown table summarizing simulation results (3 methods).

    Args:
        aggregated: output from aggregate_simulation_results
        output_path: path to save the table

    Returns:
        Markdown table string
    """
    distribution_names = aggregated["distribution_names"]
    bridging_naive = aggregated["bridging_naive"]
    bridging_ipw = aggregated["bridging_ipw"]
    polis = aggregated["polis"]

    # Sort by observation rate
    obs_rates = [(d, bridging_naive[d]["observation_rate_mean"]) for d in distribution_names]
    obs_rates.sort(key=lambda x: -x[1])

    lines = [
        "# Robustness Comparison: Simulated Polis Routing",
        "",
        "## Results by Observation Rate",
        "",
        "| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |",
        "|----------|----------------|--------------|----------------|------------|----------|------------|",
    ]

    for dist_name, obs_rate in obs_rates:
        naive_sp = bridging_naive[dist_name]["spearman_mean"]
        naive_sp_std = bridging_naive[dist_name]["spearman_std"]
        ipw_sp = bridging_ipw[dist_name]["spearman_mean"]
        ipw_sp_std = bridging_ipw[dist_name]["spearman_std"]
        polis_sp = polis[dist_name]["spearman_mean"]
        polis_sp_std = polis[dist_name]["spearman_std"]
        naive_rmse = bridging_naive[dist_name]["rmse_mean"]
        ipw_rmse = bridging_ipw[dist_name]["rmse_mean"]
        polis_rmse = polis[dist_name]["rmse_mean"]

        lines.append(
            f"| {obs_rate:.0%} | {naive_sp:.3f} ({naive_sp_std:.3f}) | "
            f"{ipw_sp:.3f} ({ipw_sp_std:.3f}) | {polis_sp:.3f} ({polis_sp_std:.3f}) | "
            f"{naive_rmse:.4f} | {ipw_rmse:.4f} | {polis_rmse:.4f} |"
        )

    lines.extend([
        "",
        "## Legend",
        "- **Obs Rate**: Actual observation rate achieved under simulated routing",
        "- **Naive**: Bridging score without IPW correction",
        "- **IPW**: Bridging score with Inverse Probability Weighting",
        "- **Polis**: Polis Group-Informed Consensus",
        "- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)",
        "- **RMSE**: Root mean squared error from ground truth",
        "",
    ])

    table_str = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str
