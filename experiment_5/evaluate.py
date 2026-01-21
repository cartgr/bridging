"""
Robustness evaluation metrics.

Computes metrics comparing estimated scores to ground truth across trials.
"""

from typing import Dict, List

import numpy as np
from scipy import stats


def compute_robustness_metrics(
    ground_truth: np.ndarray,
    estimates_list: List[np.ndarray],
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Compute robustness metrics for a list of estimates.

    Args:
        ground_truth: (n_items,) ground truth scores
        estimates_list: list of (n_items,) estimated score arrays
        k_values: k values for top-k metrics

    Returns:
        Dict containing:
        - spearman_mean, spearman_std: Spearman correlation stats
        - kendall_mean, kendall_std: Kendall tau stats
        - pearson_mean, pearson_std: Pearson correlation stats
        - rmse_mean, rmse_std: RMSE stats
        - mae_mean, mae_std: MAE stats
        - estimate_variance: variance of estimates across trials (per item)
        - mean_estimate_variance: mean variance across items
        - top_k_precision_{k}_mean/std: precision at k
    """
    n_trials = len(estimates_list)
    n_items = len(ground_truth)

    if n_trials == 0:
        return _empty_metrics(k_values)

    # Stack estimates for variance computation
    estimates_array = np.array(estimates_list)  # (n_trials, n_items)

    # Compute per-trial metrics
    spearman_vals = []
    kendall_vals = []
    pearson_vals = []
    rmse_vals = []
    mae_vals = []
    top_k_precision = {k: [] for k in k_values}

    for est in estimates_list:
        # Handle NaN values
        valid = ~np.isnan(est) & ~np.isnan(ground_truth)
        if valid.sum() < 2:
            continue

        gt_valid = ground_truth[valid]
        est_valid = est[valid]

        # Correlations
        spearman, _ = stats.spearmanr(gt_valid, est_valid)
        kendall, _ = stats.kendalltau(gt_valid, est_valid)
        pearson, _ = stats.pearsonr(gt_valid, est_valid)

        spearman_vals.append(spearman)
        kendall_vals.append(kendall)
        pearson_vals.append(pearson)

        # Error metrics
        rmse = np.sqrt(np.mean((gt_valid - est_valid) ** 2))
        mae = np.mean(np.abs(gt_valid - est_valid))
        rmse_vals.append(rmse)
        mae_vals.append(mae)

        # Top-k precision
        gt_ranking = np.argsort(ground_truth)[::-1]
        est_for_rank = np.where(np.isnan(est), -np.inf, est)
        est_ranking = np.argsort(est_for_rank)[::-1]

        for k in k_values:
            k_actual = min(k, n_items)
            gt_top_k = set(gt_ranking[:k_actual])
            est_top_k = set(est_ranking[:k_actual])
            precision = len(gt_top_k & est_top_k) / k_actual if k_actual > 0 else 0.0
            top_k_precision[k].append(precision)

    # Compute estimate variance across trials
    estimate_variance = np.nanvar(estimates_array, axis=0)  # (n_items,)
    mean_estimate_variance = np.nanmean(estimate_variance)

    # Aggregate metrics
    result = {
        "n_trials": n_trials,
        "spearman_mean": _safe_mean(spearman_vals),
        "spearman_std": _safe_std(spearman_vals),
        "kendall_mean": _safe_mean(kendall_vals),
        "kendall_std": _safe_std(kendall_vals),
        "pearson_mean": _safe_mean(pearson_vals),
        "pearson_std": _safe_std(pearson_vals),
        "rmse_mean": _safe_mean(rmse_vals),
        "rmse_std": _safe_std(rmse_vals),
        "mae_mean": _safe_mean(mae_vals),
        "mae_std": _safe_std(mae_vals),
        "estimate_variance_per_item": estimate_variance.tolist(),
        "mean_estimate_variance": mean_estimate_variance,
    }

    for k in k_values:
        result[f"top_{k}_precision_mean"] = _safe_mean(top_k_precision[k])
        result[f"top_{k}_precision_std"] = _safe_std(top_k_precision[k])

    return result


def _safe_mean(values: List[float]) -> float:
    """Compute mean, returning NaN for empty lists."""
    if not values:
        return np.nan
    return np.nanmean(values)


def _safe_std(values: List[float]) -> float:
    """Compute std, returning NaN for empty lists."""
    if not values:
        return np.nan
    return np.nanstd(values)


def _empty_metrics(k_values: List[int]) -> Dict:
    """Return empty metrics dict."""
    result = {
        "n_trials": 0,
        "spearman_mean": np.nan,
        "spearman_std": np.nan,
        "kendall_mean": np.nan,
        "kendall_std": np.nan,
        "pearson_mean": np.nan,
        "pearson_std": np.nan,
        "rmse_mean": np.nan,
        "rmse_std": np.nan,
        "mae_mean": np.nan,
        "mae_std": np.nan,
        "estimate_variance_per_item": [],
        "mean_estimate_variance": np.nan,
    }
    for k in k_values:
        result[f"top_{k}_precision_mean"] = np.nan
        result[f"top_{k}_precision_std"] = np.nan
    return result


def aggregate_by_mask_rate(
    experiment_results: Dict,
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Aggregate metrics by mask rate for both methods.

    Args:
        experiment_results: output from run_masking_experiment
        k_values: k values for top-k metrics

    Returns:
        Dict with aggregated metrics for bridging and polis at each mask rate
    """
    gt_bridging = experiment_results["gt_bridging"]
    gt_polis = experiment_results["gt_polis"]
    results = experiment_results["results"]
    mask_rates = experiment_results["mask_rates"]

    aggregated = {
        "mask_rates": mask_rates,
        "bridging": {},
        "polis": {},
    }

    for mask_rate in mask_rates:
        trials = results[mask_rate]

        # Extract estimates
        bridging_estimates = [t["bridging_scores"] for t in trials]
        polis_estimates = [t["polis_scores"] for t in trials]

        # Compute metrics
        bridging_metrics = compute_robustness_metrics(
            gt_bridging, bridging_estimates, k_values
        )
        polis_metrics = compute_robustness_metrics(
            gt_polis, polis_estimates, k_values
        )

        # Add actual mask rate info
        actual_rates = [t["actual_mask_rate"] for t in trials]
        bridging_metrics["actual_mask_rate_mean"] = np.mean(actual_rates)
        polis_metrics["actual_mask_rate_mean"] = np.mean(actual_rates)

        # Add polis k stats
        polis_ks = [t["polis_k"] for t in trials]
        polis_metrics["k_mean"] = np.mean(polis_ks)
        polis_metrics["k_std"] = np.std(polis_ks)
        polis_metrics["k_values"] = polis_ks

        aggregated["bridging"][mask_rate] = bridging_metrics
        aggregated["polis"][mask_rate] = polis_metrics

    return aggregated


def compute_summary_statistics(aggregated: Dict) -> Dict:
    """
    Compute summary statistics comparing methods.

    Args:
        aggregated: output from aggregate_by_mask_rate

    Returns:
        Dict with summary comparisons
    """
    mask_rates = aggregated["mask_rates"]
    bridging = aggregated["bridging"]
    polis = aggregated["polis"]

    summary = {
        "mask_rates": mask_rates,
        "bridging_wins_spearman": 0,
        "polis_wins_spearman": 0,
        "ties_spearman": 0,
        "bridging_wins_rmse": 0,
        "polis_wins_rmse": 0,
        "ties_rmse": 0,
    }

    for rate in mask_rates:
        b_spearman = bridging[rate]["spearman_mean"]
        p_spearman = polis[rate]["spearman_mean"]
        b_rmse = bridging[rate]["rmse_mean"]
        p_rmse = polis[rate]["rmse_mean"]

        # Spearman (higher is better)
        if b_spearman > p_spearman + 0.01:
            summary["bridging_wins_spearman"] += 1
        elif p_spearman > b_spearman + 0.01:
            summary["polis_wins_spearman"] += 1
        else:
            summary["ties_spearman"] += 1

        # RMSE (lower is better)
        if b_rmse < p_rmse - 0.001:
            summary["bridging_wins_rmse"] += 1
        elif p_rmse < b_rmse - 0.001:
            summary["polis_wins_rmse"] += 1
        else:
            summary["ties_rmse"] += 1

    # Average metrics at different sparsity levels
    low_mask = [r for r in mask_rates if r <= 0.3]
    mid_mask = [r for r in mask_rates if 0.3 < r <= 0.5]
    high_mask = [r for r in mask_rates if r > 0.5]

    for label, rates in [("low", low_mask), ("mid", mid_mask), ("high", high_mask)]:
        if rates:
            summary[f"bridging_spearman_{label}"] = np.mean(
                [bridging[r]["spearman_mean"] for r in rates]
            )
            summary[f"polis_spearman_{label}"] = np.mean(
                [polis[r]["spearman_mean"] for r in rates]
            )
            summary[f"bridging_variance_{label}"] = np.mean(
                [bridging[r]["mean_estimate_variance"] for r in rates]
            )
            summary[f"polis_variance_{label}"] = np.mean(
                [polis[r]["mean_estimate_variance"] for r in rates]
            )

    return summary


def aggregate_simulation_results(
    experiment_results: Dict,
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Aggregate metrics by observation rate for simulation experiments (3 methods).

    Args:
        experiment_results: output from run_simulation_experiment
        k_values: k values for top-k metrics

    Returns:
        Dict with aggregated metrics for bridging_naive, bridging_ipw, and polis
    """
    gt_bridging = experiment_results["gt_bridging"]
    gt_polis = experiment_results["gt_polis"]
    results = experiment_results["results"]
    distribution_names = experiment_results["distribution_names"]

    aggregated = {
        "distribution_names": distribution_names,
        "bridging_naive": {},
        "bridging_ipw": {},
        "polis": {},
    }

    for dist_name in distribution_names:
        trials = results[dist_name]

        # Extract estimates
        bridging_naive_estimates = [t["bridging_naive"] for t in trials]
        bridging_ipw_estimates = [t["bridging_ipw"] for t in trials]
        polis_estimates = [t["polis_scores"] for t in trials]

        # Compute metrics
        naive_metrics = compute_robustness_metrics(
            gt_bridging, bridging_naive_estimates, k_values
        )
        ipw_metrics = compute_robustness_metrics(
            gt_bridging, bridging_ipw_estimates, k_values
        )
        polis_metrics = compute_robustness_metrics(
            gt_polis, polis_estimates, k_values
        )

        # Add observation rate info
        obs_rates = [t["observation_rate"] for t in trials]
        naive_metrics["observation_rate_mean"] = np.mean(obs_rates)
        naive_metrics["observation_rate_std"] = np.std(obs_rates)
        ipw_metrics["observation_rate_mean"] = np.mean(obs_rates)
        ipw_metrics["observation_rate_std"] = np.std(obs_rates)
        polis_metrics["observation_rate_mean"] = np.mean(obs_rates)
        polis_metrics["observation_rate_std"] = np.std(obs_rates)

        # Add polis k stats
        polis_ks = [t["polis_k"] for t in trials]
        polis_metrics["k_mean"] = np.mean(polis_ks)
        polis_metrics["k_std"] = np.std(polis_ks)

        aggregated["bridging_naive"][dist_name] = naive_metrics
        aggregated["bridging_ipw"][dist_name] = ipw_metrics
        aggregated["polis"][dist_name] = polis_metrics

    return aggregated


def compute_simulation_summary(aggregated: Dict) -> Dict:
    """
    Compute summary statistics for simulation experiment (3 methods).

    Args:
        aggregated: output from aggregate_simulation_results

    Returns:
        Dict with summary comparisons
    """
    distribution_names = aggregated["distribution_names"]
    bridging_naive = aggregated["bridging_naive"]
    bridging_ipw = aggregated["bridging_ipw"]
    polis = aggregated["polis"]

    summary = {
        "distribution_names": distribution_names,
        # Naive vs Polis
        "naive_wins_vs_polis": 0,
        "polis_wins_vs_naive": 0,
        "ties_naive_polis": 0,
        # IPW vs Polis
        "ipw_wins_vs_polis": 0,
        "polis_wins_vs_ipw": 0,
        "ties_ipw_polis": 0,
        # IPW vs Naive
        "ipw_wins_vs_naive": 0,
        "naive_wins_vs_ipw": 0,
        "ties_ipw_naive": 0,
    }

    for dist in distribution_names:
        naive_spearman = bridging_naive[dist]["spearman_mean"]
        ipw_spearman = bridging_ipw[dist]["spearman_mean"]
        polis_spearman = polis[dist]["spearman_mean"]

        # Naive vs Polis
        if naive_spearman > polis_spearman + 0.01:
            summary["naive_wins_vs_polis"] += 1
        elif polis_spearman > naive_spearman + 0.01:
            summary["polis_wins_vs_naive"] += 1
        else:
            summary["ties_naive_polis"] += 1

        # IPW vs Polis
        if ipw_spearman > polis_spearman + 0.01:
            summary["ipw_wins_vs_polis"] += 1
        elif polis_spearman > ipw_spearman + 0.01:
            summary["polis_wins_vs_ipw"] += 1
        else:
            summary["ties_ipw_polis"] += 1

        # IPW vs Naive
        if ipw_spearman > naive_spearman + 0.01:
            summary["ipw_wins_vs_naive"] += 1
        elif naive_spearman > ipw_spearman + 0.01:
            summary["naive_wins_vs_ipw"] += 1
        else:
            summary["ties_ipw_naive"] += 1

    # Average metrics
    summary["naive_spearman_avg"] = np.mean(
        [bridging_naive[d]["spearman_mean"] for d in distribution_names]
    )
    summary["ipw_spearman_avg"] = np.mean(
        [bridging_ipw[d]["spearman_mean"] for d in distribution_names]
    )
    summary["polis_spearman_avg"] = np.mean(
        [polis[d]["spearman_mean"] for d in distribution_names]
    )

    summary["naive_variance_avg"] = np.mean(
        [bridging_naive[d]["mean_estimate_variance"] for d in distribution_names]
    )
    summary["ipw_variance_avg"] = np.mean(
        [bridging_ipw[d]["mean_estimate_variance"] for d in distribution_names]
    )
    summary["polis_variance_avg"] = np.mean(
        [polis[d]["mean_estimate_variance"] for d in distribution_names]
    )

    return summary
