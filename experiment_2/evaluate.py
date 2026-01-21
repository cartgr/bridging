"""
Evaluation metrics for bridging score estimation.

Compares estimated bridging scores to ground truth using various metrics.
"""

import numpy as np
from scipy import stats
from typing import Dict, List


def evaluate_estimation(
    true_scores: np.ndarray,
    estimated_scores: np.ndarray,
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Compute evaluation metrics comparing estimated to true bridging scores.

    Args:
        true_scores: (n_items,) array of ground truth bridging scores
        estimated_scores: (n_items,) array of estimated bridging scores
        k_values: list of k values for top-k metrics

    Returns:
        Dictionary containing:
        - spearman_correlation: Spearman's rank correlation
        - kendall_correlation: Kendall's tau correlation
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - top_k_precision: dict of precision@k for each k
        - top_k_recall: dict of recall@k for each k
        - n_valid: number of valid (non-NaN) estimates
    """
    # Handle NaN values
    valid_mask = ~np.isnan(estimated_scores) & ~np.isnan(true_scores)
    n_valid = valid_mask.sum()
    n_total = len(true_scores)

    if n_valid == 0:
        return {
            "spearman_correlation": np.nan,
            "kendall_correlation": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "top_k_precision": {k: np.nan for k in k_values},
            "top_k_recall": {k: np.nan for k in k_values},
            "n_valid": 0,
            "n_total": n_total,
        }

    true_valid = true_scores[valid_mask]
    est_valid = estimated_scores[valid_mask]

    # Rank correlations
    spearman_corr, _ = stats.spearmanr(true_valid, est_valid)
    kendall_corr, _ = stats.kendalltau(true_valid, est_valid)

    # Error metrics
    rmse = np.sqrt(np.mean((true_valid - est_valid) ** 2))
    mae = np.mean(np.abs(true_valid - est_valid))

    # Top-k metrics
    # Use all items (including NaN estimates which won't be in top-k)
    top_k_precision = {}
    top_k_recall = {}

    # True top-k items (based on ground truth)
    true_ranking = np.argsort(true_scores)[::-1]  # Descending

    # Estimated top-k items (treating NaN as -inf)
    est_for_ranking = np.where(np.isnan(estimated_scores), -np.inf, estimated_scores)
    est_ranking = np.argsort(est_for_ranking)[::-1]

    for k in k_values:
        k_actual = min(k, n_total)
        true_top_k = set(true_ranking[:k_actual])
        est_top_k = set(est_ranking[:k_actual])

        # Precision: fraction of estimated top-k that are truly top-k
        precision = len(true_top_k & est_top_k) / k_actual if k_actual > 0 else 0.0

        # Recall: fraction of true top-k that are in estimated top-k
        recall = len(true_top_k & est_top_k) / k_actual if k_actual > 0 else 0.0

        top_k_precision[k] = precision
        top_k_recall[k] = recall

    return {
        "spearman_correlation": spearman_corr,
        "kendall_correlation": kendall_corr,
        "rmse": rmse,
        "mae": mae,
        "top_k_precision": top_k_precision,
        "top_k_recall": top_k_recall,
        "n_valid": n_valid,
        "n_total": n_total,
    }


def evaluate_estimation_monte_carlo(
    true_scores: np.ndarray,
    estimated_scores_list: List[np.ndarray],
    k_values: List[int] = [1, 3, 5],
) -> Dict:
    """
    Evaluate estimation over multiple Monte Carlo simulations.

    Args:
        true_scores: (n_items,) array of ground truth bridging scores
        estimated_scores_list: list of (n_items,) arrays from different simulations
        k_values: list of k values for top-k metrics

    Returns:
        Dictionary with mean and std of each metric across simulations
    """
    metrics_list = []

    for est_scores in estimated_scores_list:
        metrics = evaluate_estimation(true_scores, est_scores, k_values)
        metrics_list.append(metrics)

    # Aggregate metrics
    n_sims = len(metrics_list)

    aggregated = {
        "n_simulations": n_sims,
        "spearman_correlation_mean": np.nanmean(
            [m["spearman_correlation"] for m in metrics_list]
        ),
        "spearman_correlation_std": np.nanstd(
            [m["spearman_correlation"] for m in metrics_list]
        ),
        "kendall_correlation_mean": np.nanmean(
            [m["kendall_correlation"] for m in metrics_list]
        ),
        "kendall_correlation_std": np.nanstd(
            [m["kendall_correlation"] for m in metrics_list]
        ),
        "rmse_mean": np.nanmean([m["rmse"] for m in metrics_list]),
        "rmse_std": np.nanstd([m["rmse"] for m in metrics_list]),
        "mae_mean": np.nanmean([m["mae"] for m in metrics_list]),
        "mae_std": np.nanstd([m["mae"] for m in metrics_list]),
    }

    # Top-k metrics
    for k in k_values:
        prec_values = [m["top_k_precision"][k] for m in metrics_list]
        rec_values = [m["top_k_recall"][k] for m in metrics_list]

        aggregated[f"top_{k}_precision_mean"] = np.nanmean(prec_values)
        aggregated[f"top_{k}_precision_std"] = np.nanstd(prec_values)
        aggregated[f"top_{k}_recall_mean"] = np.nanmean(rec_values)
        aggregated[f"top_{k}_recall_std"] = np.nanstd(rec_values)

    return aggregated


def compute_observation_statistics(
    observed_mask: np.ndarray,
) -> Dict:
    """
    Compute statistics about the observed data.

    Args:
        observed_mask: (n_items, n_voters) boolean array

    Returns:
        Dictionary with observation statistics
    """
    n_items, n_voters = observed_mask.shape
    total_cells = n_items * n_voters
    observed_cells = observed_mask.sum()

    # Votes per comment
    votes_per_comment = observed_mask.sum(axis=1)

    # Votes per voter
    votes_per_voter = observed_mask.sum(axis=0)

    return {
        "n_items": n_items,
        "n_voters": n_voters,
        "total_cells": total_cells,
        "observed_cells": int(observed_cells),
        "observation_rate": observed_cells / total_cells,
        "votes_per_comment_mean": votes_per_comment.mean(),
        "votes_per_comment_std": votes_per_comment.std(),
        "votes_per_comment_min": int(votes_per_comment.min()),
        "votes_per_comment_max": int(votes_per_comment.max()),
        "votes_per_voter_mean": votes_per_voter.mean(),
        "votes_per_voter_std": votes_per_voter.std(),
        "votes_per_voter_min": int(votes_per_voter.min()),
        "votes_per_voter_max": int(votes_per_voter.max()),
    }


def format_results(results: Dict, indent: int = 0) -> str:
    """
    Format results dictionary as human-readable string.

    Args:
        results: dictionary of results
        indent: indentation level

    Returns:
        Formatted string
    """
    lines = []
    prefix = "  " * indent

    for key, value in results.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_results(value, indent + 1))
        elif isinstance(value, float):
            if np.isnan(value):
                lines.append(f"{prefix}{key}: NaN")
            else:
                lines.append(f"{prefix}{key}: {value:.4f}")
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)
