"""
Main experiment runner for Experiment 2: Bridging Score Estimation Under Pol.is Sampling.

This script runs the full experiment pipeline:
1. Load ground truth data
2. Extract empirical vote distribution
3. Simulate Pol.is routing
4. Estimate bridging scores using IPW
5. Evaluate and compare to ground truth
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
from tqdm import tqdm

from .bridging import compute_bridging_scores_vectorized
from .simulation import get_empirical_votes_distribution, simulate_polis_routing
from .estimation import estimate_bridging_scores_ipw, estimate_bridging_scores_naive
from .evaluate import (
    evaluate_estimation,
    compute_observation_statistics,
    format_results,
)


def load_data(
    completed_dir: Path, processed_dir: Path
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load ground truth matrices and per-dataset vote distributions.

    Args:
        completed_dir: path to completed data directory
        processed_dir: path to processed data directory

    Returns:
        Tuple of:
        - List of ground truth matrices
        - List of per-dataset vote distributions
        - List of dataset names
    """
    # Find all completed files
    completed_files = sorted(completed_dir.glob("00069-*.npz"))

    if not completed_files:
        raise ValueError(f"No completed files found in {completed_dir}")

    # Load ground truth matrices and match with processed files for vote distributions
    ground_truth_matrices = []
    votes_distributions = []
    dataset_names = []

    for file_path in completed_files:
        # Load ground truth
        data = np.load(file_path)
        ground_truth_matrices.append(data["matrix"])
        dataset_names.append(file_path.stem)

        # Find matching processed file for vote distribution
        processed_path = processed_dir / file_path.name
        if processed_path.exists():
            processed_data = np.load(processed_path)
            processed_matrix = processed_data["matrix"]
            observed = ~np.isnan(processed_matrix)
            votes_per_voter = observed.sum(axis=0)
            votes_distributions.append(votes_per_voter)
        else:
            raise ValueError(f"No matching processed file found: {processed_path}")

    print(f"Loaded {len(ground_truth_matrices)} ground truth datasets with per-dataset vote distributions")

    return ground_truth_matrices, votes_distributions, dataset_names


def run_single_dataset(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    dataset_name: str,
    seed: int = 42,
    k_values: List[int] = [1, 3, 5],
    show_progress: bool = False,
) -> Dict:
    """
    Run experiment on a single dataset.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: empirical vote count distribution
        dataset_name: name of the dataset
        seed: random seed
        k_values: list of k values for top-k metrics
        show_progress: whether to show progress bar for voters

    Returns:
        Dictionary with results
    """
    n_items, n_voters = ground_truth.shape

    # Step 1: Compute ground truth bridging scores
    true_bridging_scores = compute_bridging_scores_vectorized(ground_truth)

    # Step 2: Simulate Pol.is routing
    observed_mask, inclusion_probs = simulate_polis_routing(
        ground_truth=ground_truth,
        votes_distribution=votes_distribution,
        seed=seed,
        show_progress=show_progress,
    )

    # Create observed matrix (NaN for unobserved)
    observed_matrix = np.where(observed_mask, ground_truth, np.nan)

    # Step 3: Compute observation statistics
    obs_stats = compute_observation_statistics(observed_mask)

    # Step 4: Estimate bridging scores using IPW
    ipw_bridging_scores = estimate_bridging_scores_ipw(
        observed_matrix=observed_matrix,
        observed_mask=observed_mask,
        inclusion_probs=inclusion_probs,
    )

    # Step 5: Estimate bridging scores using naive method (baseline)
    naive_bridging_scores = estimate_bridging_scores_naive(
        observed_matrix=observed_matrix,
        observed_mask=observed_mask,
    )

    # Step 6: Evaluate both estimators
    ipw_metrics = evaluate_estimation(true_bridging_scores, ipw_bridging_scores, k_values)
    naive_metrics = evaluate_estimation(true_bridging_scores, naive_bridging_scores, k_values)

    return {
        "dataset_name": dataset_name,
        "n_items": n_items,
        "n_voters": n_voters,
        "observation_stats": obs_stats,
        "ipw_metrics": ipw_metrics,
        "naive_metrics": naive_metrics,
        "true_bridging_scores": true_bridging_scores.tolist(),
        "ipw_bridging_scores": ipw_bridging_scores.tolist(),
        "naive_bridging_scores": naive_bridging_scores.tolist(),
    }


def run_experiment(
    completed_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    seed: int = 42,
    k_values: List[int] = [1, 3, 5],
    verbose: bool = True,
) -> Dict:
    """
    Run the full experiment on all datasets.

    Args:
        completed_dir: path to completed data directory
        processed_dir: path to processed data directory
        output_dir: path to output directory
        seed: random seed
        k_values: list of k values for top-k metrics
        verbose: whether to print progress

    Returns:
        Dictionary with aggregate results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ground_truth_matrices, votes_distributions, dataset_names = load_data(
        completed_dir, processed_dir
    )

    # Run experiment on each dataset
    all_results = []

    dataset_list = list(zip(ground_truth_matrices, votes_distributions, dataset_names))

    if verbose:
        dataset_iter = tqdm(
            dataset_list,
            desc="Datasets",
            unit="dataset",
            position=0,
        )
    else:
        dataset_iter = dataset_list

    for i, (gt_matrix, votes_dist, name) in enumerate(dataset_iter):
        n_items, n_voters = gt_matrix.shape

        if verbose and hasattr(dataset_iter, 'set_description'):
            dataset_iter.set_description(f"Dataset {name} ({n_items}×{n_voters})")

        result = run_single_dataset(
            ground_truth=gt_matrix,
            votes_distribution=votes_dist,
            dataset_name=name,
            seed=seed + i,  # Different seed for each dataset
            k_values=k_values,
            show_progress=verbose,  # Show voter progress bar when verbose
        )

        all_results.append(result)

        if verbose and hasattr(dataset_iter, 'set_postfix'):
            dataset_iter.set_postfix({
                'obs': f"{result['observation_stats']['observation_rate']:.0%}",
                'ρ': f"{result['ipw_metrics']['spearman_correlation']:.2f}",
            })

    # Compute aggregate statistics
    aggregate = compute_aggregate_statistics(all_results, k_values)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save per-dataset results (without full score arrays for compactness)
    per_dataset_results = []
    for r in all_results:
        r_compact = {k: v for k, v in r.items()
                     if k not in ["true_bridging_scores", "ipw_bridging_scores", "naive_bridging_scores"]}
        per_dataset_results.append(r_compact)

    results_path = output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "seed": seed,
            "k_values": k_values,
            "n_datasets": len(all_results),
            "aggregate": aggregate,
            "per_dataset": per_dataset_results,
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # Save full results with scores (for analysis)
    full_results_path = output_dir / f"full_results_{timestamp}.json"
    with open(full_results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "seed": seed,
            "all_results": all_results,
        }, f, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    if verbose:
        print(f"\nResults saved to {results_path}")
        print(f"Full results saved to {full_results_path}")
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        print(format_results(aggregate))

    return {
        "aggregate": aggregate,
        "per_dataset": all_results,
        "results_path": str(results_path),
    }


def compute_aggregate_statistics(
    all_results: List[Dict],
    k_values: List[int],
) -> Dict:
    """
    Compute aggregate statistics across all datasets.

    Args:
        all_results: list of per-dataset results
        k_values: list of k values for top-k metrics

    Returns:
        Dictionary with aggregate statistics
    """
    n = len(all_results)

    # Extract metric arrays
    ipw_spearman = [r["ipw_metrics"]["spearman_correlation"] for r in all_results]
    ipw_kendall = [r["ipw_metrics"]["kendall_correlation"] for r in all_results]
    ipw_rmse = [r["ipw_metrics"]["rmse"] for r in all_results]

    naive_spearman = [r["naive_metrics"]["spearman_correlation"] for r in all_results]
    naive_kendall = [r["naive_metrics"]["kendall_correlation"] for r in all_results]
    naive_rmse = [r["naive_metrics"]["rmse"] for r in all_results]

    obs_rates = [r["observation_stats"]["observation_rate"] for r in all_results]

    aggregate = {
        "n_datasets": n,
        "observation_rate": {
            "mean": np.mean(obs_rates),
            "std": np.std(obs_rates),
            "min": np.min(obs_rates),
            "max": np.max(obs_rates),
        },
        "ipw": {
            "spearman_mean": np.nanmean(ipw_spearman),
            "spearman_std": np.nanstd(ipw_spearman),
            "kendall_mean": np.nanmean(ipw_kendall),
            "kendall_std": np.nanstd(ipw_kendall),
            "rmse_mean": np.nanmean(ipw_rmse),
            "rmse_std": np.nanstd(ipw_rmse),
        },
        "naive": {
            "spearman_mean": np.nanmean(naive_spearman),
            "spearman_std": np.nanstd(naive_spearman),
            "kendall_mean": np.nanmean(naive_kendall),
            "kendall_std": np.nanstd(naive_kendall),
            "rmse_mean": np.nanmean(naive_rmse),
            "rmse_std": np.nanstd(naive_rmse),
        },
    }

    # Top-k metrics
    for k in k_values:
        ipw_prec = [r["ipw_metrics"]["top_k_precision"][k] for r in all_results]
        ipw_rec = [r["ipw_metrics"]["top_k_recall"][k] for r in all_results]
        naive_prec = [r["naive_metrics"]["top_k_precision"][k] for r in all_results]
        naive_rec = [r["naive_metrics"]["top_k_recall"][k] for r in all_results]

        aggregate["ipw"][f"top_{k}_precision_mean"] = np.nanmean(ipw_prec)
        aggregate["ipw"][f"top_{k}_precision_std"] = np.nanstd(ipw_prec)
        aggregate["ipw"][f"top_{k}_recall_mean"] = np.nanmean(ipw_rec)
        aggregate["ipw"][f"top_{k}_recall_std"] = np.nanstd(ipw_rec)

        aggregate["naive"][f"top_{k}_precision_mean"] = np.nanmean(naive_prec)
        aggregate["naive"][f"top_{k}_precision_std"] = np.nanstd(naive_prec)
        aggregate["naive"][f"top_{k}_recall_mean"] = np.nanmean(naive_rec)
        aggregate["naive"][f"top_{k}_recall_std"] = np.nanstd(naive_rec)

    return aggregate


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Bridging Score Estimation Under Pol.is Sampling"
    )
    parser.add_argument(
        "--completed-dir",
        type=Path,
        default=Path("data/completed"),
        help="Directory containing completed (ground truth) data files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/preflib"),
        help="Directory containing processed data files (for vote distribution)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiment_2/results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    run_experiment(
        completed_dir=args.completed_dir,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
