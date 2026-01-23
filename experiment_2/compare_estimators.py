"""
Compare different bridging score estimators on Pol.is data.

Tests: Naive, IPW, Truncated IPW, AIPW, Matrix Completion
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from tqdm import tqdm

from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_2.simulation import simulate_polis_routing
from experiment_2.estimation import (
    estimate_bridging_scores_naive,
    estimate_bridging_scores_ipw,
    estimate_bridging_scores_truncated_ipw,
    estimate_bridging_scores_aipw,
    estimate_bridging_scores_matrix_completion,
)


def evaluate(true_scores, estimated_scores):
    """Compute evaluation metrics."""
    valid = ~np.isnan(estimated_scores) & ~np.isnan(true_scores)
    if valid.sum() < 2:
        return {"spearman": np.nan, "rmse": np.nan, "mae": np.nan}

    spearman = stats.spearmanr(true_scores[valid], estimated_scores[valid])[0]
    rmse = np.sqrt(np.mean((true_scores[valid] - estimated_scores[valid])**2))
    mae = np.mean(np.abs(true_scores[valid] - estimated_scores[valid]))

    return {"spearman": spearman, "rmse": rmse, "mae": mae}


def run_comparison(completed_dir: Path, processed_dir: Path, n_datasets: int = 5):
    """Run comparison on multiple datasets."""

    # Find datasets - sort by file size to start with smaller ones
    completed_files = sorted(completed_dir.glob("00069-*.npz"))
    # Sort by size (smaller first for faster initial results)
    completed_files = sorted(completed_files, key=lambda p: p.stat().st_size)[:n_datasets]

    results = []

    for file_idx, file_path in enumerate(completed_files):
        name = file_path.stem

        # Load ground truth
        print(f"\n[{file_idx+1}/{n_datasets}] Loading {name}...")
        data = np.load(file_path)
        ground_truth = data["matrix"]
        n_items, n_voters = ground_truth.shape
        print(f"  Matrix size: {n_items} items × {n_voters} voters")

        # Load vote distribution
        processed_path = processed_dir / file_path.name
        processed_data = np.load(processed_path)
        processed_matrix = processed_data["matrix"]
        observed = ~np.isnan(processed_matrix)
        votes_distribution = observed.sum(axis=0)

        # Compute true bridging scores
        print(f"  Computing ground truth bridging scores...")
        true_scores = compute_bridging_scores_vectorized(ground_truth)

        # Simulate Pol.is routing
        print(f"  Simulating Pol.is routing...")
        observed_mask, inclusion_probs = simulate_polis_routing(
            ground_truth=ground_truth,
            votes_distribution=votes_distribution,
            seed=42,
            show_progress=True,
        )

        observed_matrix = np.where(observed_mask, ground_truth, np.nan)
        obs_rate = observed_mask.sum() / observed_mask.size
        print(f"  Observation rate: {obs_rate:.1%}")

        # Test all estimators
        estimators = [
            ("Naive", lambda: estimate_bridging_scores_naive(observed_matrix, observed_mask)),
            ("IPW", lambda: estimate_bridging_scores_ipw(observed_matrix, observed_mask, inclusion_probs)),
            ("Truncated IPW (w≤10)", lambda: estimate_bridging_scores_truncated_ipw(
                observed_matrix, observed_mask, inclusion_probs, max_weight=10.0
            )),
            ("Truncated IPW (w≤20)", lambda: estimate_bridging_scores_truncated_ipw(
                observed_matrix, observed_mask, inclusion_probs, max_weight=20.0
            )),
            ("AIPW", lambda: estimate_bridging_scores_aipw(
                observed_matrix, observed_mask, inclusion_probs, max_weight=50.0
            )),
            ("Matrix Completion", lambda: estimate_bridging_scores_matrix_completion(
                observed_matrix, observed_mask, n_factors=10, max_iter=50
            )),
        ]

        dataset_results = {
            "name": name,
            "n_items": n_items,
            "n_voters": n_voters,
            "obs_rate": obs_rate,
        }

        print(f"\n  Running estimators:")
        for est_name, est_fn in tqdm(estimators, desc="  Estimators", leave=False):
            try:
                estimated = est_fn()
                metrics = evaluate(true_scores, estimated)
                dataset_results[est_name] = metrics
                print(f"    {est_name:25s}: ρ={metrics['spearman']:.3f}, RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                dataset_results[est_name] = {"spearman": np.nan, "rmse": np.nan, "mae": np.nan, "error": str(e)}
                print(f"    {est_name:25s}: ERROR - {e}")

        results.append(dataset_results)

    return results


def print_summary(results):
    """Print summary table."""
    estimator_names = ["Naive", "IPW", "Truncated IPW (w≤10)", "Truncated IPW (w≤20)", "AIPW", "Matrix Completion"]

    print("\n" + "=" * 80)
    print("SUMMARY: Mean Spearman Correlation Across Datasets")
    print("=" * 80)

    for est_name in estimator_names:
        spearman_vals = [r[est_name]["spearman"] for r in results if est_name in r]
        rmse_vals = [r[est_name]["rmse"] for r in results if est_name in r]

        mean_sp = np.nanmean(spearman_vals)
        std_sp = np.nanstd(spearman_vals)
        mean_rmse = np.nanmean(rmse_vals)
        std_rmse = np.nanstd(rmse_vals)

        print(f"{est_name:25s}: Spearman = {mean_sp:.3f} ± {std_sp:.3f}, RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=1, help="Number of datasets to test")
    args = parser.parse_args()

    completed_dir = Path("data/completed")
    processed_dir = Path("data/processed/preflib")

    print("Comparing bridging score estimators...")
    print("=" * 80)

    results = run_comparison(completed_dir, processed_dir, n_datasets=args.n_datasets)
    print_summary(results)

    # Save results
    output_path = Path("experiment_2/results/estimator_comparison.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\nResults saved to {output_path}")
