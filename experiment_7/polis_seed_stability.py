"""
Experiment 7: Polis Ranking Stability Across Random Seeds

Tests whether Polis consensus rankings change when only the random seed changes,
given the exact same masked data. This isolates the algorithmic instability
(k-means initialization) from data variation.

Uses the 00026 French election dataset under various MCAR settings.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import stats
from itertools import combinations

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_5.polis import polis_consensus_pipeline
from experiment_5.masking import apply_random_mask


def test_polis_seed_stability(
    matrix: np.ndarray,
    mask_rate: float,
    mask_seed: int,
    n_polis_seeds: int = 20,
) -> dict:
    """
    Test Polis ranking stability across different seeds on the SAME masked data.

    Args:
        matrix: (n_items, n_voters) fully observed matrix
        mask_rate: fraction of entries to mask
        mask_seed: seed for generating the mask (fixed)
        n_polis_seeds: number of different Polis seeds to test

    Returns:
        Dict with stability metrics
    """
    n_items, n_voters = matrix.shape

    # Apply mask ONCE with fixed seed
    masked_matrix, observed_mask = apply_random_mask(
        matrix,
        mask_rate,
        seed=mask_seed,
        min_observed_per_item=2,
        min_observed_per_voter=1,
    )

    actual_obs_rate = observed_mask.sum() / observed_mask.size

    # Run Polis with different seeds
    all_scores = []
    all_rankings = []
    all_k = []

    for polis_seed in range(n_polis_seeds):
        scores, metadata = polis_consensus_pipeline(
            masked_matrix,
            observed_mask,
            n_pca_components=2,
            max_k=5,
            seed=polis_seed,
        )
        all_scores.append(scores)
        all_rankings.append(stats.rankdata(-scores))  # Higher score = lower rank
        all_k.append(metadata["k_clusters"])

    all_scores = np.array(all_scores)  # (n_seeds, n_items)
    all_rankings = np.array(all_rankings)  # (n_seeds, n_items)

    # Compute stability metrics

    # 1. Pairwise rank correlation between runs
    rank_correlations = []
    for i, j in combinations(range(n_polis_seeds), 2):
        corr = stats.spearmanr(all_rankings[i], all_rankings[j])[0]
        rank_correlations.append(corr)

    # 2. Top-1 item: how often is it the same?
    top_items = np.argmax(all_scores, axis=1)
    top_item_counts = np.bincount(top_items, minlength=n_items)
    most_common_top = top_item_counts.max()
    top_1_agreement = most_common_top / n_polis_seeds

    # 3. Top-3 overlap across runs
    top_3_sets = [set(np.argsort(-scores)[:3]) for scores in all_scores]
    top_3_overlaps = []
    for i, j in combinations(range(n_polis_seeds), 2):
        overlap = len(top_3_sets[i] & top_3_sets[j]) / 3
        top_3_overlaps.append(overlap)

    # 4. Score variance per item
    score_variance = all_scores.var(axis=0)  # (n_items,)

    # 5. k-cluster variation
    k_values = np.array(all_k)

    return {
        "mask_rate": mask_rate,
        "actual_obs_rate": actual_obs_rate,
        "n_seeds": n_polis_seeds,
        "rank_correlation_mean": np.mean(rank_correlations),
        "rank_correlation_std": np.std(rank_correlations),
        "rank_correlation_min": np.min(rank_correlations),
        "top_1_agreement": top_1_agreement,
        "top_3_overlap_mean": np.mean(top_3_overlaps),
        "top_3_overlap_min": np.min(top_3_overlaps),
        "score_variance_mean": score_variance.mean(),
        "score_variance_max": score_variance.max(),
        "k_mean": k_values.mean(),
        "k_std": k_values.std(),
        "k_unique": len(np.unique(k_values)),
    }


def run_experiment(
    data_path: Path,
    mask_rates: list,
    n_polis_seeds: int = 20,
    mask_seed: int = 42,
) -> list:
    """
    Run stability experiment across mask rates.

    Args:
        data_path: path to .npz data file
        mask_rates: list of mask rates to test
        n_polis_seeds: number of Polis seeds per mask rate
        mask_seed: fixed seed for masking

    Returns:
        List of result dicts
    """
    # Load data
    data = np.load(data_path)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape

    print(f"Dataset: {data_path.stem}")
    print(f"Matrix size: {n_items} items × {n_voters} voters")
    print(f"Testing {len(mask_rates)} mask rates with {n_polis_seeds} Polis seeds each")
    print()

    results = []

    for mask_rate in mask_rates:
        obs_rate = 1 - mask_rate
        print(f"Observation rate {obs_rate:.0%}...", end=" ", flush=True)

        result = test_polis_seed_stability(
            matrix,
            mask_rate,
            mask_seed=mask_seed,
            n_polis_seeds=n_polis_seeds,
        )

        print(f"rank_corr={result['rank_correlation_mean']:.3f}, "
              f"top1_agree={result['top_1_agreement']:.0%}, "
              f"k_unique={result['k_unique']}")

        results.append(result)

    return results


def print_summary(results: list):
    """Print summary table."""
    print()
    print("=" * 80)
    print("POLIS SEED STABILITY SUMMARY")
    print("=" * 80)
    print()
    print("Same masked data, different Polis random seeds.")
    print("If Polis were deterministic, all metrics would be 1.0")
    print()
    print(f"{'Obs Rate':>10} | {'Rank Corr':>12} | {'Top-1 Agree':>12} | "
          f"{'Top-3 Overlap':>13} | {'k unique':>8}")
    print("-" * 80)

    for r in results:
        obs = r["actual_obs_rate"]
        rc = r["rank_correlation_mean"]
        rc_std = r["rank_correlation_std"]
        t1 = r["top_1_agreement"]
        t3 = r["top_3_overlap_mean"]
        k_u = r["k_unique"]

        print(f"{obs:>9.0%} | {rc:>5.3f} ± {rc_std:.3f} | {t1:>11.0%} | "
              f"{t3:>12.0%} | {k_u:>8}")

    print()
    print("Interpretation:")
    print("  - Rank Corr < 1.0 means rankings change across seeds")
    print("  - Top-1 Agree < 100% means the 'best' item changes")
    print("  - k unique > 1 means cluster count varies")


if __name__ == "__main__":
    # Use 00026 French election dataset
    data_dir = Path("data/processed/preflib")
    data_files = sorted(data_dir.glob("00026-*.npz"))

    if not data_files:
        print(f"No data files found in {data_dir}")
        sys.exit(1)

    # Test various observation rates
    mask_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    all_results = {}

    for data_path in data_files:
        print(f"\n{'='*80}")
        print(f"DATASET: {data_path.stem}")
        print(f"{'='*80}\n")

        results = run_experiment(
            data_path,
            mask_rates,
            n_polis_seeds=20,
            mask_seed=42,
        )

        print_summary(results)
        all_results[data_path.stem] = results

    # Print aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY ACROSS ALL DATASETS")
    print("=" * 80)
    print()

    aggregate_results = []
    for mask_rate in mask_rates:
        obs_rate = 1 - mask_rate
        rank_corrs = []
        top1_agrees = []
        k_uniques = []

        for name, results in all_results.items():
            for r in results:
                if abs(r["mask_rate"] - mask_rate) < 0.01:
                    rank_corrs.append(r["rank_correlation_mean"])
                    top1_agrees.append(r["top_1_agreement"])
                    k_uniques.append(r["k_unique"])
                    break

        if rank_corrs:
            print(f"Obs {obs_rate:.0%}: rank_corr={np.mean(rank_corrs):.3f}±{np.std(rank_corrs):.3f}, "
                  f"top1={np.mean(top1_agrees):.0%}, k_unique={np.mean(k_uniques):.1f}")
            aggregate_results.append({
                "obs_rate": obs_rate,
                "rank_corr_mean": np.mean(rank_corrs),
                "rank_corr_std": np.std(rank_corrs),
                "top1_agree_mean": np.mean(top1_agrees),
                "k_unique_mean": np.mean(k_uniques),
            })

    # Save results
    import json
    results_dir = Path("experiment_7/results")
    results_dir.mkdir(exist_ok=True)

    output = {
        "description": "Polis ranking stability across random seeds (same masked data)",
        "n_polis_seeds": 20,
        "mask_seed": 42,
        "datasets": {name: [dict(r) for r in results] for name, results in all_results.items()},
        "aggregate": aggregate_results,
    }

    with open(results_dir / "polis_seed_stability.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_dir / 'polis_seed_stability.json'}")
