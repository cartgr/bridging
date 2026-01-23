"""
Test whether joint inclusion probabilities satisfy independence assumption.

The IPW estimator assumes: P(i sees c AND j sees c) = π_i,c × π_j,c

This test verifies whether this holds for Pol.is adaptive routing.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

from experiment_2.simulation import simulate_polis_routing


def test_joint_probability_independence(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    n_simulations: int = 200,
    base_seed: int = 42,
) -> dict:
    """
    Test whether joint inclusion probabilities satisfy independence.

    Runs many simulations and compares:
    - Empirical: P(i sees c AND j sees c) estimated from simulations
    - Assumed: π_i,c × π_j,c (product of marginals)

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: array of vote counts
        n_simulations: number of Monte Carlo simulations
        base_seed: random seed

    Returns:
        Dictionary with comparison results
    """
    n_items, n_voters = ground_truth.shape

    # Track counts
    # marginal_counts[c, v] = number of times voter v saw comment c
    marginal_counts = np.zeros((n_items, n_voters))

    # joint_counts[c, v1, v2] = number of times BOTH v1 and v2 saw comment c
    # Only track for a subset of voter pairs to save memory
    n_pairs_to_track = min(50, n_voters * (n_voters - 1) // 2)

    # Select random voter pairs
    rng = np.random.default_rng(base_seed)
    all_pairs = [(i, j) for i in range(n_voters) for j in range(i+1, n_voters)]
    if len(all_pairs) > n_pairs_to_track:
        pair_indices = rng.choice(len(all_pairs), n_pairs_to_track, replace=False)
        tracked_pairs = [all_pairs[i] for i in pair_indices]
    else:
        tracked_pairs = all_pairs

    # joint_counts[c, pair_idx] = count
    joint_counts = np.zeros((n_items, len(tracked_pairs)))

    # Also store the computed marginal probabilities from each simulation
    # (they should be similar across simulations for early voters, but differ for later ones)
    all_marginal_probs = []

    print(f"Running {n_simulations} simulations...")
    print(f"Tracking {len(tracked_pairs)} voter pairs")

    for sim in tqdm(range(n_simulations), desc="Simulations"):
        # Run full simulation
        observed_mask, inclusion_probs = simulate_polis_routing(
            ground_truth=ground_truth,
            votes_distribution=votes_distribution,
            seed=base_seed + sim,
            show_progress=False,
        )

        # Update marginal counts
        marginal_counts += observed_mask.astype(float)

        # Update joint counts
        for pair_idx, (v1, v2) in enumerate(tracked_pairs):
            # Both saw comment c?
            both_saw = observed_mask[:, v1] & observed_mask[:, v2]
            joint_counts[:, pair_idx] += both_saw.astype(float)

        all_marginal_probs.append(inclusion_probs.copy())

    # Compute empirical probabilities
    empirical_marginal = marginal_counts / n_simulations  # (n_items, n_voters)
    empirical_joint = joint_counts / n_simulations  # (n_items, n_pairs)

    # Compute assumed joint (product of marginals)
    assumed_joint = np.zeros_like(empirical_joint)
    for pair_idx, (v1, v2) in enumerate(tracked_pairs):
        assumed_joint[:, pair_idx] = empirical_marginal[:, v1] * empirical_marginal[:, v2]

    # Compare
    joint_diff = empirical_joint - assumed_joint

    # Statistics
    mean_diff = joint_diff.mean()
    std_diff = joint_diff.std()
    max_diff = np.abs(joint_diff).max()

    # Relative error (where assumed > 0)
    mask = assumed_joint > 0.01
    if mask.any():
        relative_error = np.abs(joint_diff[mask] / assumed_joint[mask])
        mean_relative_error = relative_error.mean()
        max_relative_error = relative_error.max()
    else:
        mean_relative_error = np.nan
        max_relative_error = np.nan

    # Check if joint is systematically higher or lower than assumed
    # Negative diff means empirical < assumed (negative dependence)
    n_negative = (joint_diff < -0.01).sum()
    n_positive = (joint_diff > 0.01).sum()
    n_neutral = joint_diff.size - n_negative - n_positive

    print("\n" + "=" * 60)
    print("JOINT PROBABILITY INDEPENDENCE TEST")
    print("=" * 60)
    print(f"Simulations: {n_simulations}")
    print(f"Matrix size: {n_items} items × {n_voters} voters")
    print(f"Voter pairs tracked: {len(tracked_pairs)}")
    print()
    print("Comparison: Empirical P(i,j both see c) vs π_i,c × π_j,c")
    print("-" * 60)
    print(f"Mean difference:        {mean_diff:+.4f}")
    print(f"Std of difference:      {std_diff:.4f}")
    print(f"Max absolute diff:      {max_diff:.4f}")
    print(f"Mean relative error:    {mean_relative_error:.1%}")
    print(f"Max relative error:     {max_relative_error:.1%}")
    print()
    print("Direction of dependence:")
    print(f"  Negative (empirical < assumed): {n_negative} ({100*n_negative/joint_diff.size:.1f}%)")
    print(f"  Neutral (within ±0.01):         {n_neutral} ({100*n_neutral/joint_diff.size:.1f}%)")
    print(f"  Positive (empirical > assumed): {n_positive} ({100*n_positive/joint_diff.size:.1f}%)")

    if mean_diff < -0.01:
        print("\n→ NEGATIVE DEPENDENCE: Voters are LESS likely to both see")
        print("  the same comment than independence would predict.")
        print("  This makes sense: if voter i sees comment c, c's priority")
        print("  decreases (it becomes less 'uncertain'), so voter j is")
        print("  less likely to see it.")
    elif mean_diff > 0.01:
        print("\n→ POSITIVE DEPENDENCE: Voters are MORE likely to both see")
        print("  the same comment than independence would predict.")
    else:
        print("\n→ APPROXIMATELY INDEPENDENT (within noise)")

    return {
        "empirical_marginal": empirical_marginal,
        "empirical_joint": empirical_joint,
        "assumed_joint": assumed_joint,
        "joint_diff": joint_diff,
        "tracked_pairs": tracked_pairs,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "max_diff": max_diff,
        "mean_relative_error": mean_relative_error,
        "max_relative_error": max_relative_error,
        "n_simulations": n_simulations,
    }


def test_marginal_probability_consistency(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    n_simulations: int = 200,
    base_seed: int = 42,
) -> dict:
    """
    Test whether stored marginal probabilities match empirical rates.

    This verifies that π_i,c (stored during simulation) is correct.
    """
    n_items, n_voters = ground_truth.shape

    marginal_counts = np.zeros((n_items, n_voters))
    stored_probs_sum = np.zeros((n_items, n_voters))

    print(f"Running {n_simulations} simulations to verify marginal probabilities...")

    for sim in tqdm(range(n_simulations), desc="Simulations"):
        observed_mask, inclusion_probs = simulate_polis_routing(
            ground_truth=ground_truth,
            votes_distribution=votes_distribution,
            seed=base_seed + sim,
            show_progress=False,
        )

        marginal_counts += observed_mask.astype(float)
        stored_probs_sum += inclusion_probs

    empirical_marginal = marginal_counts / n_simulations
    mean_stored_probs = stored_probs_sum / n_simulations

    diff = empirical_marginal - mean_stored_probs

    print("\n" + "=" * 60)
    print("MARGINAL PROBABILITY CONSISTENCY TEST")
    print("=" * 60)
    print(f"Comparing empirical inclusion rates to stored π_i,c")
    print(f"Mean difference: {diff.mean():+.4f}")
    print(f"Max absolute diff: {np.abs(diff).max():.4f}")
    print(f"Std of difference: {diff.std():.4f}")

    # Note: We expect differences because stored probs are computed at session start
    # but the actual sampling is stochastic

    return {
        "empirical_marginal": empirical_marginal,
        "mean_stored_probs": mean_stored_probs,
        "diff": diff,
    }


if __name__ == "__main__":
    # Load a small dataset for testing
    completed_dir = Path("data/completed")
    processed_dir = Path("data/processed/preflib")

    # Use smallest dataset
    completed_files = sorted(completed_dir.glob("00069-*.npz"),
                            key=lambda p: p.stat().st_size)

    if not completed_files:
        print("No data files found!")
        exit(1)

    file_path = completed_files[0]
    print(f"Using dataset: {file_path.stem}")

    # Load data
    data = np.load(file_path)
    ground_truth = data["matrix"]

    processed_path = processed_dir / file_path.name
    processed_data = np.load(processed_path)
    processed_matrix = processed_data["matrix"]
    observed = ~np.isnan(processed_matrix)
    votes_distribution = observed.sum(axis=0)

    print(f"Matrix size: {ground_truth.shape[0]} items × {ground_truth.shape[1]} voters")
    print()

    # Test marginal consistency first
    marginal_results = test_marginal_probability_consistency(
        ground_truth, votes_distribution, n_simulations=100
    )

    print()

    # Test joint probability independence
    joint_results = test_joint_probability_independence(
        ground_truth, votes_distribution, n_simulations=100
    )
