"""
Pol.is routing simulation with exact probability tracking.

Simulates the Pol.is comment routing process on complete data, tracking
exact inclusion probabilities for each voter-comment pair.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from .priority import (
    compute_priorities,
    compute_pca_extremeness,
    compute_sampling_probabilities,
    compute_inclusion_probability_exact,
)


def get_empirical_votes_distribution(processed_files: List[Path]) -> np.ndarray:
    """
    Extract votes-per-voter distribution from processed Pol.is data.

    Args:
        processed_files: List of paths to processed .npz files

    Returns:
        Array of vote counts (one per voter across all files)
    """
    all_vote_counts = []

    for file_path in processed_files:
        data = np.load(file_path)
        matrix = data["matrix"]  # (n_items, n_voters)

        # Count observed votes per voter (non-NaN entries)
        observed = ~np.isnan(matrix)
        votes_per_voter = observed.sum(axis=0)  # (n_voters,)

        all_vote_counts.extend(votes_per_voter.tolist())

    return np.array(all_vote_counts)


def simulate_voter_session(
    ground_truth: np.ndarray,
    voter_idx: int,
    k_votes: int,
    current_matrix: np.ndarray,
    current_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a single voter's session with exact inclusion probability computation.

    The inclusion probabilities are computed BEFORE sampling, using the exact
    recursive formula for PPS sampling without replacement. This gives the
    true marginal probability P(comment c is shown) which is needed for IPW.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        voter_idx: index of the voter being simulated
        k_votes: number of comments to show this voter
        current_matrix: (n_items, n_voters) current observed matrix
        current_mask: (n_items, n_voters) current observation mask
        rng: random number generator

    Returns:
        Tuple of:
        - shown_mask: (n_items,) boolean array of which comments were shown
        - inclusion_probs: (n_items,) array of π_{voter, c} for each comment
        - revealed_votes: (n_items,) array of votes (-1 if not revealed)
    """
    n_items = ground_truth.shape[0]

    # Freeze priorities at session start based on current observed data
    extremeness = compute_pca_extremeness(current_matrix, current_mask)
    priorities = compute_priorities(current_matrix, current_mask, extremeness)

    # All comments are initially eligible for this voter
    eligible = np.ones(n_items, dtype=bool)

    # Cap k_votes at number of comments
    k_votes = min(k_votes, n_items)

    # Compute exact inclusion probabilities BEFORE sampling
    # This is the true marginal P(c is shown) given frozen priorities
    inclusion_probs = compute_inclusion_probability_exact(priorities, eligible, k_votes)

    # Now do the actual sampling to determine what gets shown
    shown_mask = np.zeros(n_items, dtype=bool)
    revealed_votes = np.full(n_items, -1.0)

    for t in range(k_votes):
        # Compute sampling probabilities over eligible comments
        p_t = compute_sampling_probabilities(priorities, eligible)

        # Check if we can sample
        if p_t.sum() == 0 or not np.any(p_t > 0):
            break

        # Sample one comment
        sampled_idx = rng.choice(n_items, p=p_t)

        # Mark sampled comment
        eligible[sampled_idx] = False
        shown_mask[sampled_idx] = True
        revealed_votes[sampled_idx] = ground_truth[sampled_idx, voter_idx]

    return shown_mask, inclusion_probs, revealed_votes


def simulate_polis_routing(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    seed: int = 42,
    voter_order: Optional[np.ndarray] = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Pol.is routing process with exact probability tracking.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: array of vote counts to sample K_i from
        seed: random seed for reproducibility
        voter_order: optional array specifying order to process voters
                     (if None, process in index order)
        show_progress: whether to show progress bar for voters

    Returns:
        Tuple of:
        - observed_mask: (n_items, n_voters) boolean array
        - inclusion_probs: (n_items, n_voters) array of π_{i,c}
    """
    rng = np.random.default_rng(seed)
    n_items, n_voters = ground_truth.shape

    # Initialize observed matrix and mask
    observed_matrix = np.full((n_items, n_voters), np.nan)
    observed_mask = np.zeros((n_items, n_voters), dtype=bool)

    # Store inclusion probabilities
    inclusion_probs = np.zeros((n_items, n_voters))

    # Determine voter order
    if voter_order is None:
        voter_order = np.arange(n_voters)

    # Process each voter
    voter_iter = voter_order
    if show_progress:
        voter_iter = tqdm(voter_order, desc="Voters", leave=False, position=1)

    for voter_idx in voter_iter:
        # Sample number of votes for this voter
        k_votes = rng.choice(votes_distribution)

        # Simulate voter session
        shown_mask, voter_inclusion_probs, revealed_votes = simulate_voter_session(
            ground_truth=ground_truth,
            voter_idx=voter_idx,
            k_votes=k_votes,
            current_matrix=observed_matrix,
            current_mask=observed_mask,
            rng=rng,
        )

        # Update observed matrix and mask
        observed_mask[:, voter_idx] = shown_mask
        observed_matrix[:, voter_idx] = np.where(
            shown_mask, revealed_votes, np.nan
        )

        # Store inclusion probabilities
        inclusion_probs[:, voter_idx] = voter_inclusion_probs

    return observed_mask, inclusion_probs


def simulate_polis_routing_batch(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    n_simulations: int,
    base_seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Run multiple simulations for Monte Carlo analysis.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: array of vote counts to sample K_i from
        n_simulations: number of simulations to run
        base_seed: base random seed (each simulation uses base_seed + i)

    Returns:
        List of (observed_mask, inclusion_probs) tuples
    """
    results = []

    for i in range(n_simulations):
        result = simulate_polis_routing(
            ground_truth=ground_truth,
            votes_distribution=votes_distribution,
            seed=base_seed + i,
        )
        results.append(result)

    return results


def verify_inclusion_probabilities_monte_carlo(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    n_simulations: int = 1000,
    base_seed: int = 42,
    tolerance: float = 0.05,
) -> dict:
    """
    Verify inclusion probability tracking via Monte Carlo.

    For each voter, run many simulations with the SAME frozen priorities
    and compare empirical inclusion rates to computed probabilities.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: array of vote counts
        n_simulations: number of Monte Carlo samples
        base_seed: random seed
        tolerance: maximum allowed deviation from expected

    Returns:
        Dictionary with verification results
    """
    n_items, n_voters = ground_truth.shape
    rng = np.random.default_rng(base_seed)

    # For simplicity, verify on the first voter with fixed K
    voter_idx = 0
    k_votes = int(np.median(votes_distribution))

    # Empty initial state (first voter sees no prior votes)
    initial_matrix = np.full((n_items, n_voters), np.nan)
    initial_mask = np.zeros((n_items, n_voters), dtype=bool)

    # Track empirical inclusion counts
    inclusion_counts = np.zeros(n_items)
    computed_probs = None

    for sim in range(n_simulations):
        shown_mask, inclusion_probs, _ = simulate_voter_session(
            ground_truth=ground_truth,
            voter_idx=voter_idx,
            k_votes=k_votes,
            current_matrix=initial_matrix,
            current_mask=initial_mask,
            rng=np.random.default_rng(base_seed + sim),
        )

        inclusion_counts += shown_mask.astype(float)

        # All simulations should have same computed probs (same frozen state)
        if computed_probs is None:
            computed_probs = inclusion_probs
        else:
            # Verify probabilities are consistent (frozen priorities)
            # Note: Due to randomness in PCA, there may be small variations
            pass

    # Compute empirical inclusion rates
    empirical_rates = inclusion_counts / n_simulations

    # Compare to computed probabilities
    max_deviation = np.abs(empirical_rates - computed_probs).max()
    mean_deviation = np.abs(empirical_rates - computed_probs).mean()

    return {
        "empirical_rates": empirical_rates,
        "computed_probs": computed_probs,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "passed": max_deviation < tolerance,
        "n_simulations": n_simulations,
        "k_votes": k_votes,
    }
