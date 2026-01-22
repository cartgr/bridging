"""
Simulation-based robustness experiment.

Compares robustness of Polis Group-Informed Consensus vs our Bridging Score
under realistic Polis routing (informative missingness).

Uses Monte Carlo estimation for inclusion probabilities (faster than exact).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_2.priority import (
    compute_priorities,
    compute_pca_extremeness,
    compute_sampling_probabilities,
)
from experiment_2.estimation import estimate_bridging_scores_ipw, estimate_bridging_scores_naive
from experiment_5.polis import polis_consensus_pipeline


def _compute_inclusion_monte_carlo_fast(
    priorities: np.ndarray,
    k_votes: int,
    n_samples: int = 100,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Fast Monte Carlo estimation of inclusion probabilities.

    Args:
        priorities: (n_items,) priority values
        k_votes: number of items to sample
        n_samples: number of MC samples
        rng: random number generator

    Returns:
        (n_items,) inclusion probabilities
    """
    if rng is None:
        rng = np.random.default_rng()

    n_items = len(priorities)
    if n_items == 0 or k_votes == 0:
        return np.zeros(n_items)

    k_votes = min(k_votes, n_items)
    total = priorities.sum()

    if total == 0:
        p = np.ones(n_items) / n_items
    else:
        p = priorities / total

    # Fast vectorized MC: sample k_votes items n_samples times
    inclusion_counts = np.zeros(n_items)

    for _ in range(n_samples):
        remaining_p = p.copy()
        for _ in range(k_votes):
            if remaining_p.sum() == 0:
                break
            normalized_p = remaining_p / remaining_p.sum()
            idx = rng.choice(n_items, p=normalized_p)
            inclusion_counts[idx] += 1
            remaining_p[idx] = 0.0

    return inclusion_counts / n_samples


def simulate_voter_session_polis(
    ground_truth: np.ndarray,
    voter_idx: int,
    k_votes: int,
    priorities: np.ndarray,
    rng: np.random.Generator,
    mc_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a single voter's session with Polis-style priority-based routing.

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        voter_idx: index of the voter being simulated
        k_votes: number of comments to show this voter
        priorities: (n_items,) precomputed priority values
        rng: random number generator
        mc_samples: number of MC samples for inclusion prob estimation

    Returns:
        Tuple of (shown_mask, inclusion_probs, revealed_votes)
    """
    n_items = ground_truth.shape[0]
    k_votes = min(k_votes, n_items)

    # MC estimation of inclusion probabilities
    inclusion_probs = _compute_inclusion_monte_carlo_fast(
        priorities, k_votes, n_samples=mc_samples, rng=rng
    )

    # Priority-based sampling without replacement
    shown_mask = np.zeros(n_items, dtype=bool)
    revealed_votes = np.full(n_items, np.nan)
    eligible = np.ones(n_items, dtype=bool)

    for _ in range(k_votes):
        p = compute_sampling_probabilities(priorities, eligible)
        if p.sum() == 0:
            break
        sampled_idx = rng.choice(n_items, p=p)
        eligible[sampled_idx] = False
        shown_mask[sampled_idx] = True
        revealed_votes[sampled_idx] = ground_truth[sampled_idx, voter_idx]

    return shown_mask, inclusion_probs, revealed_votes


def simulate_polis_routing(
    ground_truth: np.ndarray,
    votes_distribution: np.ndarray,
    seed: int = 42,
    mc_samples: int = 500,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Polis-style routing with PCA-based priorities.

    Priorities are computed using PCA extremeness for every voter (like real Polis).

    Args:
        ground_truth: (n_items, n_voters) complete vote matrix
        votes_distribution: array of vote counts to sample from
        seed: random seed
        mc_samples: MC samples for inclusion prob estimation
        show_progress: whether to show voter progress bar

    Returns:
        Tuple of (observed_mask, inclusion_probs)
    """
    rng = np.random.default_rng(seed)
    n_items, n_voters = ground_truth.shape

    observed_matrix = np.full((n_items, n_voters), np.nan)
    observed_mask = np.zeros((n_items, n_voters), dtype=bool)
    inclusion_probs = np.zeros((n_items, n_voters))

    voter_iter = range(n_voters)
    if show_progress:
        voter_iter = tqdm(voter_iter, desc="    Voters", leave=False)

    for voter_idx in voter_iter:
        # Compute PCA extremeness and priorities for this voter
        extremeness = compute_pca_extremeness(observed_matrix, observed_mask)
        priorities = compute_priorities(observed_matrix, observed_mask, extremeness)

        k_votes = rng.choice(votes_distribution)

        shown_mask, voter_inclusion_probs, revealed_votes = simulate_voter_session_polis(
            ground_truth=ground_truth,
            voter_idx=voter_idx,
            k_votes=k_votes,
            priorities=priorities,
            rng=rng,
            mc_samples=mc_samples,
        )

        observed_mask[:, voter_idx] = shown_mask
        observed_matrix[:, voter_idx] = np.where(shown_mask, revealed_votes, np.nan)
        inclusion_probs[:, voter_idx] = voter_inclusion_probs

    return observed_mask, inclusion_probs


def get_default_votes_distribution(n_items: int, target_obs_rate: float = 0.5) -> np.ndarray:
    """
    Create a votes distribution that achieves approximately the target observation rate.
    """
    target_k = int(n_items * target_obs_rate)
    min_k = max(1, target_k - 2)
    max_k = min(n_items, target_k + 2)
    return np.arange(min_k, max_k + 1)


def run_single_simulation_trial(
    ground_truth_matrix: np.ndarray,
    votes_distribution: np.ndarray,
    seed: int,
    gt_bridging: np.ndarray,
    gt_polis: np.ndarray,
    polis_max_k: int = 5,
    mc_samples: int = 500,
    show_voter_progress: bool = False,
) -> Dict:
    """
    Run one simulation trial with Polis-style routing and compute all estimators.

    Args:
        ground_truth_matrix: (n_items, n_voters) fully observed matrix
        votes_distribution: array of vote counts to sample from
        seed: random seed for this trial
        gt_bridging: (n_items,) ground truth bridging scores
        gt_polis: (n_items,) ground truth polis scores
        polis_max_k: max clusters for Polis k-means
        mc_samples: MC samples for inclusion probability estimation

    Returns:
        Dict with bridging_naive, bridging_ipw, polis_scores, etc.
    """
    n_items, n_voters = ground_truth_matrix.shape

    # Simulate Polis-style routing with PCA-based priorities
    observed_mask, inclusion_probs = simulate_polis_routing(
        ground_truth_matrix,
        votes_distribution,
        seed=seed,
        mc_samples=mc_samples,
        show_progress=show_voter_progress,
    )

    # Create observed matrix
    observed_matrix = np.where(observed_mask, ground_truth_matrix, np.nan)
    observation_rate = observed_mask.sum() / observed_mask.size

    # Compute bridging scores - naive (no IPW correction)
    bridging_naive = estimate_bridging_scores_naive(observed_matrix, observed_mask)

    # Compute bridging scores - with IPW correction
    bridging_ipw = estimate_bridging_scores_ipw(
        observed_matrix, observed_mask, inclusion_probs, min_prob=1e-6
    )

    # Compute Polis consensus
    polis_scores, polis_metadata = polis_consensus_pipeline(
        observed_matrix,
        observed_mask,
        n_pca_components=2,
        max_k=polis_max_k,
        seed=seed,
    )

    return {
        "bridging_naive": bridging_naive,
        "bridging_ipw": bridging_ipw,
        "polis_scores": polis_scores,
        "observation_rate": observation_rate,
        "polis_k": polis_metadata["k_clusters"],
        "polis_cluster_sizes": polis_metadata["cluster_sizes"],
    }


def run_simulation_experiment(
    matrix: np.ndarray,
    votes_distributions: Dict[str, np.ndarray],
    n_trials: int = 30,
    base_seed: int = 42,
    polis_max_k: int = 5,
    mc_samples: int = 500,
    show_progress: bool = True,
) -> Dict:
    """
    Run simulation experiment across different observation rate targets.

    Args:
        matrix: (n_items, n_voters) fully observed ground truth matrix
        votes_distributions: dict of name -> votes distribution array
        n_trials: number of trials per distribution
        base_seed: base random seed
        polis_max_k: max clusters for Polis k-means
        mc_samples: MC samples for inclusion probability estimation
        show_progress: whether to show progress bar

    Returns:
        Dict with gt_bridging, gt_polis, results, etc.
    """
    n_items, n_voters = matrix.shape

    # Compute ground truth scores
    full_mask = np.ones_like(matrix, dtype=bool)
    gt_bridging = compute_bridging_scores_vectorized(matrix)
    gt_polis, gt_metadata = polis_consensus_pipeline(
        matrix, full_mask, max_k=polis_max_k, seed=base_seed
    )

    # Run trials
    results = {name: [] for name in votes_distributions}
    trial_idx = 0
    n_dists = len(votes_distributions)

    dist_iter = votes_distributions.items()
    if show_progress:
        dist_iter = tqdm(list(dist_iter), desc="  Obs rates", position=0)

    for dist_name, votes_dist in dist_iter:
        trial_iter = range(n_trials)
        if show_progress:
            trial_iter = tqdm(trial_iter, desc=f"    {dist_name}", position=1, leave=False)

        for trial in trial_iter:
            seed = base_seed + trial_idx
            trial_idx += 1

            # Show voter progress only for first trial of first distribution
            show_voters = show_progress and trial == 0 and trial_idx <= n_trials

            trial_result = run_single_simulation_trial(
                matrix,
                votes_dist,
                seed,
                gt_bridging,
                gt_polis,
                polis_max_k=polis_max_k,
                mc_samples=mc_samples,
                show_voter_progress=show_voters,
            )
            results[dist_name].append(trial_result)

    return {
        "gt_bridging": gt_bridging,
        "gt_polis": gt_polis,
        "gt_polis_metadata": gt_metadata,
        "results": results,
        "distribution_names": list(votes_distributions.keys()),
        "n_trials": n_trials,
        "n_items": n_items,
        "n_voters": n_voters,
    }


def create_votes_distributions(n_items: int) -> Dict[str, np.ndarray]:
    """
    Create a set of votes distributions targeting different observation rates.
    """
    distributions = {}
    target_rates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    for rate in target_rates:
        target_k = max(1, int(n_items * rate))
        min_k = max(1, target_k - 2)
        max_k = min(n_items, target_k + 2)
        name = f"obs_{int(rate * 100)}"
        distributions[name] = np.arange(min_k, max_k + 1)

    return distributions


def run_simulation_experiment_on_datasets(
    data_files: List[Path],
    n_trials: int = 30,
    base_seed: int = 42,
    polis_max_k: int = 5,
    mc_samples: int = 500,
    show_progress: bool = True,
) -> Dict:
    """
    Run simulation experiment on multiple datasets.
    """
    all_results = {}

    for data_file in data_files:
        name = data_file.stem
        if show_progress:
            print(f"\nProcessing {name}...")

        data = np.load(data_file)
        matrix = data["matrix"]
        n_items = matrix.shape[0]

        votes_distributions = create_votes_distributions(n_items)

        result = run_simulation_experiment(
            matrix,
            votes_distributions,
            n_trials=n_trials,
            base_seed=base_seed,
            polis_max_k=polis_max_k,
            mc_samples=mc_samples,
            show_progress=show_progress,
        )

        result["name"] = name
        all_results[name] = result

    return all_results
