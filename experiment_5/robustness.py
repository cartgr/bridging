"""
Core robustness experiment logic.

Compares robustness of Polis Group-Informed Consensus vs our Bridging Score
under random MCAR masking at various rates.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_5.polis import polis_consensus_pipeline
from experiment_5.masking import apply_random_mask, generate_trial_seeds


def run_single_trial(
    ground_truth_matrix: np.ndarray,
    mask_rate: float,
    seed: int,
    gt_bridging: np.ndarray,
    gt_polis: np.ndarray,
    polis_max_k: int = 5,
) -> Dict:
    """
    Run one masking trial and compute both estimators.

    Args:
        ground_truth_matrix: (n_items, n_voters) fully observed matrix
        mask_rate: fraction of entries to mask
        seed: random seed for this trial
        gt_bridging: (n_items,) ground truth bridging scores
        gt_polis: (n_items,) ground truth polis scores
        polis_max_k: max clusters for Polis k-means

    Returns:
        Dict containing:
        - bridging_scores: estimated bridging scores
        - polis_scores: estimated polis scores
        - actual_mask_rate: actual fraction masked (after constraints)
        - polis_metadata: metadata from polis pipeline
    """
    n_items, n_voters = ground_truth_matrix.shape

    # Apply random masking
    masked_matrix, observed_mask = apply_random_mask(
        ground_truth_matrix,
        mask_rate,
        seed=seed,
        min_observed_per_item=2,  # Need at least 2 voters per item for bridging
        min_observed_per_voter=1,
    )

    actual_mask_rate = 1.0 - observed_mask.sum() / observed_mask.size

    # Compute bridging scores on masked data
    # For bridging, we treat unobserved as "didn't vote"
    # The naive approach computes bridging on observed approvers only
    bridging_scores = _compute_bridging_on_masked(masked_matrix, observed_mask)

    # Compute Polis consensus on masked data
    polis_scores, polis_metadata = polis_consensus_pipeline(
        masked_matrix,
        observed_mask,
        n_pca_components=2,
        max_k=polis_max_k,
        seed=seed,
    )

    return {
        "bridging_scores": bridging_scores,
        "polis_scores": polis_scores,
        "actual_mask_rate": actual_mask_rate,
        "polis_k": polis_metadata["k_clusters"],
        "polis_cluster_sizes": polis_metadata["cluster_sizes"],
    }


def _compute_bridging_on_masked(
    masked_matrix: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute bridging scores treating masked data as complete.

    This is the "naive" estimator that ignores the missing data pattern.
    We compute pairwise disagreement only on co-observed pairs and
    bridging only on observed approvers.

    Args:
        masked_matrix: (n_items, n_voters) with NaN for missing
        observed_mask: (n_items, n_voters) boolean

    Returns:
        (n_items,) bridging scores
    """
    n_items, n_voters = masked_matrix.shape
    normalization = 4.0 / (n_voters ** 2)

    # Compute pairwise disagreement on observed pairs
    d_naive = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            # Find items observed by both
            both_observed = observed_mask[:, i] & observed_mask[:, j]
            n_both = both_observed.sum()

            if n_both == 0:
                d_naive[i, j] = np.nan
                d_naive[j, i] = np.nan
                continue

            # Count disagreements
            votes_i = masked_matrix[both_observed, i]
            votes_j = masked_matrix[both_observed, j]
            disagree_count = (votes_i != votes_j).sum()

            # Estimate disagreement rate
            d_naive[i, j] = disagree_count / n_both
            d_naive[j, i] = d_naive[i, j]

    # Compute bridging scores
    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Observed approvers of item c
        observed_c = observed_mask[c, :]
        approvers_mask = observed_c & (masked_matrix[c, :] == 1.0)
        approver_indices = np.where(approvers_mask)[0]

        if len(approver_indices) < 2:
            bridging_scores[c] = 0.0
            continue

        # Sum d_ij for pairs of observed approvers
        total = 0.0
        count = 0
        for idx_i in range(len(approver_indices)):
            for idx_j in range(idx_i + 1, len(approver_indices)):
                i = approver_indices[idx_i]
                j = approver_indices[idx_j]
                if not np.isnan(d_naive[i, j]):
                    total += d_naive[i, j]
                    count += 1

        if count > 0:
            bridging_scores[c] = normalization * total
        else:
            bridging_scores[c] = 0.0

    return bridging_scores


def run_masking_experiment(
    matrix: np.ndarray,
    mask_rates: List[float],
    n_trials: int = 50,
    base_seed: int = 42,
    polis_max_k: int = 5,
    show_progress: bool = True,
) -> Dict:
    """
    Run full masking experiment across all mask rates and trials.

    Args:
        matrix: (n_items, n_voters) fully observed ground truth matrix
        mask_rates: list of mask rates to test
        n_trials: number of trials per mask rate
        base_seed: base random seed
        polis_max_k: max clusters for Polis k-means
        show_progress: whether to show progress bar

    Returns:
        Dict containing:
        - gt_bridging: ground truth bridging scores
        - gt_polis: ground truth polis scores
        - results: dict of mask_rate -> list of trial results
        - mask_rates: list of mask rates tested
        - n_trials: number of trials
    """
    n_items, n_voters = matrix.shape

    # Compute ground truth scores on fully observed data
    full_mask = np.ones_like(matrix, dtype=bool)
    gt_bridging = compute_bridging_scores_vectorized(matrix)
    gt_polis, gt_metadata = polis_consensus_pipeline(
        matrix, full_mask, max_k=polis_max_k, seed=base_seed
    )

    # Generate all trial seeds
    total_trials = len(mask_rates) * n_trials
    all_seeds = generate_trial_seeds(total_trials, base_seed)

    # Run trials
    results = {rate: [] for rate in mask_rates}
    seed_idx = 0

    iterator = mask_rates
    if show_progress:
        iterator = tqdm(mask_rates, desc="Mask rates")

    for mask_rate in iterator:
        for trial in range(n_trials):
            seed = int(all_seeds[seed_idx])
            seed_idx += 1

            trial_result = run_single_trial(
                matrix,
                mask_rate,
                seed,
                gt_bridging,
                gt_polis,
                polis_max_k=polis_max_k,
            )
            results[mask_rate].append(trial_result)

    return {
        "gt_bridging": gt_bridging,
        "gt_polis": gt_polis,
        "gt_polis_metadata": gt_metadata,
        "results": results,
        "mask_rates": mask_rates,
        "n_trials": n_trials,
        "n_items": n_items,
        "n_voters": n_voters,
    }


def run_experiment_on_datasets(
    data_files: List[Path],
    mask_rates: List[float],
    n_trials: int = 50,
    base_seed: int = 42,
    polis_max_k: int = 5,
    show_progress: bool = True,
) -> Dict:
    """
    Run masking experiment on multiple datasets.

    Args:
        data_files: list of paths to .npz data files
        mask_rates: list of mask rates to test
        n_trials: number of trials per mask rate
        base_seed: base random seed
        polis_max_k: max clusters for Polis k-means
        show_progress: whether to show progress bar

    Returns:
        Dict containing results for each dataset
    """
    all_results = {}

    for data_file in data_files:
        name = data_file.stem
        if show_progress:
            print(f"\nProcessing {name}...")

        # Load data
        data = np.load(data_file)
        matrix = data["matrix"]

        # Run experiment
        result = run_masking_experiment(
            matrix,
            mask_rates,
            n_trials=n_trials,
            base_seed=base_seed,
            polis_max_k=polis_max_k,
            show_progress=show_progress,
        )

        result["name"] = name
        all_results[name] = result

    return all_results
