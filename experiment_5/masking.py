"""
Random masking utilities for robustness experiments.

Implements Missing Completely At Random (MCAR) masking to simulate
partial observation of voting data.
"""

from typing import Tuple

import numpy as np


def apply_random_mask(
    matrix: np.ndarray,
    mask_rate: float,
    seed: int = 42,
    min_observed_per_item: int = 2,
    min_observed_per_voter: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply MCAR random masking to a fully observed matrix.

    Masks (hides) a fraction of entries uniformly at random, with optional
    constraints to ensure minimum observations per item and voter.

    Args:
        matrix: (n_items, n_voters) fully observed array
        mask_rate: fraction of entries to mask (0.0 to 1.0)
        seed: random seed for reproducibility
        min_observed_per_item: minimum votes per item to keep (default: 2)
        min_observed_per_voter: minimum votes per voter to keep (default: 1)

    Returns:
        Tuple of:
        - masked_matrix: (n_items, n_voters) array with NaN for masked entries
        - observed_mask: (n_items, n_voters) boolean array, True where observed
    """
    if mask_rate < 0 or mask_rate > 1:
        raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")

    n_items, n_voters = matrix.shape
    rng = np.random.default_rng(seed)

    if mask_rate == 0:
        # No masking
        return matrix.copy(), np.ones_like(matrix, dtype=bool)

    if mask_rate == 1:
        # Mask everything (edge case)
        return np.full_like(matrix, np.nan), np.zeros_like(matrix, dtype=bool)

    # Generate random mask
    random_values = rng.random(matrix.shape)
    observed_mask = random_values >= mask_rate

    # Apply minimum observation constraints if needed
    if min_observed_per_item > 0 or min_observed_per_voter > 0:
        observed_mask = _apply_min_observation_constraints(
            observed_mask,
            min_observed_per_item,
            min_observed_per_voter,
            rng,
        )

    # Create masked matrix
    masked_matrix = np.where(observed_mask, matrix, np.nan)

    return masked_matrix, observed_mask


def _apply_min_observation_constraints(
    observed_mask: np.ndarray,
    min_per_item: int,
    min_per_voter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ensure minimum observations per item and voter by un-masking entries.

    Args:
        observed_mask: (n_items, n_voters) initial boolean mask
        min_per_item: minimum observations per item (row)
        min_per_voter: minimum observations per voter (column)
        rng: random number generator

    Returns:
        Updated (n_items, n_voters) boolean mask
    """
    n_items, n_voters = observed_mask.shape
    mask = observed_mask.copy()

    # Ensure minimum per item (row)
    if min_per_item > 0:
        for i in range(n_items):
            observed_count = mask[i, :].sum()
            if observed_count < min_per_item:
                # Find masked positions and randomly un-mask some
                masked_positions = np.where(~mask[i, :])[0]
                n_to_unmask = min(min_per_item - observed_count, len(masked_positions))
                if n_to_unmask > 0:
                    to_unmask = rng.choice(
                        masked_positions, size=int(n_to_unmask), replace=False
                    )
                    mask[i, to_unmask] = True

    # Ensure minimum per voter (column)
    if min_per_voter > 0:
        for j in range(n_voters):
            observed_count = mask[:, j].sum()
            if observed_count < min_per_voter:
                # Find masked positions and randomly un-mask some
                masked_positions = np.where(~mask[:, j])[0]
                n_to_unmask = min(min_per_voter - observed_count, len(masked_positions))
                if n_to_unmask > 0:
                    to_unmask = rng.choice(
                        masked_positions, size=int(n_to_unmask), replace=False
                    )
                    mask[to_unmask, j] = True

    return mask


def generate_trial_seeds(n_trials: int, base_seed: int = 42) -> np.ndarray:
    """
    Generate reproducible seeds for multiple masking trials.

    Args:
        n_trials: number of trials
        base_seed: base random seed

    Returns:
        (n_trials,) array of integer seeds
    """
    rng = np.random.default_rng(base_seed)
    # Generate seeds in a large range to avoid collisions
    seeds = rng.integers(0, 2**31, size=n_trials)
    return seeds


def compute_actual_mask_rate(observed_mask: np.ndarray) -> float:
    """
    Compute the actual observation rate after applying constraints.

    Args:
        observed_mask: (n_items, n_voters) boolean mask

    Returns:
        Fraction of entries that are observed
    """
    return observed_mask.sum() / observed_mask.size
