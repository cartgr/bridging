"""
Pol.is priority formula and PCA extremeness computation.

Priority formula (from Pol.is source code - conversation.clj):
    importance = (1 - p) × (E + 1) × a
    priority = [importance × (1 + 8 × 2^(-S/5))]²

Where (with Laplace smoothing):
- a = (A + 1) / (S + 2): smoothed agree rate
- p = (P + 1) / (S + 2): smoothed pass rate (P=0 in binary setting)
- E: extremeness in PCA space (L2 norm in first 2 PCs)
- S: total votes on comment

Reference: https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj
"""

from typing import Optional

import numpy as np


def power_iteration(
    data: np.ndarray, n_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """
    Compute first eigenvector using power iteration method.

    This matches Pol.is's approach (pca.clj uses power iteration).

    Args:
        data: (n_samples, n_features) centered data matrix
        n_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        (n_features,) first eigenvector (unit norm)
    """
    n_samples, n_features = data.shape

    # Initialize with random vector
    rng = np.random.default_rng(42)
    v = rng.standard_normal(n_features)
    v = v / np.linalg.norm(v)

    # Power iteration: v <- X^T X v, then normalize
    # This finds the eigenvector of X^T X (covariance matrix)
    for _ in range(n_iter):
        v_old = v.copy()

        # X^T X v = X^T (X v)
        Xv = data @ v  # (n_samples,)
        XtXv = data.T @ Xv  # (n_features,)

        # Normalize
        norm = np.linalg.norm(XtXv)
        if norm < 1e-10:
            return np.zeros(n_features)
        v = XtXv / norm

        # Check convergence
        if np.linalg.norm(v - v_old) < tol:
            break

    return v


def power_iteration_pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Compute PCA using power iteration (deflation method).

    Args:
        data: (n_samples, n_features) centered data matrix
        n_components: number of components to compute

    Returns:
        (n_components, n_features) principal components (eigenvectors)
    """
    n_samples, n_features = data.shape
    components = []
    residual = data.copy()

    for k in range(n_components):
        # Find leading eigenvector of residual
        v = power_iteration(residual)

        if np.linalg.norm(v) < 1e-10:
            # No more variance to explain
            v = np.zeros(n_features)

        components.append(v)

        # Deflate: remove this component's contribution
        # residual = residual - (residual @ v) @ v^T
        scores = residual @ v  # (n_samples,)
        residual = residual - np.outer(scores, v)

    return np.array(components)  # (n_components, n_features)


def compute_pca_extremeness(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    n_components: int = 2,
    min_votes_per_participant: int = 7,
    min_participants: int = 15,
) -> np.ndarray:
    """
    Compute PCA extremeness E(c) for each comment.

    E(c) = L2 norm of comment c's loading in first n_components PCs.

    Following Pol.is implementation:
    - Filter participants: need min(min_votes_per_participant, n_comments) votes
    - If fewer than min_participants qualify, greedily add top contributors
    - Missing values are imputed with per-comment (column) mean
    - PCA via power iteration
    - Extremeness is the L2 norm of each comment's loading on the first n PCs

    Reference: https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj

    Args:
        matrix: (n_items, n_voters) array with actual vote values
        observed_mask: (n_items, n_voters) boolean array, True where observed
        n_components: number of PCA components to use (default: 2)
        min_votes_per_participant: minimum votes to include participant (default: 7)
        min_participants: minimum participants for PCA (default: 15)

    Returns:
        (n_items,) array of extremeness values
    """
    n_items, n_voters = matrix.shape

    # Handle edge case: no observations at all
    if not observed_mask.any():
        return np.zeros(n_items)

    # === PARTICIPANT FILTERING (following Pol.is) ===
    # Count votes per participant (voter)
    votes_per_voter = observed_mask.sum(axis=0)  # (n_voters,)

    # Threshold: min(min_votes_per_participant, n_items)
    threshold = min(min_votes_per_participant, n_items)

    # Find participants meeting threshold
    qualifying_mask = votes_per_voter >= threshold
    n_qualifying = qualifying_mask.sum()

    if n_qualifying < min_participants:
        # Greedily add top contributors until we have min_participants
        sorted_indices = np.argsort(-votes_per_voter)  # Descending by vote count
        selected = np.zeros(n_voters, dtype=bool)

        for idx in sorted_indices:
            selected[idx] = True
            if selected.sum() >= min_participants:
                break

        participant_mask = selected
    else:
        participant_mask = qualifying_mask

    # If still not enough participants, use all available
    if participant_mask.sum() < 2:
        participant_mask = votes_per_voter > 0

    if participant_mask.sum() < 2:
        return np.zeros(n_items)

    # Filter to selected participants
    filtered_matrix = matrix[:, participant_mask]
    filtered_mask = observed_mask[:, participant_mask]
    n_filtered_voters = filtered_matrix.shape[1]

    # === IMPUTATION (per-comment mean) ===
    # Compute per-comment mean from observed values only
    observed_counts = filtered_mask.sum(axis=1, keepdims=True)  # (n_items, 1)
    observed_counts_safe = np.maximum(observed_counts, 1)  # Avoid division by zero

    # Sum of observed values per comment
    safe_matrix = np.where(filtered_mask, filtered_matrix, 0.0)
    observed_sums = safe_matrix.sum(axis=1, keepdims=True)  # (n_items, 1)

    # Per-comment mean
    comment_means = observed_sums / observed_counts_safe
    comment_means = np.where(observed_counts > 0, comment_means, 0.5)

    # Impute missing values with per-comment mean
    imputed_matrix = np.where(filtered_mask, filtered_matrix, comment_means)

    # === PCA via POWER ITERATION ===
    # Transpose to (n_voters, n_items) for PCA on voters
    data = imputed_matrix.T  # (n_filtered_voters, n_items)

    # Center the data (mean-center each comment/feature)
    data_centered = data - data.mean(axis=0)

    # Handle edge cases
    if n_filtered_voters < n_components or n_items < n_components:
        return np.zeros(n_items)

    if np.allclose(data_centered, 0):
        return np.zeros(n_items)

    try:
        # Use power iteration PCA (like Pol.is)
        n_comp = min(n_components, min(n_filtered_voters, n_items))
        components = power_iteration_pca(data_centered, n_comp)

        # Components has shape (n_components, n_items)
        # Each row is a principal component, each column is a comment's loading
        loadings = components.T  # (n_items, n_components)

        # Compute L2 norm of loadings for each comment
        extremeness = np.linalg.norm(loadings, axis=1)

        # Handle any NaN values
        extremeness = np.nan_to_num(extremeness, nan=0.0)

        return extremeness
    except Exception:
        # If PCA fails for any reason, return zeros
        return np.zeros(n_items)


def compute_vote_stats(
    matrix: np.ndarray, observed_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute vote statistics for each comment.

    Args:
        matrix: (n_items, n_voters) array with actual vote values (0 or 1)
                May contain NaN for unobserved entries.
        observed_mask: (n_items, n_voters) boolean array, True where observed

    Returns:
        Tuple of:
        - n_votes: (n_items,) array of total vote counts per comment (S)
        - n_agrees: (n_items,) array of agree counts per comment (A)
        - n_passes: (n_items,) array of pass counts (always 0 in binary setting)
    """
    # Count total votes per comment (S)
    n_votes = observed_mask.sum(axis=1).astype(float)

    # Count agrees (1s) per comment - only where observed (A)
    # Replace NaN with 0 before multiplying to avoid NaN propagation
    safe_matrix = np.nan_to_num(matrix, nan=0.0)
    n_agrees = (safe_matrix * observed_mask).sum(axis=1)

    # In binary setting, there are no passes (P = 0)
    n_passes = np.zeros_like(n_agrees)

    return n_votes, n_agrees, n_passes


def compute_priorities(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    extremeness: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Pol.is priority for each comment using Laplace smoothing.

    From Pol.is source (conversation.clj):
        importance = (1 - p) × (E + 1) × a
        priority = [importance × (1 + 8 × 2^(-S/5))]²

    Where (with Laplace smoothing):
        a = (A + 1) / (S + 2)  ; smoothed agree rate
        p = (P + 1) / (S + 2)  ; smoothed pass rate

    In binary setting (no passes), P = 0:
        a = (A + 1) / (S + 2)
        p = 1 / (S + 2)
        (1 - p) = (S + 1) / (S + 2)

    For a new comment (S=0, A=0):
        a = 1/2, (1-p) = 1/2, so base factor = 0.25

    Args:
        matrix: (n_items, n_voters) array with actual vote values
        observed_mask: (n_items, n_voters) boolean array, True where observed
        extremeness: (n_items,) array of PCA extremeness values.
                     If None, will be computed from the data.

    Returns:
        (n_items,) array of priority values
    """
    n_items, n_voters = matrix.shape

    # Compute vote statistics: S (total), A (agrees), P (passes)
    n_votes, n_agrees, n_passes = compute_vote_stats(matrix, observed_mask)

    # Laplace-smoothed rates (following Pol.is exactly)
    # a = (A + 1) / (S + 2)
    a = (n_agrees + 1) / (n_votes + 2)

    # p = (P + 1) / (S + 2), where P = 0 in binary setting
    p = (n_passes + 1) / (n_votes + 2)

    # Compute extremeness if not provided
    if extremeness is None:
        extremeness = compute_pca_extremeness(matrix, observed_mask)

    # Handle NaN in extremeness (can happen with degenerate PCA)
    extremeness = np.nan_to_num(extremeness, nan=0.0)

    # Compute importance: (1 - p) × (E + 1) × a
    importance = (1 - p) * (1 + extremeness) * a

    # Vote factor: (1 + 8 × 2^(-S/5)) = (1 + 2^(3 - S/5))
    # This favors comments with fewer votes (exploration bonus)
    vote_factor = 1 + np.power(2.0, 3 - n_votes / 5)

    # Combine and square
    priorities = (importance * vote_factor) ** 2

    return priorities


def compute_sampling_probabilities(
    priorities: np.ndarray, eligible_mask: np.ndarray
) -> np.ndarray:
    """
    Compute sampling probability for each comment given current priorities.

    p(c) = Priority(c) / Σ_{c' eligible} Priority(c')

    Args:
        priorities: (n_items,) array of priority values
        eligible_mask: (n_items,) boolean array, True for eligible comments

    Returns:
        (n_items,) array of sampling probabilities (0 for ineligible)
    """
    # Handle NaN in priorities
    safe_priorities = np.nan_to_num(priorities, nan=0.0)

    # Mask out ineligible comments
    masked_priorities = safe_priorities * eligible_mask

    # Compute total priority of eligible comments
    total = masked_priorities.sum()

    if total == 0 or np.isnan(total):
        # No eligible comments or all have zero priority
        # Fall back to uniform distribution over eligible
        n_eligible = eligible_mask.sum()
        if n_eligible > 0:
            return eligible_mask.astype(float) / n_eligible
        return np.zeros_like(priorities, dtype=float)

    # Normalize
    probs = masked_priorities / total

    return probs


def _compute_inclusion_exact_recursive(
    priorities: np.ndarray, eligible_indices: tuple, k_votes: int
) -> np.ndarray:
    """
    Compute exact inclusion probabilities using recursive formula with memoization.

    Uses: π_c(S, k) = p_c(S) + Σ_{j∈S, j≠c} p_j(S) × π_c(S\\{j}, k-1)
    """
    n_items = len(priorities)
    cache = {}

    def recurse(remaining_indices: tuple, remaining_k: int) -> np.ndarray:
        if remaining_k <= 0:
            return np.zeros(n_items)

        if remaining_k >= len(remaining_indices):
            result = np.zeros(n_items)
            for i in remaining_indices:
                result[i] = 1.0
            return result

        cache_key = (remaining_indices, remaining_k)
        if cache_key in cache:
            return cache[cache_key]

        remaining_priorities = np.array([priorities[i] for i in remaining_indices])
        total = remaining_priorities.sum()

        if total == 0:
            result = np.zeros(n_items)
            uniform_prob = remaining_k / len(remaining_indices)
            for i in remaining_indices:
                result[i] = uniform_prob
            cache[cache_key] = result
            return result

        selection_probs = remaining_priorities / total
        result = np.zeros(n_items)

        for idx, j in enumerate(remaining_indices):
            p_j = selection_probs[idx]
            result[j] += p_j
            new_remaining = tuple(i for i in remaining_indices if i != j)
            sub_probs = recurse(new_remaining, remaining_k - 1)
            for i in remaining_indices:
                if i != j:
                    result[i] += p_j * sub_probs[i]

        cache[cache_key] = result
        return result

    return recurse(eligible_indices, k_votes)


def _compute_inclusion_monte_carlo(
    priorities: np.ndarray,
    eligible_indices: tuple,
    k_votes: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Estimate inclusion probabilities via Monte Carlo simulation.

    Runs the sampling process n_samples times and counts inclusion frequency.
    """
    n_items = len(priorities)
    rng = np.random.default_rng(seed)
    inclusion_counts = np.zeros(n_items)

    eligible_priorities = np.array([priorities[i] for i in eligible_indices])
    total = eligible_priorities.sum()

    if total == 0:
        # Uniform when all zero
        p = np.ones(len(eligible_indices)) / len(eligible_indices)
    else:
        p = eligible_priorities / total

    for _ in range(n_samples):
        # Simulate one sampling process
        remaining_mask = np.ones(len(eligible_indices), dtype=bool)
        remaining_p = p.copy()

        for _ in range(k_votes):
            if remaining_p.sum() == 0:
                break
            # Normalize
            normalized_p = remaining_p / remaining_p.sum()
            # Sample
            idx = rng.choice(len(eligible_indices), p=normalized_p)
            # Record
            inclusion_counts[eligible_indices[idx]] += 1
            # Remove from eligible
            remaining_mask[idx] = False
            remaining_p[idx] = 0.0

    return inclusion_counts / n_samples


def compute_inclusion_probability_exact(
    priorities: np.ndarray, eligible_mask: np.ndarray, k_votes: int
) -> np.ndarray:
    """
    Compute exact inclusion probability for PPS sampling without replacement.

    For small problems (n_eligible <= 20 and k_votes <= 10), uses exact recursive
    computation. For larger problems, falls back to Monte Carlo estimation.

    Args:
        priorities: (n_items,) array of priority values (frozen)
        eligible_mask: (n_items,) boolean array, True for eligible comments
        k_votes: number of votes to draw

    Returns:
        (n_items,) array of inclusion probabilities
    """
    n_items = len(priorities)

    # Get eligible indices
    eligible_indices = tuple(i for i in range(n_items) if eligible_mask[i])
    n_eligible = len(eligible_indices)

    # Edge cases
    if n_eligible == 0 or k_votes == 0:
        return np.zeros(n_items)

    if k_votes >= n_eligible:
        return eligible_mask.astype(float)

    # Estimate state space size: Σ_{j=0}^{k} C(n, j)
    # Use threshold to decide exact vs Monte Carlo
    # For n=20, k=10: ~180k states (fast)
    # For n=30, k=10: ~30M states (slow)
    from math import comb
    estimated_states = sum(comb(n_eligible, j) for j in range(min(k_votes, n_eligible) + 1))

    if estimated_states <= 500_000:
        # Use exact recursive computation
        return _compute_inclusion_exact_recursive(priorities, eligible_indices, k_votes)
    else:
        # Fall back to Monte Carlo
        return _compute_inclusion_monte_carlo(
            priorities, eligible_indices, k_votes, n_samples=2000
        )
