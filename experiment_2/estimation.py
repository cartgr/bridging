"""
IPW (Inverse Probability Weighting) estimation for bridging scores.

Estimates bridging scores from partially observed data using inclusion
probabilities to correct for sampling bias.

Includes:
- Standard IPW (high variance)
- Truncated IPW (capped weights)
- Normalized IPW (Hajek estimator)
- AIPW (Augmented/Doubly Robust)
"""

import numpy as np
from experiment_2.bridging import compute_bridging_pnorm


def estimate_pairwise_disagreement_ipw(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    inclusion_probs: np.ndarray,
    min_prob: float = 1e-6,
) -> np.ndarray:
    """
    Estimate pairwise disagreement d_ij using IPW.

    For each pair (i, j) and each comment c observed by both:
        d̂_ij = (1/|C|) × Σ_{c observed by both i,j} [1[disagree] / π_{ij,c}]

    where π_{ij,c} = π_{i,c} × π_{j,c} (independence assumption)

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes (NaN for missing)
        observed_mask: (n_items, n_voters) boolean array, True where observed
        inclusion_probs: (n_items, n_voters) array of π_{i,c}
        min_prob: minimum probability to avoid division by very small numbers

    Returns:
        (n_voters, n_voters) estimated pairwise disagreement matrix
    """
    n_items, n_voters = observed_matrix.shape

    # Clip inclusion probabilities to avoid instability
    clipped_probs = np.maximum(inclusion_probs, min_prob)

    # Initialize disagreement matrix
    d_hat = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            # Find comments observed by both i and j
            both_observed = observed_mask[:, i] & observed_mask[:, j]

            if not both_observed.any():
                # No common observations, can't estimate
                d_hat[i, j] = np.nan
                d_hat[j, i] = np.nan
                continue

            # Get votes for comments observed by both
            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]

            # Indicator of disagreement
            disagree = (votes_i != votes_j).astype(float)

            # Joint inclusion probability: π_{ij,c} = π_{i,c} × π_{j,c}
            pi_i = clipped_probs[both_observed, i]
            pi_j = clipped_probs[both_observed, j]
            pi_ij = pi_i * pi_j

            # IPW estimate
            # Note: We normalize by n_items (total comments), not just observed
            weighted_disagree = (disagree / pi_ij).sum()
            d_hat[i, j] = weighted_disagree / n_items
            d_hat[j, i] = d_hat[i, j]

    return d_hat


def estimate_bridging_scores_ipw(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    inclusion_probs: np.ndarray,
    min_prob: float = 1e-6,
) -> np.ndarray:
    """
    Estimate bridging scores using IPW.

    b̂^PD(c) = (4/n²) × Σ_{i<j, i,j∈observed_N_c} d̂_ij

    where d̂_ij is the IPW-estimated pairwise disagreement.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        inclusion_probs: (n_items, n_voters) array of π_{i,c}
        min_prob: minimum probability threshold

    Returns:
        (n_items,) array of estimated bridging scores in [0, 1]
    """
    n_items, n_voters = observed_matrix.shape

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # First compute estimated pairwise disagreement
    d_hat = estimate_pairwise_disagreement_ipw(
        observed_matrix, observed_mask, inclusion_probs, min_prob
    )

    # Compute bridging score for each comment
    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Find voters who approved comment c (among those who observed it)
        observed_c = observed_mask[c, :]
        if not observed_c.any():
            bridging_scores[c] = np.nan
            continue

        # Get approvers (those who voted 1 on comment c)
        approvers_mask = observed_c & (observed_matrix[c, :] == 1.0)
        approver_indices = np.where(approvers_mask)[0]

        if len(approver_indices) < 2:
            bridging_scores[c] = 0.0
            continue

        # Sum d̂_ij for all pairs of observed approvers
        total = 0.0
        for idx_i in range(len(approver_indices)):
            for idx_j in range(idx_i + 1, len(approver_indices)):
                i = approver_indices[idx_i]
                j = approver_indices[idx_j]
                if not np.isnan(d_hat[i, j]):
                    total += d_hat[i, j]

        bridging_scores[c] = normalization * total

    return bridging_scores


def estimate_bridging_scores_ipw_direct(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    inclusion_probs: np.ndarray,
    min_prob: float = 1e-6,
) -> np.ndarray:
    """
    Estimate bridging scores using direct IPW formulation.

    This uses the efficient O(m) computation directly with IPW weighting:

    b̂_c = (4/n²) × (1/|C|) × Σ_{c'∈C} [IPW-weighted count of approvers of c who agree on c']
                                      × [IPW-weighted count of approvers of c who disagree on c']

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        inclusion_probs: (n_items, n_voters) array of π_{i,c}
        min_prob: minimum probability threshold

    Returns:
        (n_items,) array of estimated bridging scores in [0, 1]
    """
    n_items, n_voters = observed_matrix.shape

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # Clip inclusion probabilities
    clipped_probs = np.maximum(inclusion_probs, min_prob)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Find observed approvers of comment c
        observed_c = observed_mask[c, :]
        approvers_mask = observed_c & (observed_matrix[c, :] == 1.0)

        if approvers_mask.sum() < 2:
            bridging_scores[c] = 0.0
            continue

        total_disagreement = 0.0

        for cp in range(n_items):
            # For each comment c', count approvers of c who:
            # - observed c' AND approved c'
            # - observed c' AND disapproved c'

            # Approvers of c who also observed c'
            approvers_observed_cp = approvers_mask & observed_mask[cp, :]

            if not approvers_observed_cp.any():
                continue

            # Among approvers of c who observed c':
            votes_cp = observed_matrix[cp, approvers_observed_cp]
            probs_c = clipped_probs[c, approvers_observed_cp]
            probs_cp = clipped_probs[cp, approvers_observed_cp]

            # IPW weights for each approver's observation of c'
            # Joint probability of observing both c and c' for this voter
            weights = 1.0 / (probs_c * probs_cp)

            # Weighted count of those who approve c'
            weighted_approve = (weights * (votes_cp == 1.0)).sum()
            # Weighted count of those who disapprove c'
            weighted_disapprove = (weights * (votes_cp == 0.0)).sum()

            # Product counts disagreeing pairs
            total_disagreement += weighted_approve * weighted_disapprove

        bridging_scores[c] = normalization * total_disagreement / n_items

    return bridging_scores


def estimate_bridging_scores_naive(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """
    Naive bridging score estimation without IPW correction.

    Simply computes bridging scores treating observed data as if it were
    complete. This serves as a baseline for comparison.

    b̂^PD(c) = (4/n²) × Σ_{i<j, i,j∈observed_N_c} d̂_ij

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array

    Returns:
        (n_items,) array of estimated bridging scores in [0, 1]
    """
    n_items, n_voters = observed_matrix.shape

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # Compute pairwise disagreement only on observed entries
    d_naive = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            # Find comments observed by both
            both_observed = observed_mask[:, i] & observed_mask[:, j]
            n_both = both_observed.sum()

            if n_both == 0:
                d_naive[i, j] = np.nan
                d_naive[j, i] = np.nan
                continue

            # Count disagreements
            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]
            disagree_count = (votes_i != votes_j).sum()

            # Naive estimate: just use fraction of observed
            d_naive[i, j] = disagree_count / n_both
            d_naive[j, i] = d_naive[i, j]

    # Compute bridging scores
    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        observed_c = observed_mask[c, :]
        if not observed_c.any():
            bridging_scores[c] = np.nan
            continue

        approvers_mask = observed_c & (observed_matrix[c, :] == 1.0)
        approver_indices = np.where(approvers_mask)[0]

        if len(approver_indices) < 2:
            bridging_scores[c] = 0.0
            continue

        total = 0.0
        count = 0
        for idx_i in range(len(approver_indices)):
            for idx_j in range(idx_i + 1, len(approver_indices)):
                i = approver_indices[idx_i]
                j = approver_indices[idx_j]
                if not np.isnan(d_naive[i, j]):
                    total += d_naive[i, j]
                    count += 1

        # Apply normalization factor
        if count > 0:
            bridging_scores[c] = normalization * total
        else:
            bridging_scores[c] = 0.0

    return bridging_scores


def estimate_pnorm_naive(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    p: float = -10.0,
) -> np.ndarray:
    """
    Naive p-norm estimation from partially observed data.

    Simply computes p-norm scores treating observed data as complete,
    using NaN for missing values.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes (NaN for missing)
        observed_mask: (n_items, n_voters) boolean array
        p: p-norm parameter (default -10 for approx min)

    Returns:
        (n_items,) array of estimated p-norm bridging scores
    """
    # Create a matrix where unobserved entries are NaN
    matrix_with_nan = np.where(observed_mask, observed_matrix, np.nan)

    # Use nanmean for approval calculations
    return compute_bridging_pnorm(matrix_with_nan, p=p)


# =============================================================================
# Improved IPW Estimators
# =============================================================================


def estimate_bridging_scores_truncated_ipw(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    inclusion_probs: np.ndarray,
    max_weight: float = 20.0,
    min_prob: float = 1e-6,
) -> np.ndarray:
    """
    Estimate bridging scores using truncated IPW.

    Caps the IPW weights at max_weight to reduce variance at the cost of
    some bias. This is a simple but effective variance reduction technique.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        inclusion_probs: (n_items, n_voters) array of π_{i,c}
        max_weight: maximum weight (default 20)
        min_prob: minimum probability threshold

    Returns:
        (n_items,) array of estimated bridging scores
    """
    n_items, n_voters = observed_matrix.shape
    normalization = 4.0 / (n_voters ** 2)

    # Clip inclusion probabilities
    clipped_probs = np.maximum(inclusion_probs, min_prob)

    # Compute pairwise disagreement with truncated weights
    d_hat = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            both_observed = observed_mask[:, i] & observed_mask[:, j]

            if not both_observed.any():
                d_hat[i, j] = np.nan
                d_hat[j, i] = np.nan
                continue

            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]
            disagree = (votes_i != votes_j).astype(float)

            # Joint inclusion probability with truncated weights
            pi_i = clipped_probs[both_observed, i]
            pi_j = clipped_probs[both_observed, j]
            pi_ij = pi_i * pi_j

            # Truncate weights
            weights = np.minimum(1.0 / pi_ij, max_weight)

            # Weighted estimate (normalized by sum of weights for stability)
            weighted_disagree = (disagree * weights).sum()
            weight_sum = weights.sum()

            if weight_sum > 0:
                d_hat[i, j] = weighted_disagree / weight_sum
            else:
                d_hat[i, j] = np.nan
            d_hat[j, i] = d_hat[i, j]

    # Compute bridging scores
    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        observed_c = observed_mask[c, :]
        if not observed_c.any():
            bridging_scores[c] = np.nan
            continue

        approvers_mask = observed_c & (observed_matrix[c, :] == 1.0)
        approver_indices = np.where(approvers_mask)[0]

        if len(approver_indices) < 2:
            bridging_scores[c] = 0.0
            continue

        total = 0.0
        count = 0
        for idx_i in range(len(approver_indices)):
            for idx_j in range(idx_i + 1, len(approver_indices)):
                i = approver_indices[idx_i]
                j = approver_indices[idx_j]
                if not np.isnan(d_hat[i, j]):
                    total += d_hat[i, j]
                    count += 1

        if count > 0:
            bridging_scores[c] = normalization * total
        else:
            bridging_scores[c] = 0.0

    return bridging_scores


def estimate_bridging_scores_aipw(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    inclusion_probs: np.ndarray,
    min_prob: float = 1e-6,
    max_weight: float = 50.0,
) -> np.ndarray:
    """
    Estimate bridging scores using AIPW (Augmented IPW / Doubly Robust).

    AIPW combines IPW with an outcome model:
        θ̂ = E[m(X)] + E[(Y - m(X)) × R / π]

    For disagreement estimation, we use the naive disagreement rate as the
    outcome model m(X). This gives us:
        d̂_ij = d_naive_ij + Σ_c [(disagree - d_naive_ij) × observed / π_ij]

    This is doubly robust: consistent if either:
    1. The propensity model (π) is correct, OR
    2. The outcome model (d_naive) is correct

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        inclusion_probs: (n_items, n_voters) array of π_{i,c}
        min_prob: minimum probability threshold
        max_weight: maximum IPW weight for stability

    Returns:
        (n_items,) array of estimated bridging scores
    """
    n_items, n_voters = observed_matrix.shape
    normalization = 4.0 / (n_voters ** 2)

    clipped_probs = np.maximum(inclusion_probs, min_prob)

    # Step 1: Compute naive disagreement estimates (outcome model)
    d_naive = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            both_observed = observed_mask[:, i] & observed_mask[:, j]
            n_both = both_observed.sum()

            if n_both == 0:
                d_naive[i, j] = 0.5  # Prior: assume 50% disagreement
                d_naive[j, i] = 0.5
                continue

            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]
            disagree_count = (votes_i != votes_j).sum()

            d_naive[i, j] = disagree_count / n_both
            d_naive[j, i] = d_naive[i, j]

    # Step 2: Compute AIPW correction
    d_aipw = d_naive.copy()

    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            both_observed = observed_mask[:, i] & observed_mask[:, j]

            if not both_observed.any():
                continue

            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]
            disagree = (votes_i != votes_j).astype(float)

            # Residuals from outcome model
            residuals = disagree - d_naive[i, j]

            # Joint inclusion probabilities
            pi_i = clipped_probs[both_observed, i]
            pi_j = clipped_probs[both_observed, j]
            pi_ij = pi_i * pi_j

            # Truncated IPW weights
            weights = np.minimum(1.0 / pi_ij, max_weight)

            # AIPW correction: add weighted residuals
            # Normalize by n_items to get per-comment contribution
            correction = (residuals * weights).sum() / n_items

            d_aipw[i, j] = d_naive[i, j] + correction
            d_aipw[j, i] = d_aipw[i, j]

            # Clip to valid range [0, 1]
            d_aipw[i, j] = np.clip(d_aipw[i, j], 0.0, 1.0)
            d_aipw[j, i] = d_aipw[i, j]

    # Step 3: Compute bridging scores
    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        observed_c = observed_mask[c, :]
        if not observed_c.any():
            bridging_scores[c] = np.nan
            continue

        approvers_mask = observed_c & (observed_matrix[c, :] == 1.0)
        approver_indices = np.where(approvers_mask)[0]

        if len(approver_indices) < 2:
            bridging_scores[c] = 0.0
            continue

        total = 0.0
        count = 0
        for idx_i in range(len(approver_indices)):
            for idx_j in range(idx_i + 1, len(approver_indices)):
                i = approver_indices[idx_i]
                j = approver_indices[idx_j]
                if not np.isnan(d_aipw[i, j]):
                    total += d_aipw[i, j]
                    count += 1

        if count > 0:
            bridging_scores[c] = normalization * total
        else:
            bridging_scores[c] = 0.0

    return bridging_scores


def estimate_bridging_scores_matrix_completion(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    n_factors: int = 10,
    max_iter: int = 100,
    learning_rate: float = 0.01,
    reg: float = 0.1,
) -> np.ndarray:
    """
    Estimate bridging scores via matrix completion.

    Uses alternating least squares to learn a low-rank factorization:
        Vote(c, v) ≈ σ(item_factors[c] · voter_factors[v])

    Then imputes missing votes and computes bridging on completed matrix.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        n_factors: number of latent factors
        max_iter: maximum iterations for ALS
        learning_rate: SGD learning rate
        reg: regularization strength

    Returns:
        (n_items,) array of estimated bridging scores
    """
    from experiment_2.bridging import compute_bridging_scores_vectorized

    n_items, n_voters = observed_matrix.shape

    # Initialize factors randomly
    rng = np.random.default_rng(42)
    item_factors = rng.normal(0, 0.1, (n_items, n_factors))
    voter_factors = rng.normal(0, 0.1, (n_voters, n_factors))

    # Get observed entries
    obs_rows, obs_cols = np.where(observed_mask)
    obs_values = observed_matrix[obs_rows, obs_cols]

    # SGD for matrix factorization
    for iteration in range(max_iter):
        # Shuffle observations
        perm = rng.permutation(len(obs_rows))

        for idx in perm:
            c, v = obs_rows[idx], obs_cols[idx]
            y = obs_values[idx]

            # Prediction
            pred = 1 / (1 + np.exp(-np.dot(item_factors[c], voter_factors[v])))

            # Error
            error = y - pred

            # Gradient update
            item_grad = error * pred * (1 - pred) * voter_factors[v] - reg * item_factors[c]
            voter_grad = error * pred * (1 - pred) * item_factors[c] - reg * voter_factors[v]

            item_factors[c] += learning_rate * item_grad
            voter_factors[v] += learning_rate * voter_grad

    # Impute missing values
    completed_matrix = 1 / (1 + np.exp(-item_factors @ voter_factors.T))

    # Threshold to binary
    completed_matrix = (completed_matrix > 0.5).astype(float)

    # Fill in observed values (trust observations over predictions)
    completed_matrix = np.where(observed_mask, observed_matrix, completed_matrix)

    # Compute bridging on completed matrix
    return compute_bridging_scores_vectorized(completed_matrix)
