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

            # Indicator of disagreement (only 1.0 vs 0.0, not skips)
            disagree = (
                ((votes_i == 1.0) & (votes_j == 0.0)) |
                ((votes_i == 0.0) & (votes_j == 1.0))
            ).astype(float)

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
    min_approvers: int = 5,
) -> np.ndarray:
    """
    Naive PD bridging score estimation using splitter-based approach.

    For each target x and splitter y, estimates the PD contribution using:
        b̂_PD(x,y) = 4 * â_{x|y} * ŵ_y * â_{x|ȳ} * (1 - ŵ_y)

    where:
        ŵ_y = marginal approval rate of y among voters who observed y
        â_{x|y} = P(approve x | approve y, observed both x and y)
        â_{x|ȳ} = P(approve x | disapprove y, observed both x and y)

    Final score is the average over all splitters y ≠ x.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes
        observed_mask: (n_items, n_voters) boolean array
        min_approvers: minimum number of approvers required for a valid score

    Returns:
        (n_items,) array of estimated bridging scores in [0, 1]
    """
    n_items, n_voters = observed_matrix.shape

    # observed_mask as float for matrix operations
    obs = observed_mask.astype(np.float64)

    # approve[c, v] = 1 if voter v observed and approved c, else 0
    approve = np.where(observed_mask, observed_matrix, 0.0)

    # Co-observed counts: how many voters observed both c and c'
    co_observed = obs @ obs.T  # (n_items, n_items)

    # n_approve_cprime[c, c'] = voters who observed both c AND approved c'
    n_approve_cprime = obs @ approve.T  # (n_items, n_items)

    # Marginal approval rate of c' among all voters who observed c'
    obs_count = obs.sum(axis=1)  # (n_items,)
    approve_count = approve.sum(axis=1)  # (n_items,)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_hat = approve_count / obs_count
        w_hat = np.nan_to_num(w_hat, nan=0.5)

    # Broadcast w_hat[c'] over the (c, c') grid
    w = w_hat[np.newaxis, :]  # shape (1, n_items), broadcasts to (n_items, n_items)

    # both_approve[c, c'] = voters who observed both AND approved both
    both_approve = approve @ approve.T  # (n_items, n_items)

    # a_1[c, c'] = P(approve c | approve c', observed both)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_1 = both_approve / n_approve_cprime
        a_1 = np.nan_to_num(a_1, nan=0.0)

    # n_disapprove_cprime = voters who observed both AND disapproved c'
    n_disapprove_cprime = co_observed - n_approve_cprime

    # approve_c_disapprove_cprime = voters who observed both, approve c, disapprove c'
    approve_c_and_obs_cprime = approve @ obs.T
    approve_c_disapprove_cprime = approve_c_and_obs_cprime - both_approve

    # a_2[c, c'] = P(approve c | disapprove c', observed both)
    with np.errstate(divide='ignore', invalid='ignore'):
        a_2 = approve_c_disapprove_cprime / n_disapprove_cprime
        a_2 = np.nan_to_num(a_2, nan=0.0)

    # PD term: 4 * a_1 * w * a_2 * (1 - w)
    terms = 4 * a_1 * w * a_2 * (1 - w)

    # Zero out diagonal (can't use x as its own splitter)
    np.fill_diagonal(terms, 0.0)

    # Average over valid splitters (where co_observed > 0)
    valid_pairs = co_observed > 0
    np.fill_diagonal(valid_pairs, False)
    n_valid = valid_pairs.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(n_valid > 0, (terms * valid_pairs).sum(axis=1) / n_valid, 0.0)

    # Zero out scores for items with fewer than min_approvers
    scores = np.where(approve_count >= min_approvers, scores, 0.0)

    return scores


def estimate_pnorm_naive(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    p: float = -10.0,
    min_approvers: int = 5,
) -> np.ndarray:
    """
    Naive p-norm estimation from partially observed data.

    Computes p-norm bridging scores using only observed entries.
    For each pair (c, c'), approval rates a_1 and a_2 are computed
    among voters who observed *both* c and c'.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes (NaN for missing)
        observed_mask: (n_items, n_voters) boolean array
        p: p-norm parameter (default -10 for approx min)
        min_approvers: minimum number of approvers required for a valid score

    Returns:
        (n_items,) array of estimated p-norm bridging scores
    """
    matrix = np.where(observed_mask, observed_matrix, np.nan)
    n_items, n_voters = matrix.shape

    # For each pair (c, c'), we need voters who observed both
    # observed_mask is boolean (n_items, n_voters)
    obs = observed_mask.astype(np.float64)

    # Co-observed counts: how many voters observed both c and c'
    co_observed = obs @ obs.T  # (n_items, n_items)

    # Among co-observed voters, count approvers of c'
    # For voters who observed both c and c', how many approve c'?
    # obs[c] * matrix[c'] gives approve-c' indicator only where both observed
    # But we need: voters who observed c AND observed c' AND approve c'
    approve = np.where(observed_mask, observed_matrix, 0.0)

    # n_approve_cprime_among_coobs[c, c'] = sum over voters of obs[c,v] * obs[c',v] * matrix[c',v]
    n_approve_cprime = obs @ approve.T  # (n_items, n_items)

    # Marginal approval rate of c' among all voters who observed c'
    # (not conditioned on also observing c)
    obs_count = obs.sum(axis=1)  # (n_items,)
    approve_count = approve.sum(axis=1)  # (n_items,)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_hat = approve_count / obs_count
        w_hat = np.nan_to_num(w_hat, nan=0.5)  # default to 0.5 if no observations

    # Broadcast w_hat[c'] over the (c, c') grid
    w = w_hat[np.newaxis, :]  # shape (1, n_items), broadcasts to (n_items, n_items)

    # both_approve[c, c'] = voters who observed both, approve both
    both_approve = approve @ approve.T  # (n_items, n_items)

    # a_1[c, c'] = P(approve c | approve c', observed both)
    #            = both_approve[c,c'] / n_approve_cprime[c',c] ... wait
    # n_approve_cprime[c, c'] = voters who observed c AND approved c'
    # We need: voters who observed both c and c' AND approved c'
    # That's: sum_v obs[c,v] * obs[c',v] * approve[c',v] = n_approve_cprime[c, c']
    # And among those, how many also approve c?
    # = sum_v obs[c,v] * obs[c',v] * approve[c',v] * approve[c,v] = approve[c] * (obs[c'] * approve[c']).T
    # = (approve * (obs * approve).T) ... no, let's think again

    # Voters who observed both c and c' and approve c': mask_both_approve_cprime[v] = obs[c,v] * obs[c',v] * approve[c',v]
    # Among those, approve c: approve[c,v]
    # So a_1[c,c'] = sum_v (approve[c,v] * obs[c,v] * obs[c',v] * approve[c',v]) / sum_v (obs[c,v] * obs[c',v] * approve[c',v])
    # Numerator = sum_v approve[c,v] * approve[c',v] * obs[c,v] * obs[c',v]
    # But approve[c,v] is 0 when not observed (we set it to 0), and approve already has obs baked in
    # Actually approve[c,v] = matrix[c,v] if observed, 0 if not. So approve[c,v]*approve[c',v] counts
    # voters who are observed on both AND approve both. But obs[c,v]*obs[c',v] is redundant when
    # multiplied with approve, since approve is 0 when not observed.
    # So numerator = both_approve[c,c'] and denominator = n_approve_cprime[c,c']
    # But wait: n_approve_cprime[c,c'] = obs[c] @ approve[c'].T = sum_v obs[c,v]*approve[c',v]
    # This counts voters observed on c who approve c'. They might not be observed on c' though!
    # No — approve[c',v] = matrix[c',v] if obs[c',v] else 0. So approve[c',v] > 0 implies obs[c',v].
    # So obs[c,v]*approve[c',v] > 0 implies obs on both c and c'.
    # Therefore n_approve_cprime[c,c'] = voters observed on both who approve c'. Correct.

    with np.errstate(divide='ignore', invalid='ignore'):
        a_1 = both_approve / n_approve_cprime
        a_1 = np.nan_to_num(a_1, nan=0.0)

    # n_disapprove_cprime[c,c'] = co_observed[c,c'] - n_approve_cprime[c,c']
    n_disapprove_cprime = co_observed - n_approve_cprime

    # approve_c_disapprove_cprime[c,c'] = voters co-observed who approve c but disapprove c'
    # = sum_v approve[c,v] * obs[c',v] * (1 - approve[c',v]) where obs on both
    # = sum_v approve[c,v] * obs[c',v] - approve[c,v] * approve[c',v]
    # = (approve @ obs.T)[c,c'] - both_approve[c,c']
    approve_c_and_obs_cprime = approve @ obs.T
    approve_c_disapprove_cprime = approve_c_and_obs_cprime - both_approve

    with np.errstate(divide='ignore', invalid='ignore'):
        a_2 = approve_c_disapprove_cprime / n_disapprove_cprime
        a_2 = np.nan_to_num(a_2, nan=0.0)

    # Compute p-norm terms
    # Broadcast w to full (n_items, n_items) for boolean indexing
    w_cprime = np.broadcast_to(w, (n_items, n_items)).copy()

    if p == float('inf'):
        # Max of a_1 and a_2
        terms = np.maximum(a_1, a_2)
    elif p == float('-inf'):
        # Min of a_1 and a_2
        terms = np.minimum(a_1, a_2)
    elif p == 0:
        # Geometric mean: a_1^w × a_2^(1-w)
        # Handle zeros: if either is 0, term is 0
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(
                (a_1 > 0) & (a_2 > 0),
                np.power(a_1, w_cprime) * np.power(a_2, 1 - w_cprime),
                0.0
            )
    elif p < 0:
        # For negative p, a^p diverges if a=0, so term should be 0 when a_1=0 or a_2=0
        safe_a1 = np.where(a_1 > 0, a_1, 0.0)
        safe_a2 = np.where(a_2 > 0, a_2, 0.0)
        both_positive = (safe_a1 > 0) & (safe_a2 > 0)
        terms = np.zeros((n_items, n_items))
        terms[both_positive] = (
            w_cprime[both_positive] * safe_a1[both_positive] ** p
            + (1 - w_cprime[both_positive]) * safe_a2[both_positive] ** p
        ) ** (1.0 / p)
    else:
        terms = (w_cprime * a_1 ** p + (1 - w_cprime) * a_2 ** p) ** (1.0 / p)
        terms = np.nan_to_num(terms, nan=0.0)

    # Zero out diagonal
    np.fill_diagonal(terms, 0.0)

    # Average over c' != c, but only where co_observed > 0
    valid_pairs = co_observed > 0
    np.fill_diagonal(valid_pairs, False)
    n_valid = valid_pairs.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(n_valid > 0, (terms * valid_pairs).sum(axis=1) / n_valid, 0.0)

    # Zero out scores for items with fewer than min_approvers
    approve_count = approve.sum(axis=1)
    scores = np.where(approve_count >= min_approvers, scores, 0.0)

    return scores


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
            # Disagreement: only 1.0 vs 0.0, not skips
            disagree = (
                ((votes_i == 1.0) & (votes_j == 0.0)) |
                ((votes_i == 0.0) & (votes_j == 1.0))
            ).astype(float)

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
            # Only count 1.0 vs 0.0 as disagreement (skips don't disagree with anything)
            disagree_count = (
                ((votes_i == 1.0) & (votes_j == 0.0)) |
                ((votes_i == 0.0) & (votes_j == 1.0))
            ).sum()

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
            # Only count 1.0 vs 0.0 as disagreement (skips don't disagree with anything)
            disagree = (
                ((votes_i == 1.0) & (votes_j == 0.0)) |
                ((votes_i == 0.0) & (votes_j == 1.0))
            ).astype(float)

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
