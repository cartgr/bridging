"""
Ground truth bridging score computation.

Pairwise Disagreement Bridging score measures how much a comment bridges
across disagreeing groups.

For comment c with approvers N_c:
    b^PD(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij

Where:
- d_ij = (1/|C|) × Σ_{c'∈C} 1[c' ∈ (A_i \ A_j) ∪ (A_j \ A_i)]
       = fraction of comments on which voters i and j disagree
- n = total number of voters
- 4/n² is a normalization constant so that b^PD ∈ [0, 1]
"""

import numpy as np


def compute_pairwise_disagreement(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise disagreement d_ij for all voter pairs.

    d_ij = (# comments where both voted 1.0/0.0 and disagree) / (# comments both saw)

    Skips (0.5) are treated as "observed but no opinion":
    - Numerator: Only counts 1.0 vs 0.0 as disagreement
    - Denominator: All non-NaN observations (includes skips)

    Args:
        matrix: (n_items, n_voters) array with values 0.0, 0.5, 1.0, or NaN

    Returns:
        (n_voters, n_voters) symmetric matrix where entry [i,j] is d_ij
    """
    n_items, n_voters = matrix.shape

    # Transpose to (n_voters, n_items) for easier pairwise comparison
    votes = matrix.T  # (n_voters, n_items)

    # Masks for real votes (not skips, not NaN)
    is_approve = (votes == 1.0)   # (n_voters, n_items)
    is_disapprove = (votes == 0.0)  # (n_voters, n_items)
    is_observed = ~np.isnan(votes)  # (n_voters, n_items) - includes skips

    # Disagreement: one approves (1.0) and other disapproves (0.0)
    # Using broadcasting: (n_voters, 1, n_items) vs (1, n_voters, n_items)
    disagree = (
        (is_approve[:, None, :] & is_disapprove[None, :, :]) |
        (is_disapprove[:, None, :] & is_approve[None, :, :])
    )
    disagreements = disagree.sum(axis=2)  # (n_voters, n_voters)

    # Denominator: number of comments both voters observed (includes skips)
    both_observed = is_observed[:, None, :] & is_observed[None, :, :]
    n_both_observed = both_observed.sum(axis=2)  # (n_voters, n_voters)

    # Avoid division by zero
    n_both_observed = np.maximum(n_both_observed, 1)

    d_matrix = disagreements.astype(float) / n_both_observed

    return d_matrix


def compute_bridging_scores(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Pairwise Disagreement bridging score for each comment.

    For comment c with approvers N_c:
        b^PD(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij

    Handles skips (0.5) properly by delegating to compute_bridging_scores_vectorized.

    Args:
        matrix: (n_items, n_voters) array with values 0.0, 0.5, 1.0, or NaN

    Returns:
        (n_items,) array of bridging scores in [0, 1]
    """
    # Delegate to vectorized version which handles skips properly
    return compute_bridging_scores_vectorized(matrix)


def compute_bridging_scores_vectorized(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized version of compute_bridging_scores for better performance.

    Handles skips (0.5) properly:
    - Skips don't count as approvals or disapprovals
    - Disagreement only when one voter approves (1.0) and other disapproves (0.0)
    - Denominator includes all co-observed items (including skips)

    Args:
        matrix: (n_items, n_voters) array with values 0.0, 0.5, 1.0, or NaN

    Returns:
        (n_items,) array of bridging scores in [0, 1]
    """
    n_items, n_voters = matrix.shape

    # Check if we have skips or NaNs - if so, use disagreement matrix approach
    has_skips = np.any(matrix == 0.5)
    has_nans = np.any(np.isnan(matrix))

    if has_skips or has_nans:
        # Use disagreement matrix approach for proper skip handling
        d_matrix = compute_pairwise_disagreement(matrix)
        return compute_bridging_scores_from_disagreement(matrix, d_matrix)

    # Fast path for fully observed binary data (no skips, no NaNs)
    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # Convert to boolean
    approves = (matrix == 1.0)  # (n_items, n_voters)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        approvers_c = approves[c]  # (n_voters,)

        # For all comments c', compute intersection sizes with N_c
        # Number of approvers of c who also approve each c'
        approve_both = (approves & approvers_c).sum(axis=1)  # (n_items,)

        # Number of approvers of c who disapprove each c' (only 0.0, not skips)
        disapproves = (matrix == 0.0)
        approve_c_disapprove_cp = (disapproves & approvers_c).sum(axis=1)  # (n_items,)

        # Sum of products
        total_disagreement = (approve_both * approve_c_disapprove_cp).sum()

        bridging_scores[c] = normalization * total_disagreement / n_items

    return bridging_scores


def compute_bridging_scores_from_disagreement(
    matrix: np.ndarray, d_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute bridging scores using pre-computed pairwise disagreement.

    b^PD(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0
        d_matrix: (n_voters, n_voters) pairwise disagreement matrix

    Returns:
        (n_items,) array of bridging scores in [0, 1]
    """
    n_items, n_voters = matrix.shape
    # Use explicit == 1.0 check so 0.5 (pass) values are not treated as approvals
    approves = (matrix == 1.0)

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Get indices of approvers
        approver_indices = np.where(approves[c])[0]
        n_approvers = len(approver_indices)

        if n_approvers < 2:
            # Need at least 2 approvers for a pair
            bridging_scores[c] = 0.0
            continue

        # Sum d_ij for all pairs i < j in N_c
        total = 0.0
        for i_idx in range(n_approvers):
            for j_idx in range(i_idx + 1, n_approvers):
                i = approver_indices[i_idx]
                j = approver_indices[j_idx]
                total += d_matrix[i, j]

        bridging_scores[c] = normalization * total

    return bridging_scores


def compute_bridging_pnorm(matrix: np.ndarray, p: float = 2.0) -> np.ndarray:
    """
    Compute p-norm bridging score for each comment (vectorized).

    b_p(c) = (1/|C-1|) × Σ_{c' ≠ c} (w_{c'} × a_1^p + (1-w_{c'}) × a_2^p)^(1/p)

    Where:
    - w_{c'} = proportion of all voters who approve c'
    - a_1 = approval proportion of c among approvers of c'
    - a_2 = approval proportion of c among disapprovers of c'

    This measures how consistently c is approved across different "slices"
    of the population defined by their opinion on other comments.

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0
        p: the p-norm parameter (default: 2.0)

    Returns:
        (n_items,) array of p-norm bridging scores
    """
    n_items, n_voters = matrix.shape

    # Precompute approval counts for all items
    # w[c'] = approval rate of c'
    w = matrix.sum(axis=1) / n_voters  # (n_items,)

    # Precompute: for each (c, c') pair, compute a_1 and a_2
    # a_1[c, c'] = approval of c among approvers of c'
    # a_2[c, c'] = approval of c among disapprovers of c'

    # Number of approvers/disapprovers per item
    n_approvers = matrix.sum(axis=1)  # (n_items,)
    n_disapprovers = n_voters - n_approvers

    # matrix[c] @ matrix[c'].T = number who approve both c and c'
    # This gives us a (n_items, n_items) matrix
    both_approve = matrix @ matrix.T  # (n_items, n_items)

    # a_1[c, c'] = both_approve[c, c'] / n_approvers[c']
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        a_1 = both_approve / n_approvers[np.newaxis, :]  # (n_items, n_items)
        a_1 = np.nan_to_num(a_1, nan=0.0, posinf=0.0, neginf=0.0)

    # Approvers of c who disapprove c' = approvers of c - both_approve
    approve_c_only = matrix.sum(axis=1)[:, np.newaxis] - both_approve
    with np.errstate(divide='ignore', invalid='ignore'):
        a_2 = approve_c_only / n_disapprovers[np.newaxis, :]
        a_2 = np.nan_to_num(a_2, nan=0.0, posinf=0.0, neginf=0.0)

    # Now compute p-norm terms for all (c, c') pairs
    w_row = w[np.newaxis, :]  # (1, n_items) for broadcasting

    if p == float('inf'):
        terms = np.maximum(a_1, a_2)
    elif p == float('-inf'):
        terms = np.minimum(a_1, a_2)
    elif p == 0:
        # Geometric mean: a_1^w × a_2^(1-w)
        # Handle zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(
                (a_1 > 0) & (a_2 > 0),
                np.power(a_1, w_row) * np.power(a_2, 1 - w_row),
                0.0
            )
    elif p < 0:
        # For negative p, zeros cause issues
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(
                (a_1 > 0) & (a_2 > 0),
                np.power(w_row * np.power(a_1, p) + (1 - w_row) * np.power(a_2, p), 1/p),
                0.0
            )
    else:
        terms = np.power(w_row * np.power(a_1, p) + (1 - w_row) * np.power(a_2, p), 1/p)

    # Zero out diagonal (c' = c)
    np.fill_diagonal(terms, 0.0)

    # Average over all other comments
    scores = terms.sum(axis=1) / (n_items - 1) if n_items > 1 else np.zeros(n_items)

    return scores


def compute_pairwise_voting_scores(matrix: np.ndarray) -> dict:
    """
    Compute Vitalik-style pairwise voting scores.

    For each pair of voters (i, j):
    - They get 1 vote total
    - Split evenly among items where they agree (both approve or both disapprove)
    - Agreement on approval contributes to approval_score
    - Agreement on disapproval contributes to disapproval_score

    Items with high approval_scores are "bridging" - they get support from
    voters who rarely agree on other things.

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0

    Returns:
        Dict with:
        - approval_scores: (n_items,) total weighted approval
        - disapproval_scores: (n_items,) total weighted disapproval
        - net_scores: approval - disapproval
    """
    n_items, n_voters = matrix.shape

    # Convert to boolean
    # Use explicit == 1.0 check so 0.5 (pass) values are not treated as approvals
    approves = (matrix == 1.0)  # (n_items, n_voters)
    disapproves = (matrix == 0.0)  # (n_items, n_voters)

    # Initialize scores
    approval_scores = np.zeros(n_items)
    disapproval_scores = np.zeros(n_items)

    # Iterate over all pairs of voters
    for i in range(n_voters):
        for j in range(i + 1, n_voters):
            # Find agreements
            both_approve = approves[:, i] & approves[:, j]
            both_disapprove = disapproves[:, i] & disapproves[:, j]

            # Total agreements for this pair
            n_agreements = both_approve.sum() + both_disapprove.sum()

            if n_agreements == 0:
                continue

            # Weight per agreement
            weight = 1.0 / n_agreements

            # Add to scores
            approval_scores[both_approve] += weight
            disapproval_scores[both_disapprove] += weight

    return {
        "approval_scores": approval_scores,
        "disapproval_scores": disapproval_scores,
        "net_scores": approval_scores - disapproval_scores,
    }


def compute_bridging_harmonic_pd(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Harmonic Pairwise Disagreement bridging score.

    b(a_1, a_2; w) = 2w(1-w)a_1*a_2 / (w*a_1 + (1-w)*a_2)

    This is the harmonic mean of (w*a_1) and ((1-w)*a_2), normalized by approval.

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0

    Returns:
        (n_items,) array of harmonic PD bridging scores
    """
    n_items, n_voters = matrix.shape

    # w[c'] = approval rate of c'
    w = matrix.sum(axis=1) / n_voters  # (n_items,)

    # Precompute overlap counts
    both_approve = matrix @ matrix.T  # (n_items, n_items)
    n_approvers = matrix.sum(axis=1)  # (n_items,)
    n_disapprovers = n_voters - n_approvers

    # a_1[c, c'] = approval of c among approvers of c'
    with np.errstate(divide='ignore', invalid='ignore'):
        a_1 = both_approve / n_approvers[np.newaxis, :]
        a_1 = np.nan_to_num(a_1, nan=0.0)

    # a_2[c, c'] = approval of c among disapprovers of c'
    approve_c_only = matrix.sum(axis=1)[:, np.newaxis] - both_approve
    with np.errstate(divide='ignore', invalid='ignore'):
        a_2 = approve_c_only / n_disapprovers[np.newaxis, :]
        a_2 = np.nan_to_num(a_2, nan=0.0)

    w_row = w[np.newaxis, :]  # (1, n_items)

    # Harmonic PD: 2w(1-w)a_1*a_2 / (w*a_1 + (1-w)*a_2)
    numerator = 2 * w_row * (1 - w_row) * a_1 * a_2
    denominator = w_row * a_1 + (1 - w_row) * a_2

    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(denominator > 0, numerator / denominator, 0.0)

    # Zero out diagonal
    np.fill_diagonal(terms, 0.0)

    # Average over all other comments
    scores = terms.sum(axis=1) / (n_items - 1) if n_items > 1 else np.zeros(n_items)

    return scores
