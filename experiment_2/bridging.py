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

    d_ij = (1/|C|) × Σ_{c'∈C} 1[v_{i,c'} ≠ v_{j,c'}]

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0
                (no NaN values allowed for ground truth computation)

    Returns:
        (n_voters, n_voters) symmetric matrix where entry [i,j] is d_ij
    """
    n_items, n_voters = matrix.shape

    # Transpose to (n_voters, n_items) for easier pairwise comparison
    votes = matrix.T  # (n_voters, n_items)

    # Compute disagreement using broadcasting
    # votes[:, None, :] has shape (n_voters, 1, n_items)
    # votes[None, :, :] has shape (1, n_voters, n_items)
    # Disagreement when votes differ
    disagreements = (votes[:, None, :] != votes[None, :, :]).sum(axis=2)

    # Normalize by number of comments
    d_matrix = disagreements.astype(float) / n_items

    return d_matrix


def compute_bridging_scores(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Pairwise Disagreement bridging score for each comment.

    For comment c with approvers N_c:
        b^PD(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij

    Efficient O(m × n²) computation using:
        Σ d_ij = (1/|C|) × Σ_{c'∈C} |N_c ∩ N_{c'}| × |N_c ∩ (V \ N_{c'})|

    where |N_c ∩ N_{c'}| counts approvers of c who also approve c',
    and |N_c ∩ (V \ N_{c'})| counts approvers of c who disapprove c'.

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0

    Returns:
        (n_items,) array of bridging scores in [0, 1]
    """
    n_items, n_voters = matrix.shape

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # Convert to boolean for easier computation
    approves = matrix.astype(bool)  # (n_items, n_voters)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Approvers of comment c
        approvers_c = approves[c]  # (n_voters,) boolean

        # For each other comment c', count:
        # - approvers of c who also approve c'
        # - approvers of c who disapprove c'
        total_disagreement = 0.0

        for cp in range(n_items):
            # Approvers of c who approve c'
            approve_both = (approvers_c & approves[cp]).sum()
            # Approvers of c who disapprove c'
            approve_c_disapprove_cp = (approvers_c & ~approves[cp]).sum()

            # This counts pairs (i,j) in N_c where i approves c' and j disapproves c'
            total_disagreement += approve_both * approve_c_disapprove_cp

        # Normalize by number of comments and apply 4/n² factor
        bridging_scores[c] = normalization * total_disagreement / n_items

    return bridging_scores


def compute_bridging_scores_vectorized(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized version of compute_bridging_scores for better performance.

    Uses matrix operations instead of explicit loops.

    Args:
        matrix: (n_items, n_voters) array with values 0.0 or 1.0

    Returns:
        (n_items,) array of bridging scores in [0, 1]
    """
    n_items, n_voters = matrix.shape

    # Normalization constant: 4/n²
    normalization = 4.0 / (n_voters ** 2)

    # Convert to boolean
    approves = matrix.astype(bool)  # (n_items, n_voters)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        approvers_c = approves[c]  # (n_voters,)

        # For all comments c', compute intersection sizes with N_c
        # approves has shape (n_items, n_voters)
        # approvers_c has shape (n_voters,)

        # Number of approvers of c who also approve each c'
        approve_both = (approves & approvers_c).sum(axis=1)  # (n_items,)

        # Number of approvers of c who disapprove each c'
        approve_c_disapprove_cp = (~approves & approvers_c).sum(axis=1)  # (n_items,)

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
    approves = matrix.astype(bool)

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
