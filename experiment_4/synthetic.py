"""
Generate synthetic election matrices with known group structures.

The synthetic data creates voters who belong to distinct groups based on their
approval of "base" comments. This allows us to study bridging scores analytically
where the pairwise disagreement structure is known.

Key idea:
- Group 1 voters approve even-indexed base comments (0, 2, 4, ...)
- Group 2 voters approve odd-indexed base comments (1, 3, 5, ...)
- This ensures d_ij ≈ 1 between groups, d_ij ≈ 0 within groups
"""

from typing import List, Optional

import numpy as np


def assign_voters_to_groups(
    n_voters: int,
    group_sizes: List[float],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Assign voters to groups based on specified size fractions.

    Args:
        n_voters: Total number of voters
        group_sizes: List of fractions summing to 1, one per group
        seed: Random seed for reproducibility

    Returns:
        (n_voters,) array of group indices (0, 1, 2, ...)
    """
    rng = np.random.default_rng(seed)

    # Validate group sizes
    assert abs(sum(group_sizes) - 1.0) < 1e-6, "Group sizes must sum to 1"

    # Compute cumulative counts
    cumsum = np.cumsum(group_sizes)
    boundaries = (cumsum * n_voters).astype(int)
    boundaries[-1] = n_voters  # Ensure we include all voters

    # Assign groups
    assignments = np.zeros(n_voters, dtype=int)
    start = 0
    for group_idx, end in enumerate(boundaries):
        assignments[start:end] = group_idx
        start = end

    # Shuffle to randomize positions
    rng.shuffle(assignments)

    return assignments


def generate_base_comments_matrix(
    n_voters: int,
    n_base_comments: int,
    group_assignments: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """
    Generate base comments that define group membership.

    For k groups:
    - Comments with index i mod k are approved by group i voters only

    Args:
        n_voters: Number of voters
        n_base_comments: Number of base comments
        group_assignments: (n_voters,) array of group indices
        n_groups: Number of groups

    Returns:
        (n_base_comments, n_voters) boolean matrix
    """
    matrix = np.zeros((n_base_comments, n_voters), dtype=bool)

    for comment_idx in range(n_base_comments):
        # Comment i is approved by group (i mod n_groups)
        approving_group = comment_idx % n_groups
        matrix[comment_idx] = (group_assignments == approving_group)

    return matrix


def generate_focal_comment(
    n_voters: int,
    group_assignments: np.ndarray,
    focal_approval_rates: List[float],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a focal comment with specified approval rates per group.

    Args:
        n_voters: Number of voters
        group_assignments: (n_voters,) array of group indices
        focal_approval_rates: List of approval rates, one per group
        seed: Random seed for reproducibility

    Returns:
        (n_voters,) boolean array of approvals
    """
    rng = np.random.default_rng(seed)

    approvals = np.zeros(n_voters, dtype=bool)
    n_groups = len(focal_approval_rates)

    for group_idx in range(n_groups):
        # Find voters in this group
        group_mask = (group_assignments == group_idx)
        group_size = group_mask.sum()

        if group_size > 0:
            # Each voter in the group approves with probability a_i
            approval_rate = focal_approval_rates[group_idx]
            approvals[group_mask] = rng.random(group_size) < approval_rate

    return approvals


def generate_synthetic_matrix(
    n_voters: int,
    n_base_comments: int,
    group_sizes: List[float],
    focal_approval_rates: List[float],
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate complete synthetic election matrix.

    Creates a matrix with base comments defining group structure and a focal
    comment with specified approval rates from each group.

    Args:
        n_voters: Number of voters
        n_base_comments: Number of base comments defining group structure
        group_sizes: Fractions of voters in each group (must sum to 1)
        focal_approval_rates: Approval rate for focal comment from each group

    Returns:
        (n_base_comments + 1, n_voters) matrix where:
        - Rows 0 to n_base_comments-1 are base comments
        - Last row is the focal comment
    """
    n_groups = len(group_sizes)
    assert len(focal_approval_rates) == n_groups, \
        "Must have one approval rate per group"

    # Assign voters to groups
    group_assignments = assign_voters_to_groups(n_voters, group_sizes, seed=seed)

    # Generate base comments
    base_matrix = generate_base_comments_matrix(
        n_voters, n_base_comments, group_assignments, n_groups
    )

    # Generate focal comment (use different seed to ensure independence)
    focal_seed = seed + 1000 if seed is not None else None
    focal_comment = generate_focal_comment(
        n_voters, group_assignments, focal_approval_rates, seed=focal_seed
    )

    # Combine: base comments + focal comment
    matrix = np.vstack([base_matrix, focal_comment.reshape(1, -1)])

    return matrix.astype(float)


def compute_theoretical_within_group_disagreement(n_base_comments: int, n_groups: int) -> float:
    """
    Compute theoretical d_ij for voters in the same group.

    Within a group, voters agree on all base comments.
    d_ij = 0 for voters in the same group (ignoring focal comment).

    Returns:
        Expected d_ij for same-group voters
    """
    # Within same group: agree on all base comments
    return 0.0


def compute_theoretical_between_group_disagreement(n_base_comments: int, n_groups: int) -> float:
    """
    Compute theoretical d_ij for voters in different groups.

    Between groups i and j:
    - Agree on comments belonging to groups other than i and j
    - Disagree on comments belonging to groups i and j

    For 2 groups: disagree on all comments, so d_ij = 1
    For k groups: disagree on (2/k) fraction of comments

    Returns:
        Expected d_ij for different-group voters
    """
    # For k groups, each pair of groups disagrees on 2/k of comments
    # (the comments belonging to either group)
    return 2.0 / n_groups


def compute_focal_bridging_score_fast(
    n_voters: int,
    n_base_comments: int,
    group_assignments: np.ndarray,
    focal_approvals: np.ndarray,
    n_groups: int,
) -> float:
    """
    Compute bridging score for focal comment using known group structure.

    This is much faster than the general algorithm because:
    - Within-group d_ij = 0 (voters agree on all base comments)
    - Between-group d_ij = n_base / (n_base + 1) for 2 groups
      (disagree on all base comments, agree on focal since both approve)

    For k groups:
    - d_ij between groups g1 and g2 = 2/k * n_base / (n_base + 1)
      (disagree on 2/k of base comments)

    Args:
        n_voters: Total number of voters
        n_base_comments: Number of base comments
        group_assignments: (n_voters,) array of group indices
        focal_approvals: (n_voters,) boolean array of focal comment approvals
        n_groups: Number of groups

    Returns:
        Bridging score in [0, 1]
    """
    # Get approvers
    approver_indices = np.where(focal_approvals)[0]
    n_approvers = len(approver_indices)

    if n_approvers < 2:
        return 0.0

    # Get group assignments for approvers
    approver_groups = group_assignments[approver_indices]

    # Count approvers per group
    group_counts = np.bincount(approver_groups, minlength=n_groups)

    # Compute d_ij for approver pairs
    # - Same group: d_ij = 0
    # - Different groups: d_ij = (2/k) * n_base / (n_base + 1)
    #   because they disagree on (2/k) fraction of base comments

    total_comments = n_base_comments + 1
    d_between = (2.0 / n_groups) * n_base_comments / total_comments

    # Count cross-group pairs
    total_cross_pairs = 0
    for g1 in range(n_groups):
        for g2 in range(g1 + 1, n_groups):
            total_cross_pairs += group_counts[g1] * group_counts[g2]

    # Sum of d_ij over all approver pairs
    total_d = total_cross_pairs * d_between

    # Bridging score: (4/n²) × Σ d_ij
    normalization = 4.0 / (n_voters ** 2)
    bridging_score = normalization * total_d

    return bridging_score


def generate_and_compute_bridging_fast(
    n_voters: int,
    n_base_comments: int,
    group_sizes: List[float],
    focal_approval_rates: List[float],
    seed: Optional[int] = None,
) -> float:
    """
    Generate synthetic election and compute focal comment bridging score efficiently.

    Uses the known group structure to compute bridging analytically without
    generating the full base comment matrix.

    Args:
        n_voters: Number of voters
        n_base_comments: Number of base comments (for disagreement calculation)
        group_sizes: Fractions of voters in each group
        focal_approval_rates: Approval rate for focal comment from each group
        seed: Random seed

    Returns:
        Bridging score for the focal comment
    """
    n_groups = len(group_sizes)
    assert len(focal_approval_rates) == n_groups

    # Assign voters to groups
    group_assignments = assign_voters_to_groups(n_voters, group_sizes, seed=seed)

    # Generate focal comment approvals
    focal_seed = seed + 1000 if seed is not None else None
    focal_approvals = generate_focal_comment(
        n_voters, group_assignments, focal_approval_rates, seed=focal_seed
    )

    # Compute bridging score using fast method
    bridging_score = compute_focal_bridging_score_fast(
        n_voters, n_base_comments, group_assignments, focal_approvals, n_groups
    )

    return bridging_score
