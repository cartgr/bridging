"""
Polis Group-Informed Consensus implementation.

Based on the Polis source code and documentation:
- https://compdemocracy.org/group-informed-consensus/
- https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj

Pipeline:
1. Imputation: Replace missing votes with per-comment (column) mean
2. PCA: Compute voter projections (2 components) using power iteration
3. K-means clustering: Cluster voters with silhouette-based k selection
4. Group-Informed Consensus: Product of per-group approval rates (Laplace smoothed)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_2.priority import power_iteration_pca


def filter_participants_by_votes(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    min_votes: int = 7,
    min_participants: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter participants based on vote count, matching Polis behavior.

    Polis requires participants to have voted on at least min(7, n_comments) comments.
    If fewer than 15 participants meet this criterion, greedily add top contributors.

    Args:
        matrix: (n_items, n_voters) array with vote values
        observed_mask: (n_items, n_voters) boolean array
        min_votes: minimum votes required (default: 7, matching Polis)
        min_participants: minimum participants to keep (default: 15)

    Returns:
        Tuple of:
        - filtered_matrix: (n_items, n_kept_voters) array
        - filtered_mask: (n_items, n_kept_voters) boolean array
        - kept_indices: (n_kept_voters,) indices of kept voters in original
    """
    n_items, n_voters = matrix.shape

    # Count votes per participant
    votes_per_participant = observed_mask.sum(axis=0)  # (n_voters,)

    # Threshold: min(min_votes, n_items) - if few comments, require all
    threshold = min(min_votes, n_items)

    # Find participants meeting threshold
    meets_threshold = votes_per_participant >= threshold
    n_meeting = meets_threshold.sum()

    if n_meeting >= min_participants:
        # Enough participants meet threshold
        kept_indices = np.where(meets_threshold)[0]
    else:
        # Not enough - greedily add top contributors
        # Sort by vote count descending
        sorted_indices = np.argsort(-votes_per_participant)

        # Take top min_participants (or all if fewer voters)
        n_to_keep = min(min_participants, n_voters)
        kept_indices = sorted_indices[:n_to_keep]

    # Filter matrix and mask
    filtered_matrix = matrix[:, kept_indices]
    filtered_mask = observed_mask[:, kept_indices]

    return filtered_matrix, filtered_mask, kept_indices


def impute_column_mean(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """
    Replace missing values with per-comment (column in voter-centric view) mean.

    In our data format (n_items, n_voters), comments are rows and voters are columns.
    Polis imputes missing votes with the mean of observed votes for that comment.

    Args:
        matrix: (n_items, n_voters) array with vote values
        observed_mask: (n_items, n_voters) boolean array, True where observed

    Returns:
        (n_items, n_voters) array with missing values imputed
    """
    n_items, n_voters = matrix.shape
    imputed = matrix.copy()

    # Compute per-comment (row) mean from observed values
    observed_counts = observed_mask.sum(axis=1, keepdims=True)  # (n_items, 1)
    observed_counts_safe = np.maximum(observed_counts, 1)  # Avoid division by zero

    # Sum of observed values per comment
    safe_matrix = np.where(observed_mask, matrix, 0.0)
    observed_sums = safe_matrix.sum(axis=1, keepdims=True)  # (n_items, 1)

    # Per-comment mean
    comment_means = observed_sums / observed_counts_safe
    # Default to 0.5 for comments with no observations
    comment_means = np.where(observed_counts > 0, comment_means, 0.5)

    # Broadcast means to all voters and fill missing
    imputed = np.where(observed_mask, matrix, comment_means)

    return imputed


def compute_voter_pca_projections(
    imputed_matrix: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Compute PCA projections for voters using power iteration.

    Args:
        imputed_matrix: (n_items, n_voters) array with no missing values
        n_components: number of PCA components (default: 2)

    Returns:
        (n_voters, n_components) array of voter projections
    """
    # Transpose to (n_voters, n_items) - voters as samples, comments as features
    data = imputed_matrix.T  # (n_voters, n_items)
    n_voters, n_items = data.shape

    # Center the data (mean-center each comment/feature)
    data_centered = data - data.mean(axis=0)

    # Handle edge cases
    if n_voters < n_components or n_items < n_components:
        return np.zeros((n_voters, n_components))

    if np.allclose(data_centered, 0):
        return np.zeros((n_voters, n_components))

    # Compute principal components using power iteration
    n_comp = min(n_components, min(n_voters, n_items))
    components = power_iteration_pca(data_centered, n_comp)  # (n_components, n_items)

    # Project voters onto principal components
    # projections[i, k] = data_centered[i, :] @ components[k, :]
    projections = data_centered @ components.T  # (n_voters, n_components)

    return projections


def cluster_voters_kmeans(
    projections: np.ndarray,
    max_k: int = 5,
    min_k: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    K-means clustering of voters with silhouette-based k selection.

    Following Polis:
    - Try k from min_k to min(max_k, 2 + n_voters/12)
    - Select k with highest silhouette score

    Args:
        projections: (n_voters, n_components) voter PCA projections
        max_k: maximum k to try (default: 5)
        min_k: minimum k to try (default: 2)
        seed: random seed for reproducibility

    Returns:
        Tuple of:
        - labels: (n_voters,) array of cluster assignments
        - best_k: selected number of clusters
    """
    n_voters = projections.shape[0]

    # Determine max k based on Polis formula
    polis_max_k = min(max_k, 2 + n_voters // 12)
    actual_max_k = max(min_k, min(polis_max_k, n_voters - 1))

    # Edge case: too few voters for clustering
    if n_voters < 3:
        return np.zeros(n_voters, dtype=int), 1

    if actual_max_k < min_k:
        # Can't cluster, assign all to one group
        return np.zeros(n_voters, dtype=int), 1

    best_score = -1
    best_k = min_k
    best_labels = None

    for k in range(min_k, actual_max_k + 1):
        if k >= n_voters:
            continue

        kmeans = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,  # Multiple restarts for stability
        )
        labels = kmeans.fit_predict(projections)

        # Compute silhouette score (requires at least 2 clusters with >1 sample each)
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2 or np.any(counts < 2):
            # Invalid clustering
            continue

        try:
            score = silhouette_score(projections, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels.copy()
        except ValueError:
            # Silhouette computation can fail in edge cases
            continue

    if best_labels is None:
        # Fallback: all in one cluster
        return np.zeros(n_voters, dtype=int), 1

    return best_labels, best_k


def compute_group_informed_consensus(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    cluster_labels: np.ndarray,
) -> np.ndarray:
    """
    Compute Group-Informed Consensus scores for each comment.

    For comment c and group g:
        P_g(c) = (A_g + 1) / (S_g + 2)   # Laplace-smoothed approval rate

    consensus(c) = product over all groups of P_g(c)

    Uses log-sum-exp for numerical stability.

    Args:
        matrix: (n_items, n_voters) array with vote values
        observed_mask: (n_items, n_voters) boolean array
        cluster_labels: (n_voters,) array of cluster assignments

    Returns:
        (n_items,) array of consensus scores in [0, 1]
    """
    n_items, n_voters = matrix.shape
    unique_labels = np.unique(cluster_labels)
    n_groups = len(unique_labels)

    if n_groups == 0:
        return np.zeros(n_items)

    # Compute per-group approval rates for each comment
    log_consensus = np.zeros(n_items)

    for c in range(n_items):
        log_p_product = 0.0

        for g in unique_labels:
            # Voters in group g
            group_mask = cluster_labels == g

            # Votes on comment c from group g (where observed)
            group_observed = observed_mask[c, :] & group_mask
            n_seen = group_observed.sum()  # S_g

            # Agrees on comment c from group g
            group_agrees = group_observed & (matrix[c, :] == 1.0)
            n_agrees = group_agrees.sum()  # A_g

            # Laplace-smoothed approval rate
            # P_g(c) = (A_g + 1) / (S_g + 2)
            p_g = (n_agrees + 1) / (n_seen + 2)

            # Accumulate log probability for numerical stability
            log_p_product += np.log(p_g)

        # Convert back from log space
        # Clip to avoid underflow
        log_consensus[c] = log_p_product

    # Convert log scores to regular scores
    # Normalize so that max possible score (all P_g = 1) = 1
    # Max log score = n_groups * log(1) = 0, so exp(0) = 1
    consensus_scores = np.exp(log_consensus)

    return consensus_scores


def polis_consensus_pipeline(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    n_pca_components: int = 2,
    max_k: int = 5,
    seed: int = 42,
    filter_participants: bool = True,
    min_votes: int = 7,
    min_participants: int = 15,
) -> Tuple[np.ndarray, Dict]:
    """
    Full Polis Group-Informed Consensus pipeline.

    Pipeline steps:
    0. (Optional) Filter participants by vote count (matching Polis production)
    1. Impute missing values with per-comment mean
    2. Compute voter PCA projections
    3. Cluster voters using k-means with silhouette selection
    4. Compute group-informed consensus scores

    Args:
        matrix: (n_items, n_voters) array with vote values (may contain NaN)
        observed_mask: (n_items, n_voters) boolean array
        n_pca_components: number of PCA components (default: 2)
        max_k: maximum clusters to try (default: 5)
        seed: random seed for reproducibility
        filter_participants: whether to filter low-activity participants (default: True)
        min_votes: minimum votes for participant inclusion (default: 7)
        min_participants: minimum participants to keep (default: 15)

    Returns:
        Tuple of:
        - consensus_scores: (n_items,) array of scores in [0, 1]
        - metadata: dict with pipeline details (k, n_groups, etc.)
    """
    n_items, n_voters_original = matrix.shape

    # Step 0: Filter participants (matching Polis production behavior)
    if filter_participants:
        filtered_matrix, filtered_mask, kept_indices = filter_participants_by_votes(
            matrix, observed_mask, min_votes=min_votes, min_participants=min_participants
        )
        n_voters_filtered = filtered_matrix.shape[1]
    else:
        filtered_matrix = matrix
        filtered_mask = observed_mask
        kept_indices = np.arange(n_voters_original)
        n_voters_filtered = n_voters_original

    # Step 1: Impute missing values
    imputed = impute_column_mean(filtered_matrix, filtered_mask)

    # Step 2: PCA projections
    projections = compute_voter_pca_projections(imputed, n_pca_components)

    # Step 3: Cluster voters
    cluster_labels, best_k = cluster_voters_kmeans(
        projections, max_k=max_k, seed=seed
    )

    # Step 4: Compute consensus (on filtered data)
    consensus_scores = compute_group_informed_consensus(
        filtered_matrix, filtered_mask, cluster_labels
    )

    # Metadata
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    metadata = {
        "n_items": n_items,
        "n_voters_original": n_voters_original,
        "n_voters_filtered": n_voters_filtered,
        "participants_filtered": filter_participants,
        "n_pca_components": n_pca_components,
        "k_clusters": best_k,
        "cluster_sizes": counts.tolist(),
        "observation_rate": filtered_mask.sum() / filtered_mask.size,
    }

    return consensus_scores, metadata
