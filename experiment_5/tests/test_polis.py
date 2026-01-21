"""Tests for Polis Group-Informed Consensus implementation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiment_5.polis import (
    impute_column_mean,
    compute_voter_pca_projections,
    cluster_voters_kmeans,
    compute_group_informed_consensus,
    polis_consensus_pipeline,
)


class TestImputeColumnMean:
    """Tests for impute_column_mean function."""

    def test_fully_observed(self):
        """Fully observed matrix should be unchanged."""
        matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        mask = np.ones_like(matrix, dtype=bool)

        imputed = impute_column_mean(matrix, mask)

        np.testing.assert_array_equal(imputed, matrix)

    def test_missing_values_imputed(self):
        """Missing values should be replaced with row mean."""
        matrix = np.array([[1.0, 0.0, np.nan], [0.0, np.nan, 1.0]])
        mask = ~np.isnan(matrix)

        imputed = impute_column_mean(matrix, mask)

        # Row 0: mean of [1, 0] = 0.5, so imputed[0, 2] = 0.5
        assert imputed[0, 2] == 0.5
        # Row 1: mean of [0, 1] = 0.5, so imputed[1, 1] = 0.5
        assert imputed[1, 1] == 0.5
        # Non-missing values unchanged
        assert imputed[0, 0] == 1.0
        assert imputed[0, 1] == 0.0

    def test_all_missing_row(self):
        """Row with all missing should be imputed with 0.5."""
        matrix = np.array([[np.nan, np.nan], [1.0, 0.0]])
        mask = ~np.isnan(matrix)

        imputed = impute_column_mean(matrix, mask)

        # Row 0 has no observations, default to 0.5
        assert imputed[0, 0] == 0.5
        assert imputed[0, 1] == 0.5

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        matrix = np.random.rand(10, 20)
        mask = np.random.rand(10, 20) > 0.3

        imputed = impute_column_mean(matrix, mask)

        assert imputed.shape == matrix.shape


class TestComputeVoterPcaProjections:
    """Tests for compute_voter_pca_projections function."""

    def test_basic_projection(self):
        """Should compute projections without error."""
        # Simple matrix with clear structure
        matrix = np.array([
            [1.0, 1.0, 0.0, 0.0],  # Comment 1: group A approves
            [0.0, 0.0, 1.0, 1.0],  # Comment 2: group B approves
        ])

        projections = compute_voter_pca_projections(matrix, n_components=2)

        assert projections.shape == (4, 2)  # 4 voters, 2 components
        # Groups should be separated in first PC
        # Voters 0,1 vs voters 2,3
        assert not np.allclose(projections[0], projections[2])

    def test_output_shape(self):
        """Output shape should be (n_voters, n_components)."""
        matrix = np.random.rand(10, 50)

        projections = compute_voter_pca_projections(matrix, n_components=2)

        assert projections.shape == (50, 2)

    def test_small_matrix(self):
        """Should handle small matrices gracefully."""
        matrix = np.array([[1.0], [0.0]])  # 2 items, 1 voter

        projections = compute_voter_pca_projections(matrix, n_components=2)

        assert projections.shape == (1, 2)

    def test_constant_matrix(self):
        """Constant matrix should return zero projections."""
        matrix = np.ones((5, 10))

        projections = compute_voter_pca_projections(matrix, n_components=2)

        np.testing.assert_array_equal(projections, np.zeros((10, 2)))


class TestClusterVotersKmeans:
    """Tests for cluster_voters_kmeans function."""

    def test_basic_clustering(self):
        """Should cluster voters into groups."""
        # Two clear clusters
        projections = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],  # Cluster 1
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.0],  # Cluster 2
        ])

        labels, k = cluster_voters_kmeans(projections, max_k=5, seed=42)

        assert len(labels) == 6
        assert k >= 2
        # First 3 should be same cluster, last 3 should be same cluster
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_returns_k_in_range(self):
        """Selected k should be in valid range."""
        projections = np.random.rand(100, 2)

        labels, k = cluster_voters_kmeans(projections, max_k=5, seed=42)

        assert 2 <= k <= 5

    def test_too_few_voters(self):
        """Should handle case with too few voters."""
        projections = np.array([[0.0, 0.0], [1.0, 1.0]])

        labels, k = cluster_voters_kmeans(projections, max_k=5, seed=42)

        assert len(labels) == 2
        assert k >= 1

    def test_reproducibility(self):
        """Same seed should give same results."""
        projections = np.random.rand(50, 2)

        labels1, k1 = cluster_voters_kmeans(projections, seed=42)
        labels2, k2 = cluster_voters_kmeans(projections, seed=42)

        np.testing.assert_array_equal(labels1, labels2)
        assert k1 == k2


class TestComputeGroupInformedConsensus:
    """Tests for compute_group_informed_consensus function."""

    def test_unanimous_approval(self):
        """Unanimous approval should give high score."""
        matrix = np.array([[1.0, 1.0, 1.0, 1.0]])  # All approve
        mask = np.ones_like(matrix, dtype=bool)
        labels = np.array([0, 0, 1, 1])  # Two groups

        scores = compute_group_informed_consensus(matrix, mask, labels)

        # Both groups approve -> high consensus
        assert scores[0] > 0.5

    def test_split_approval(self):
        """One group approves, other disapproves -> low score."""
        matrix = np.array([[1.0, 1.0, 0.0, 0.0]])
        mask = np.ones_like(matrix, dtype=bool)
        labels = np.array([0, 0, 1, 1])

        scores = compute_group_informed_consensus(matrix, mask, labels)

        # Group 0: 2/2 approve, Group 1: 0/2 approve
        # P_0 = (2+1)/(2+2) = 0.75
        # P_1 = (0+1)/(2+2) = 0.25
        # consensus = 0.75 * 0.25 = 0.1875
        assert 0.1 < scores[0] < 0.3

    def test_multiple_items(self):
        """Should compute scores for multiple items."""
        matrix = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        mask = np.ones_like(matrix, dtype=bool)
        labels = np.array([0, 0, 1, 1])

        scores = compute_group_informed_consensus(matrix, mask, labels)

        assert len(scores) == 3
        # First item (all approve) should have highest score
        # Third item (none approve) should have lowest score
        assert scores[0] > scores[1] > scores[2]

    def test_scores_in_valid_range(self):
        """Scores should be in [0, 1]."""
        matrix = np.random.randint(0, 2, size=(10, 50)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        labels = np.random.randint(0, 3, size=50)

        scores = compute_group_informed_consensus(matrix, mask, labels)

        assert np.all(scores >= 0)
        assert np.all(scores <= 1)


class TestPolisConsensusPipeline:
    """Tests for full polis_consensus_pipeline function."""

    def test_basic_pipeline(self):
        """Should run without error on basic input."""
        matrix = np.random.randint(0, 2, size=(10, 50)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)

        scores, metadata = polis_consensus_pipeline(matrix, mask, seed=42)

        assert len(scores) == 10
        assert "k_clusters" in metadata
        assert "cluster_sizes" in metadata

    def test_with_missing_data(self):
        """Should handle missing data."""
        matrix = np.random.randint(0, 2, size=(10, 50)).astype(float)
        mask = np.random.rand(10, 50) > 0.3  # 30% missing

        scores, metadata = polis_consensus_pipeline(matrix, mask, seed=42)

        assert len(scores) == 10
        assert not np.any(np.isnan(scores))

    def test_metadata_content(self):
        """Metadata should contain expected fields."""
        matrix = np.random.randint(0, 2, size=(10, 50)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)

        scores, metadata = polis_consensus_pipeline(matrix, mask, seed=42)

        assert "n_items" in metadata
        assert "n_voters" in metadata
        assert "n_pca_components" in metadata
        assert "k_clusters" in metadata
        assert "cluster_sizes" in metadata
        assert "observation_rate" in metadata

    def test_reproducibility(self):
        """Same seed should give same results."""
        matrix = np.random.randint(0, 2, size=(10, 50)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)

        scores1, _ = polis_consensus_pipeline(matrix, mask, seed=42)
        scores2, _ = polis_consensus_pipeline(matrix, mask, seed=42)

        np.testing.assert_array_almost_equal(scores1, scores2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
