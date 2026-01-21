"""
Tests for bridging.py - ground truth bridging score computation.
"""

import numpy as np
import pytest
from experiment_2.bridging import (
    compute_pairwise_disagreement,
    compute_bridging_scores,
    compute_bridging_scores_vectorized,
    compute_bridging_scores_from_disagreement,
)


class TestPairwiseDisagreement:
    """Tests for compute_pairwise_disagreement."""

    def test_identical_voters(self):
        """Voters who agree on everything have zero disagreement."""
        # 3 comments, 2 voters who always agree
        matrix = np.array([
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ])
        d = compute_pairwise_disagreement(matrix)
        assert d[0, 1] == 0.0
        assert d[1, 0] == 0.0

    def test_opposite_voters(self):
        """Voters who disagree on everything have disagreement = 1."""
        matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        d = compute_pairwise_disagreement(matrix)
        assert d[0, 1] == 1.0
        assert d[1, 0] == 1.0

    def test_partial_disagreement(self):
        """Voters who disagree on half the comments have d = 0.5."""
        matrix = np.array([
            [1.0, 1.0],  # agree
            [0.0, 1.0],  # disagree
            [1.0, 1.0],  # agree
            [0.0, 1.0],  # disagree
        ])
        d = compute_pairwise_disagreement(matrix)
        assert d[0, 1] == 0.5

    def test_symmetry(self):
        """Disagreement matrix should be symmetric."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(10, 5)).astype(float)
        d = compute_pairwise_disagreement(matrix)
        np.testing.assert_array_equal(d, d.T)

    def test_diagonal_zero(self):
        """Diagonal should be zero (voter agrees with themselves)."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(10, 5)).astype(float)
        d = compute_pairwise_disagreement(matrix)
        np.testing.assert_array_equal(np.diag(d), np.zeros(5))

    def test_range(self):
        """Disagreement values should be in [0, 1]."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(20, 10)).astype(float)
        d = compute_pairwise_disagreement(matrix)
        assert np.all(d >= 0)
        assert np.all(d <= 1)


class TestBridgingScores:
    """Tests for bridging score computation."""

    def test_single_approver(self):
        """Comment with only one approver has bridging score 0."""
        matrix = np.array([
            [1.0, 0.0, 0.0],  # Only voter 0 approves
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        scores = compute_bridging_scores(matrix)
        assert scores[0] == 0.0

    def test_all_approvers_agree(self):
        """Comment approved by agreeing voters has low bridging score."""
        # All voters agree on everything
        matrix = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])
        scores = compute_bridging_scores(matrix)
        assert scores[0] == 0.0  # No disagreement among approvers

    def test_bridging_comment(self):
        """Comment approved by disagreeing voters has positive bridging score."""
        # Voters 0 and 1 disagree on other comments but both approve comment 0
        matrix = np.array([
            [1.0, 1.0, 0.0],  # comment 0: voters 0,1 approve
            [1.0, 0.0, 0.0],  # comment 1: only voter 0 approves
            [0.0, 1.0, 0.0],  # comment 2: only voter 1 approves
        ])
        scores = compute_bridging_scores(matrix)
        # Voters 0 and 1 disagree on comments 1 and 2 (d_01 = 2/3)
        # With n=3 voters, normalization = 4/n² = 4/9
        # Bridging score of comment 0 = (4/9) × d_01 = (4/9) × (2/3) = 8/27
        assert scores[0] > 0.0
        np.testing.assert_almost_equal(scores[0], 8/27)

    def test_vectorized_matches_original(self):
        """Vectorized version should match original."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(15, 8)).astype(float)
        scores_original = compute_bridging_scores(matrix)
        scores_vectorized = compute_bridging_scores_vectorized(matrix)
        np.testing.assert_array_almost_equal(scores_original, scores_vectorized)

    def test_from_disagreement_matches(self):
        """Computation from disagreement matrix should match."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(12, 6)).astype(float)
        d_matrix = compute_pairwise_disagreement(matrix)
        scores_direct = compute_bridging_scores(matrix)
        scores_from_d = compute_bridging_scores_from_disagreement(matrix, d_matrix)
        np.testing.assert_array_almost_equal(scores_direct, scores_from_d)

    def test_nonnegative(self):
        """Bridging scores should be non-negative."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, size=(20, 10)).astype(float)
        scores = compute_bridging_scores_vectorized(matrix)
        assert np.all(scores >= 0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_comment(self):
        """Single comment matrix."""
        matrix = np.array([[1.0, 1.0, 0.0]])
        d = compute_pairwise_disagreement(matrix)
        scores = compute_bridging_scores(matrix)
        # With only one comment, disagreement is based on that comment alone
        assert d[0, 1] == 0.0  # Both approve
        assert d[0, 2] == 1.0  # 0 approves, 2 disapproves
        # Bridging score: approvers are 0 and 1, they agree on the only comment
        assert scores[0] == 0.0

    def test_single_voter(self):
        """Single voter matrix."""
        matrix = np.array([
            [1.0],
            [0.0],
            [1.0],
        ])
        d = compute_pairwise_disagreement(matrix)
        scores = compute_bridging_scores(matrix)
        assert d.shape == (1, 1)
        assert d[0, 0] == 0.0
        # No pairs of approvers, so all scores are 0
        np.testing.assert_array_equal(scores, [0.0, 0.0, 0.0])

    def test_all_zeros(self):
        """Matrix with all disapprovals."""
        matrix = np.zeros((5, 4))
        scores = compute_bridging_scores(matrix)
        # No approvers for any comment
        np.testing.assert_array_equal(scores, np.zeros(5))

    def test_all_ones(self):
        """Matrix with all approvals."""
        matrix = np.ones((5, 4))
        d = compute_pairwise_disagreement(matrix)
        scores = compute_bridging_scores(matrix)
        # All voters agree on everything
        np.testing.assert_array_equal(d, np.zeros((4, 4)))
        np.testing.assert_array_equal(scores, np.zeros(5))
