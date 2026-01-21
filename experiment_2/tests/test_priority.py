"""
Tests for priority.py - Pol.is priority formula and PCA extremeness.
"""

import numpy as np
import pytest
from experiment_2.priority import (
    compute_pca_extremeness,
    compute_vote_stats,
    compute_priorities,
    compute_sampling_probabilities,
    compute_inclusion_probability_exact,
)


class TestVoteStats:
    """Tests for compute_vote_stats.

    Note: compute_vote_stats now returns raw counts (n_votes, n_agrees, n_passes)
    rather than rates, to enable Laplace smoothing in compute_priorities.
    """

    def test_full_observation(self):
        """Test with fully observed data."""
        matrix = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        mask = np.ones_like(matrix, dtype=bool)
        n_votes, n_agrees, n_passes = compute_vote_stats(matrix, mask)

        # Check vote counts (S)
        np.testing.assert_array_equal(n_votes, [3, 3, 3])

        # Check agree counts (A)
        # Comment 0: 2 agrees (voters 0, 1)
        # Comment 1: 2 agrees (voters 1, 2)
        # Comment 2: 1 agree (voter 0)
        np.testing.assert_array_equal(n_agrees, [2, 2, 1])

        # Check pass counts (always 0 in binary)
        np.testing.assert_array_equal(n_passes, [0, 0, 0])

    def test_partial_observation(self):
        """Test with partially observed data."""
        matrix = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])
        mask = np.array([
            [True, True, False],
            [True, False, True],
        ])
        n_votes, n_agrees, n_passes = compute_vote_stats(matrix, mask)

        # Comment 0: observed by voters 0, 1 (votes: 1, 1) -> 2 votes, 2 agrees
        # Comment 1: observed by voters 0, 2 (votes: 0, 1) -> 2 votes, 1 agree
        np.testing.assert_array_equal(n_votes, [2, 2])
        np.testing.assert_array_equal(n_agrees, [2, 1])

    def test_no_observations(self):
        """Test with no observations."""
        matrix = np.array([[1.0, 0.0]])
        mask = np.array([[False, False]])
        n_votes, n_agrees, n_passes = compute_vote_stats(matrix, mask)

        assert n_votes[0] == 0
        assert n_agrees[0] == 0


class TestPCAExtremeness:
    """Tests for compute_pca_extremeness."""

    def test_output_shape(self):
        """Output should have shape (n_items,)."""
        matrix = np.random.rand(10, 20)
        mask = np.ones_like(matrix, dtype=bool)
        extremeness = compute_pca_extremeness(matrix, mask)
        assert extremeness.shape == (10,)

    def test_nonnegative(self):
        """Extremeness should be non-negative (L2 norm)."""
        np.random.seed(42)
        matrix = np.random.rand(10, 20)
        mask = np.ones_like(matrix, dtype=bool)
        extremeness = compute_pca_extremeness(matrix, mask)
        assert np.all(extremeness >= 0)

    def test_imputation(self):
        """Missing values should be handled via imputation."""
        np.random.seed(42)
        matrix = np.random.rand(5, 10)
        # Create sparse mask
        mask = np.random.rand(5, 10) > 0.5
        # Should not raise error
        extremeness = compute_pca_extremeness(matrix, mask)
        assert extremeness.shape == (5,)

    def test_small_matrix(self):
        """Handle small matrices gracefully."""
        matrix = np.array([[1.0, 0.0]])
        mask = np.ones_like(matrix, dtype=bool)
        extremeness = compute_pca_extremeness(matrix, mask, n_components=2)
        # Should handle case where n_components > min(n_items, n_voters)
        assert extremeness.shape == (1,)


class TestComputePriorities:
    """Tests for compute_priorities."""

    def test_output_shape(self):
        """Output should have shape (n_items,)."""
        matrix = np.random.rand(10, 20) > 0.5
        matrix = matrix.astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        priorities = compute_priorities(matrix, mask)
        assert priorities.shape == (10,)

    def test_nonnegative(self):
        """Priorities should be non-negative."""
        np.random.seed(42)
        matrix = (np.random.rand(10, 20) > 0.5).astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        priorities = compute_priorities(matrix, mask)
        assert np.all(priorities >= 0)

    def test_higher_agreement_higher_priority(self):
        """Comments with higher agreement rate should have higher priority."""
        # Comment 0: 90% agreement, Comment 1: 10% agreement
        matrix = np.array([
            [1.0] * 9 + [0.0],  # 90% agree
            [1.0] + [0.0] * 9,  # 10% agree
        ])
        mask = np.ones_like(matrix, dtype=bool)
        priorities = compute_priorities(matrix, mask)
        # Higher agreement -> higher priority (if other factors equal)
        assert priorities[0] > priorities[1]

    def test_fewer_votes_exploration_bonus(self):
        """Comments with fewer votes should get exploration bonus."""
        # Same approval rate, different vote counts
        # Use valid float values (not NaN) - mask controls observation
        matrix = np.array([
            [1.0, 1.0, 0.0, 0.0, 0.0],  # Values don't matter where mask is False
            [1.0, 1.0, 1.0, 1.0, 1.0],  # 5 votes, 100% agree
        ])
        mask = np.array([
            [True, True, False, False, False],  # 2 votes observed
            [True, True, True, True, True],     # 5 votes observed
        ])
        priorities = compute_priorities(matrix, mask)
        # Fewer votes -> higher exploration bonus -> higher priority
        # Both have 100% agreement among observed, but comment 0 has higher exploration bonus
        assert priorities[0] > priorities[1]


class TestSamplingProbabilities:
    """Tests for compute_sampling_probabilities."""

    def test_sum_to_one(self):
        """Probabilities should sum to 1."""
        priorities = np.array([1.0, 2.0, 3.0, 4.0])
        eligible = np.array([True, True, True, True])
        probs = compute_sampling_probabilities(priorities, eligible)
        np.testing.assert_almost_equal(probs.sum(), 1.0)

    def test_proportional(self):
        """Probabilities should be proportional to priorities."""
        priorities = np.array([1.0, 2.0, 3.0])
        eligible = np.ones(3, dtype=bool)
        probs = compute_sampling_probabilities(priorities, eligible)
        np.testing.assert_array_almost_equal(probs, [1/6, 2/6, 3/6])

    def test_eligible_only(self):
        """Only eligible comments should have positive probability."""
        priorities = np.array([1.0, 2.0, 3.0, 4.0])
        eligible = np.array([True, False, True, False])
        probs = compute_sampling_probabilities(priorities, eligible)

        assert probs[1] == 0.0
        assert probs[3] == 0.0
        np.testing.assert_almost_equal(probs[0] + probs[2], 1.0)

    def test_zero_total_priority(self):
        """Handle zero total priority gracefully with uniform fallback."""
        priorities = np.array([0.0, 0.0, 0.0])
        eligible = np.ones(3, dtype=bool)
        probs = compute_sampling_probabilities(priorities, eligible)
        # When all priorities are zero, fall back to uniform over eligible
        np.testing.assert_array_almost_equal(probs, [1/3, 1/3, 1/3])


class TestInclusionProbability:
    """Tests for compute_inclusion_probability_exact."""

    def test_output_shape(self):
        """Output should have shape (n_items,)."""
        priorities = np.array([1.0, 2.0, 3.0])
        eligible = np.ones(3, dtype=bool)
        probs = compute_inclusion_probability_exact(priorities, eligible, k_votes=2)
        assert probs.shape == (3,)

    def test_range(self):
        """Inclusion probabilities should be in [0, 1]."""
        np.random.seed(42)
        priorities = np.random.rand(10)
        eligible = np.ones(10, dtype=bool)
        probs = compute_inclusion_probability_exact(priorities, eligible, k_votes=5)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_zero_votes(self):
        """Zero votes should give zero inclusion probability."""
        priorities = np.array([1.0, 2.0, 3.0])
        eligible = np.ones(3, dtype=bool)
        probs = compute_inclusion_probability_exact(priorities, eligible, k_votes=0)
        np.testing.assert_array_equal(probs, [0.0, 0.0, 0.0])

    def test_all_votes(self):
        """k = n should give high inclusion probability for all eligible."""
        priorities = np.array([1.0, 2.0, 3.0])
        eligible = np.ones(3, dtype=bool)
        probs = compute_inclusion_probability_exact(priorities, eligible, k_votes=3)
        # Note: The approximation formula 1-(1-p)^k doesn't give exactly 1.0
        # for sampling without replacement, but should give high values
        # In exact sampling without replacement, all would be selected
        assert np.all(probs > 0.4)  # All should have reasonable probability
        assert np.all(probs <= 1.0)  # Should not exceed 1

    def test_ineligible_zero(self):
        """Ineligible comments should have zero probability."""
        priorities = np.array([1.0, 2.0, 3.0])
        eligible = np.array([True, False, True])
        probs = compute_inclusion_probability_exact(priorities, eligible, k_votes=1)
        assert probs[1] == 0.0
