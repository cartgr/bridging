"""
Tests for estimation.py - IPW bridging score estimation.
"""

import numpy as np
import pytest

from experiment_2.bridging import (
    compute_pairwise_disagreement,
    compute_bridging_scores_vectorized,
)
from experiment_2.estimation import (
    estimate_pairwise_disagreement_ipw,
    estimate_bridging_scores_ipw,
    estimate_bridging_scores_naive,
)


class TestIPWPairwiseDisagreement:
    """Tests for estimate_pairwise_disagreement_ipw."""

    def test_full_observation_matches_ground_truth(self):
        """With full observation and prob=1, IPW should match ground truth."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 5)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        probs = np.ones_like(matrix)

        d_true = compute_pairwise_disagreement(matrix)
        d_ipw = estimate_pairwise_disagreement_ipw(matrix, mask, probs)

        np.testing.assert_array_almost_equal(d_ipw, d_true)

    def test_symmetry(self):
        """Estimated disagreement should be symmetric."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 5)).astype(float)
        mask = np.random.rand(10, 5) > 0.3
        probs = np.random.rand(10, 5) * 0.5 + 0.5  # [0.5, 1]

        d_ipw = estimate_pairwise_disagreement_ipw(matrix, mask, probs)

        # Check symmetry for non-NaN values
        for i in range(5):
            for j in range(i + 1, 5):
                if not np.isnan(d_ipw[i, j]):
                    np.testing.assert_almost_equal(d_ipw[i, j], d_ipw[j, i])

    def test_nan_for_no_overlap(self):
        """Return NaN when voters have no common observations."""
        matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        # Voter 0 sees comment 0, voter 1 sees comment 1
        mask = np.array([
            [True, False],
            [False, True],
        ])
        probs = np.ones_like(matrix)

        d_ipw = estimate_pairwise_disagreement_ipw(matrix, mask, probs)

        assert np.isnan(d_ipw[0, 1])
        assert np.isnan(d_ipw[1, 0])

    def test_ipw_weighting(self):
        """IPW should correctly weight by inverse probability."""
        # Simple case: 2 comments, 2 voters, one comment always observed
        matrix = np.array([
            [1.0, 0.0],  # disagree on comment 0
            [1.0, 1.0],  # agree on comment 1
        ])
        mask = np.array([
            [True, True],   # Both see comment 0
            [True, True],   # Both see comment 1
        ])
        # Different inclusion probabilities
        probs = np.array([
            [1.0, 1.0],   # Comment 0: prob 1 for both
            [0.5, 0.5],   # Comment 1: prob 0.5 for both
        ])

        d_ipw = estimate_pairwise_disagreement_ipw(matrix, mask, probs)

        # True disagreement: 1 comment disagree out of 2 -> 0.5
        # IPW estimate:
        # - Comment 0: disagree=1, weight=1/(1*1)=1, contribution=1
        # - Comment 1: disagree=0, weight=1/(0.5*0.5)=4, contribution=0
        # d_ipw = (1*1 + 0*4) / 2 = 0.5
        np.testing.assert_almost_equal(d_ipw[0, 1], 0.5)


class TestIPWBridgingScores:
    """Tests for estimate_bridging_scores_ipw."""

    def test_full_observation_matches_ground_truth(self):
        """With full observation and prob=1, IPW should match ground truth."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 5)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        probs = np.ones_like(matrix)

        true_scores = compute_bridging_scores_vectorized(matrix)
        ipw_scores = estimate_bridging_scores_ipw(matrix, mask, probs)

        np.testing.assert_array_almost_equal(ipw_scores, true_scores)

    def test_nonnegative(self):
        """Estimated bridging scores should be non-negative."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 8)).astype(float)
        mask = np.random.rand(10, 8) > 0.3
        probs = np.random.rand(10, 8) * 0.5 + 0.5

        scores = estimate_bridging_scores_ipw(matrix, mask, probs)

        # Filter out NaN values
        valid_scores = scores[~np.isnan(scores)]
        assert np.all(valid_scores >= 0)

    def test_single_approver_zero_score(self):
        """Comment with single observed approver should have score 0."""
        matrix = np.array([
            [1.0, 0.0, 0.0],  # Only voter 0 approves
            [1.0, 1.0, 1.0],
        ])
        mask = np.ones_like(matrix, dtype=bool)
        probs = np.ones_like(matrix)

        scores = estimate_bridging_scores_ipw(matrix, mask, probs)

        assert scores[0] == 0.0  # Only one approver


class TestNaiveEstimator:
    """Tests for estimate_bridging_scores_naive."""

    def test_full_observation_matches_ground_truth(self):
        """With full observation, naive should match ground truth."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 5)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)

        true_scores = compute_bridging_scores_vectorized(matrix)
        naive_scores = estimate_bridging_scores_naive(matrix, mask)

        # Naive method normalizes differently, so compare rankings
        true_ranking = np.argsort(true_scores)
        naive_ranking = np.argsort(naive_scores)

        # With full data, rankings should be identical
        np.testing.assert_array_equal(true_ranking, naive_ranking)

    def test_nonnegative(self):
        """Naive estimates should be non-negative."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 8)).astype(float)
        mask = np.random.rand(10, 8) > 0.3

        scores = estimate_bridging_scores_naive(matrix, mask)

        valid_scores = scores[~np.isnan(scores)]
        assert np.all(valid_scores >= 0)


class TestUnbiasedness:
    """Tests for unbiasedness of IPW estimator."""

    @pytest.mark.slow
    def test_ipw_unbiased_monte_carlo(self):
        """
        Monte Carlo test: IPW estimates should be unbiased.

        Run many simulations with known inclusion probabilities,
        average the estimates should approach ground truth.
        """
        np.random.seed(42)
        n_items, n_voters = 8, 6
        n_simulations = 200

        # Fixed ground truth
        ground_truth = np.random.randint(0, 2, (n_items, n_voters)).astype(float)
        true_d = compute_pairwise_disagreement(ground_truth)

        # Fixed inclusion probabilities (simulating known sampling)
        base_probs = np.random.rand(n_items, n_voters) * 0.5 + 0.3  # [0.3, 0.8]

        # Run simulations
        d_estimates = []
        for sim in range(n_simulations):
            rng = np.random.default_rng(42 + sim)

            # Generate observations according to probabilities
            mask = rng.random((n_items, n_voters)) < base_probs

            if mask.sum() < n_items:
                continue  # Skip if too sparse

            d_est = estimate_pairwise_disagreement_ipw(
                ground_truth, mask, base_probs
            )
            d_estimates.append(d_est)

        # Average estimates (handling NaN)
        d_avg = np.nanmean(d_estimates, axis=0)

        # Check that average is close to truth for non-NaN entries
        valid = ~np.isnan(d_avg) & ~np.isnan(true_d)
        if valid.sum() > 0:
            mae = np.abs(d_avg[valid] - true_d[valid]).mean()
            # Allow some tolerance due to variance
            assert mae < 0.15, f"IPW appears biased: MAE = {mae:.3f}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_same_votes(self):
        """Handle case where all votes are the same."""
        matrix = np.ones((5, 4))
        mask = np.ones_like(matrix, dtype=bool)
        probs = np.ones_like(matrix)

        d_ipw = estimate_pairwise_disagreement_ipw(matrix, mask, probs)
        scores = estimate_bridging_scores_ipw(matrix, mask, probs)

        # No disagreement
        np.testing.assert_array_almost_equal(d_ipw, np.zeros((4, 4)))
        np.testing.assert_array_almost_equal(scores, np.zeros(5))

    def test_very_small_probabilities(self):
        """Handle very small inclusion probabilities."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (5, 3)).astype(float)
        mask = np.ones_like(matrix, dtype=bool)
        probs = np.full_like(matrix, 0.001)  # Very small

        # Should not raise error (min_prob clipping)
        scores = estimate_bridging_scores_ipw(matrix, mask, probs, min_prob=0.01)

        assert not np.any(np.isinf(scores))

    def test_sparse_observations(self):
        """Handle very sparse observations."""
        np.random.seed(42)
        matrix = np.random.randint(0, 2, (10, 8)).astype(float)
        # Very sparse: only ~10% observed
        mask = np.random.rand(10, 8) > 0.9
        probs = np.where(mask, 0.1, 0.1)  # Uniform prob

        # Should handle gracefully (may produce NaN for some)
        scores = estimate_bridging_scores_ipw(matrix, mask, probs)

        # At minimum, should not crash
        assert scores.shape == (10,)
