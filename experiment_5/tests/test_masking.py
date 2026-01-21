"""Tests for masking utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiment_5.masking import (
    apply_random_mask,
    generate_trial_seeds,
    compute_actual_mask_rate,
)


class TestApplyRandomMask:
    """Tests for apply_random_mask function."""

    def test_zero_mask_rate(self):
        """Zero mask rate should keep all entries."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

        masked, observed = apply_random_mask(matrix, mask_rate=0.0, seed=42)

        np.testing.assert_array_equal(masked, matrix)
        np.testing.assert_array_equal(observed, np.ones_like(matrix, dtype=bool))

    def test_full_mask_rate(self):
        """Full mask rate should hide all entries."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

        masked, observed = apply_random_mask(matrix, mask_rate=1.0, seed=42)

        assert np.all(np.isnan(masked))
        assert np.all(~observed)

    def test_partial_mask_rate(self):
        """Partial mask rate should hide some entries."""
        matrix = np.ones((100, 100))

        masked, observed = apply_random_mask(matrix, mask_rate=0.5, seed=42)

        # Should have approximately 50% observed (with some variance)
        obs_rate = observed.sum() / observed.size
        assert 0.4 < obs_rate < 0.6

        # Masked entries should be NaN
        assert np.isnan(masked[~observed]).all()
        # Observed entries should be unchanged
        np.testing.assert_array_equal(masked[observed], matrix[observed])

    def test_reproducibility(self):
        """Same seed should give same mask."""
        matrix = np.random.rand(50, 50)

        masked1, obs1 = apply_random_mask(matrix, mask_rate=0.3, seed=42)
        masked2, obs2 = apply_random_mask(matrix, mask_rate=0.3, seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds(self):
        """Different seeds should give different masks."""
        matrix = np.random.rand(50, 50)

        _, obs1 = apply_random_mask(matrix, mask_rate=0.3, seed=42)
        _, obs2 = apply_random_mask(matrix, mask_rate=0.3, seed=123)

        assert not np.array_equal(obs1, obs2)

    def test_min_per_item_constraint(self):
        """Should ensure minimum observations per item."""
        matrix = np.ones((10, 100))

        _, observed = apply_random_mask(
            matrix, mask_rate=0.9, seed=42, min_observed_per_item=5
        )

        # Each row should have at least 5 observations
        per_item = observed.sum(axis=1)
        assert np.all(per_item >= 5)

    def test_min_per_voter_constraint(self):
        """Should ensure minimum observations per voter."""
        matrix = np.ones((100, 10))

        _, observed = apply_random_mask(
            matrix, mask_rate=0.9, seed=42, min_observed_per_voter=3
        )

        # Each column should have at least 3 observations
        per_voter = observed.sum(axis=0)
        assert np.all(per_voter >= 3)

    def test_invalid_mask_rate(self):
        """Should raise error for invalid mask rate."""
        matrix = np.ones((5, 5))

        with pytest.raises(ValueError):
            apply_random_mask(matrix, mask_rate=-0.1, seed=42)

        with pytest.raises(ValueError):
            apply_random_mask(matrix, mask_rate=1.5, seed=42)

    def test_shape_preserved(self):
        """Output shapes should match input shape."""
        matrix = np.random.rand(15, 25)

        masked, observed = apply_random_mask(matrix, mask_rate=0.4, seed=42)

        assert masked.shape == matrix.shape
        assert observed.shape == matrix.shape


class TestGenerateTrialSeeds:
    """Tests for generate_trial_seeds function."""

    def test_correct_count(self):
        """Should generate correct number of seeds."""
        seeds = generate_trial_seeds(n_trials=10, base_seed=42)

        assert len(seeds) == 10

    def test_unique_seeds(self):
        """Seeds should be unique."""
        seeds = generate_trial_seeds(n_trials=100, base_seed=42)

        assert len(np.unique(seeds)) == 100

    def test_reproducibility(self):
        """Same base seed should give same seeds."""
        seeds1 = generate_trial_seeds(n_trials=10, base_seed=42)
        seeds2 = generate_trial_seeds(n_trials=10, base_seed=42)

        np.testing.assert_array_equal(seeds1, seeds2)

    def test_different_base_seeds(self):
        """Different base seeds should give different seeds."""
        seeds1 = generate_trial_seeds(n_trials=10, base_seed=42)
        seeds2 = generate_trial_seeds(n_trials=10, base_seed=123)

        assert not np.array_equal(seeds1, seeds2)

    def test_seeds_are_integers(self):
        """Seeds should be valid integer values."""
        seeds = generate_trial_seeds(n_trials=10, base_seed=42)

        assert seeds.dtype in [np.int32, np.int64]


class TestComputeActualMaskRate:
    """Tests for compute_actual_mask_rate function."""

    def test_fully_observed(self):
        """Fully observed should return 1.0."""
        mask = np.ones((10, 20), dtype=bool)

        rate = compute_actual_mask_rate(mask)

        assert rate == 1.0

    def test_fully_masked(self):
        """Fully masked should return 0.0."""
        mask = np.zeros((10, 20), dtype=bool)

        rate = compute_actual_mask_rate(mask)

        assert rate == 0.0

    def test_partial_observation(self):
        """Should compute correct rate for partial observation."""
        mask = np.array([[True, True, False], [False, True, True]])

        rate = compute_actual_mask_rate(mask)

        assert rate == 4 / 6  # 4 out of 6 observed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
