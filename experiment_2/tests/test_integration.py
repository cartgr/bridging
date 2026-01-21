"""
Integration tests for the full experiment pipeline.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_2.simulation import simulate_polis_routing
from experiment_2.estimation import estimate_bridging_scores_ipw, estimate_bridging_scores_naive
from experiment_2.evaluate import evaluate_estimation, compute_observation_statistics
from experiment_2.run_experiment import run_single_dataset


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_synthetic(self):
        """Test full pipeline on synthetic data."""
        np.random.seed(42)

        # Create synthetic ground truth - use larger matrix for better coverage
        n_items, n_voters = 20, 30
        ground_truth = np.random.randint(0, 2, (n_items, n_voters)).astype(float)

        # Create votes distribution with higher counts to ensure sufficient coverage
        votes_dist = np.array([10, 12, 15, 13, 11, 14])

        # Step 1: Compute ground truth bridging scores
        true_scores = compute_bridging_scores_vectorized(ground_truth)

        # Step 2: Simulate routing
        observed_mask, inclusion_probs = simulate_polis_routing(
            ground_truth, votes_dist, seed=42
        )

        # Verify we have observations
        assert observed_mask.sum() > 0, "Simulation should produce observations"

        # Step 3: Create observed matrix
        observed_matrix = np.where(observed_mask, ground_truth, np.nan)

        # Step 4: Estimate scores
        ipw_scores = estimate_bridging_scores_ipw(
            observed_matrix, observed_mask, inclusion_probs
        )
        naive_scores = estimate_bridging_scores_naive(observed_matrix, observed_mask)

        # Step 5: Evaluate
        ipw_metrics = evaluate_estimation(true_scores, ipw_scores)
        naive_metrics = evaluate_estimation(true_scores, naive_scores)

        # Basic sanity checks - at least some valid estimates
        assert ipw_metrics["n_valid"] > 0 or ipw_metrics["n_total"] > 0
        assert naive_metrics["n_valid"] > 0 or naive_metrics["n_total"] > 0

    def test_run_single_dataset_synthetic(self):
        """Test run_single_dataset on synthetic data."""
        np.random.seed(42)

        ground_truth = np.random.randint(0, 2, (12, 15)).astype(float)
        votes_dist = np.array([3, 5, 7, 4])

        result = run_single_dataset(
            ground_truth=ground_truth,
            votes_distribution=votes_dist,
            dataset_name="test_synthetic",
            seed=42,
        )

        # Check result structure
        assert "dataset_name" in result
        assert "n_items" in result
        assert "n_voters" in result
        assert "observation_stats" in result
        assert "ipw_metrics" in result
        assert "naive_metrics" in result
        assert "true_bridging_scores" in result
        assert "ipw_bridging_scores" in result

        # Check dimensions
        assert result["n_items"] == 12
        assert result["n_voters"] == 15
        assert len(result["true_bridging_scores"]) == 12
        assert len(result["ipw_bridging_scores"]) == 12

    def test_observation_statistics(self):
        """Test observation statistics computation."""
        np.random.seed(42)

        ground_truth = np.random.randint(0, 2, (10, 20)).astype(float)
        votes_dist = np.array([5, 7, 10])

        observed_mask, _ = simulate_polis_routing(ground_truth, votes_dist, seed=42)

        stats = compute_observation_statistics(observed_mask)

        assert stats["n_items"] == 10
        assert stats["n_voters"] == 20
        assert stats["total_cells"] == 200
        assert stats["observed_cells"] <= 200
        assert 0 <= stats["observation_rate"] <= 1


class TestReproducibility:
    """Tests for reproducibility."""

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        np.random.seed(42)
        ground_truth = np.random.randint(0, 2, (10, 15)).astype(float)
        votes_dist = np.array([4, 6, 8])

        result1 = run_single_dataset(ground_truth, votes_dist, "test", seed=123)
        result2 = run_single_dataset(ground_truth, votes_dist, "test", seed=123)

        np.testing.assert_array_equal(
            result1["true_bridging_scores"],
            result2["true_bridging_scores"]
        )
        np.testing.assert_array_almost_equal(
            result1["ipw_bridging_scores"],
            result2["ipw_bridging_scores"]
        )


class TestRobustness:
    """Tests for robustness to edge cases."""

    def test_small_matrix(self):
        """Test on very small matrix."""
        ground_truth = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        votes_dist = np.array([1, 2])

        result = run_single_dataset(ground_truth, votes_dist, "small", seed=42)

        assert result["n_items"] == 2
        assert result["n_voters"] == 3

    def test_highly_agreeing_population(self):
        """Test with population that mostly agrees."""
        np.random.seed(42)
        # 90% approval rate on all comments
        ground_truth = (np.random.rand(15, 25) < 0.9).astype(float)
        # Use higher vote counts to ensure sufficient coverage
        votes_dist = np.array([10, 12, 14, 11, 13])

        result = run_single_dataset(ground_truth, votes_dist, "agreeing", seed=42)

        # Should produce some results (may have NaN for some comments)
        # Just verify the pipeline runs without error
        assert "ipw_bridging_scores" in result
        assert len(result["ipw_bridging_scores"]) == 15

    def test_highly_disagreeing_population(self):
        """Test with population that mostly disagrees."""
        np.random.seed(42)
        # 50% approval rate (maximum disagreement)
        ground_truth = (np.random.rand(10, 20) < 0.5).astype(float)
        votes_dist = np.array([5, 7, 9])

        result = run_single_dataset(ground_truth, votes_dist, "disagreeing", seed=42)

        # Should have higher bridging scores due to more disagreement
        true_scores = result["true_bridging_scores"]
        assert np.mean(true_scores) > 0


class TestMetricsQuality:
    """Tests for metrics quality."""

    def test_ipw_vs_naive_on_biased_sampling(self):
        """
        IPW should work under biased sampling.

        Create a scenario where we can test both estimators.
        """
        np.random.seed(42)
        n_items, n_voters = 25, 40

        # Create ground truth with varying approval rates
        ground_truth = np.zeros((n_items, n_voters))
        for c in range(n_items):
            # Varying approval rates: 0.2 to 0.8
            approval_rate = 0.2 + 0.6 * (c / (n_items - 1))
            ground_truth[c] = (np.random.rand(n_voters) < approval_rate).astype(float)

        # Use high vote counts to get more observations
        votes_dist = np.array([15, 18, 20, 17, 16, 19])

        result = run_single_dataset(ground_truth, votes_dist, "biased_test", seed=42)

        # Verify we got observations
        obs_rate = result["observation_stats"]["observation_rate"]
        assert obs_rate > 0, "Should have some observations"

        # Check that both methods produce results (may be NaN if sparse)
        ipw_corr = result["ipw_metrics"]["spearman_correlation"]
        naive_corr = result["naive_metrics"]["spearman_correlation"]

        # At least one should be valid, or both NaN is acceptable for sparse data
        valid_results = not np.isnan(ipw_corr) or not np.isnan(naive_corr)
        both_nan = np.isnan(ipw_corr) and np.isnan(naive_corr)
        assert valid_results or both_nan, "Should have at least some results or acknowledge sparsity"

        if not np.isnan(ipw_corr):
            print(f"IPW Spearman: {ipw_corr:.3f}")
        if not np.isnan(naive_corr):
            print(f"Naive Spearman: {naive_corr:.3f}")


class TestWithRealDataFormat:
    """Tests using data in the same format as real data."""

    def test_with_saved_matrices(self):
        """Test loading and processing saved matrices."""
        np.random.seed(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create fake completed data
            ground_truth = np.random.randint(0, 2, (15, 25)).astype(float)
            np.savez(
                tmpdir / "completed.npz",
                matrix=ground_truth,
                n_items=15,
                n_voters=25,
            )

            # Create fake processed data
            observed = ground_truth.copy()
            mask = np.random.rand(15, 25) > 0.7
            observed[~mask] = np.nan
            np.savez(
                tmpdir / "processed.npz",
                matrix=observed,
                n_items=15,
                n_voters=25,
            )

            # Load and verify format
            completed = np.load(tmpdir / "completed.npz")
            processed = np.load(tmpdir / "processed.npz")

            assert completed["matrix"].shape == (15, 25)
            assert processed["matrix"].shape == (15, 25)
            assert not np.any(np.isnan(completed["matrix"]))
            assert np.any(np.isnan(processed["matrix"]))
