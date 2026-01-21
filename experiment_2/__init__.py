"""
Experiment 2: Bridging Score Estimation Under Pol.is Sampling

This experiment tests how well we can estimate bridging scores from partially
observed Pol.is data using inverse-probability weighting (IPW).
"""

from .bridging import (
    compute_pairwise_disagreement,
    compute_bridging_scores,
    compute_bridging_scores_vectorized,
    compute_bridging_scores_from_disagreement,
)

from .priority import (
    compute_pca_extremeness,
    compute_vote_stats,
    compute_priorities,
    compute_sampling_probabilities,
    compute_inclusion_probability_exact,
)

from .simulation import (
    get_empirical_votes_distribution,
    simulate_voter_session,
    simulate_polis_routing,
    simulate_polis_routing_batch,
    verify_inclusion_probabilities_monte_carlo,
)

from .estimation import (
    estimate_pairwise_disagreement_ipw,
    estimate_bridging_scores_ipw,
    estimate_bridging_scores_ipw_direct,
    estimate_bridging_scores_naive,
)

from .evaluate import (
    evaluate_estimation,
    evaluate_estimation_monte_carlo,
    compute_observation_statistics,
    format_results,
)

__all__ = [
    # bridging
    "compute_pairwise_disagreement",
    "compute_bridging_scores",
    "compute_bridging_scores_vectorized",
    "compute_bridging_scores_from_disagreement",
    # priority
    "compute_pca_extremeness",
    "compute_vote_stats",
    "compute_priorities",
    "compute_sampling_probabilities",
    "compute_inclusion_probability_exact",
    # simulation
    "get_empirical_votes_distribution",
    "simulate_voter_session",
    "simulate_polis_routing",
    "simulate_polis_routing_batch",
    "verify_inclusion_probabilities_monte_carlo",
    # estimation
    "estimate_pairwise_disagreement_ipw",
    "estimate_bridging_scores_ipw",
    "estimate_bridging_scores_ipw_direct",
    "estimate_bridging_scores_naive",
    # evaluate
    "evaluate_estimation",
    "evaluate_estimation_monte_carlo",
    "compute_observation_statistics",
    "format_results",
]
