"""
Experiment 5: Robustness Comparison Under Random Masking

Compares the robustness of bridging/consensus metrics:
1. Polis Group-Informed Consensus
2. Our Pairwise Disagreement Bridging Score (Naive and IPW)

Two experimental conditions:
- MCAR: Random masking (run_experiment.py)
- Simulated: Polis routing simulation (run_simulation_experiment.py)
"""

from .polis import (
    impute_column_mean,
    compute_voter_pca_projections,
    cluster_voters_kmeans,
    compute_group_informed_consensus,
    polis_consensus_pipeline,
)
from .masking import (
    apply_random_mask,
    generate_trial_seeds,
)
from .robustness import (
    run_single_trial,
    run_masking_experiment,
)
from .robustness_simulated import (
    run_single_simulation_trial,
    run_simulation_experiment,
    create_votes_distributions,
)
from .evaluate import (
    compute_robustness_metrics,
    aggregate_by_mask_rate,
    aggregate_simulation_results,
    compute_simulation_summary,
)
from .visualize import (
    plot_robustness_comparison,
    plot_score_scatter,
    create_summary_table,
    plot_simulation_comparison,
    plot_simulation_multi_metric,
    create_simulation_summary_table,
)

__all__ = [
    # Polis implementation
    "impute_column_mean",
    "compute_voter_pca_projections",
    "cluster_voters_kmeans",
    "compute_group_informed_consensus",
    "polis_consensus_pipeline",
    # Masking utilities
    "apply_random_mask",
    "generate_trial_seeds",
    # MCAR experiment
    "run_single_trial",
    "run_masking_experiment",
    # Simulation experiment
    "run_single_simulation_trial",
    "run_simulation_experiment",
    "create_votes_distributions",
    # Evaluation
    "compute_robustness_metrics",
    "aggregate_by_mask_rate",
    "aggregate_simulation_results",
    "compute_simulation_summary",
    # Visualization
    "plot_robustness_comparison",
    "plot_score_scatter",
    "create_summary_table",
    "plot_simulation_comparison",
    "plot_simulation_multi_metric",
    "create_simulation_summary_table",
]
