"""
Experiment 4: Synthetic Elections with Group Structure

This experiment creates synthetic elections with known group structures to
understand bridging score behavior analytically. Groups are defined by approval
of disjoint sets of "base" comments, and focal comments vary in their approval
rates from each group.

Key insights:
- Maximum bridging occurs when a comment receives balanced support from
  different groups that otherwise disagree
- Bridging score depends on both approval rates and group sizes
"""

from .synthetic import (
    generate_synthetic_matrix,
    assign_voters_to_groups,
    compute_focal_bridging_score_fast,
    generate_and_compute_bridging_fast,
)

from .analyze import (
    compute_bridging_surface_equal_groups,
    compute_bridging_surface_unequal_groups,
    plot_3d_surface,
    main,
)

__all__ = [
    # synthetic
    "generate_synthetic_matrix",
    "assign_voters_to_groups",
    "compute_focal_bridging_score_fast",
    "generate_and_compute_bridging_fast",
    # analyze
    "compute_bridging_surface_equal_groups",
    "compute_bridging_surface_unequal_groups",
    "plot_3d_surface",
    "main",
]
