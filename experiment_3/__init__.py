"""
Experiment 3: Voter PCA Spectrum Visualization

This experiment visualizes which voters (positioned on a left-right political
spectrum via PCA) approve each candidate in the French Election data. This
helps visualize whether candidates are "bridging" (approved by voters across
the spectrum) or "polarizing" (approved by only one side).
"""

from .visualize import (
    compute_voter_pca_scores,
    plot_voter_spectrum,
    main,
)

__all__ = [
    "compute_voter_pca_scores",
    "plot_voter_spectrum",
    "main",
]
