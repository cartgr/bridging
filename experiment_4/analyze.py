"""
Analyze bridging scores for synthetic elections with group structure.

Generates 3D surface plots showing bridging score as a function of group
approval rates.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def compute_expected_bridging_score(
    w: float,
    a1: float,
    a2: float,
    n_base_comments: int,
) -> float:
    """
    Compute the expected bridging score analytically.

    E[bridging] = 4 × w × (1-w) × a₁ × a₂ × d_between

    Where d_between = n_base / (n_base + 1) for the synthetic group structure.

    Args:
        w: Group 1 size fraction
        a1: Group 1 approval rate
        a2: Group 2 approval rate
        n_base_comments: Number of base comments (affects d_between)

    Returns:
        Expected bridging score
    """
    d_between = n_base_comments / (n_base_comments + 1)
    return 4 * w * (1 - w) * a1 * a2 * d_between


def compute_bridging_surface_equal_groups(
    n_base_comments: int,
    a1_range: np.ndarray,
    a2_range: np.ndarray,
) -> np.ndarray:
    """
    Compute expected bridging scores for grid of (a1, a2) values with two equal groups.

    Uses analytical formula: E[b] = 4 × 0.5 × 0.5 × a₁ × a₂ × d_between = a₁ × a₂ × d_between

    Args:
        n_base_comments: Number of base comments (affects d_between)
        a1_range: Array of approval rates for group 1
        a2_range: Array of approval rates for group 2

    Returns:
        (len(a1_range), len(a2_range)) array of bridging scores
    """
    A1, A2 = np.meshgrid(a1_range, a2_range, indexing='ij')
    d_between = n_base_comments / (n_base_comments + 1)

    # For equal groups: w = 0.5, so 4 × w × (1-w) = 4 × 0.25 = 1
    bridging_scores = A1 * A2 * d_between

    return bridging_scores


def plot_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    output_path: Path,
    mask: Optional[np.ndarray] = None,
    invert_x: bool = False,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> Optional[plt.Figure]:
    """
    Create 3D surface plot.

    Args:
        x: 1D array for x-axis values
        y: 1D array for y-axis values
        z: 2D array of z values (shape: len(x) x len(y))
        title: Plot title
        xlabel, ylabel, zlabel: Axis labels
        output_path: Path to save the plot (ignored if ax is provided)
        mask: Optional boolean mask for valid points
        invert_x: If True, invert the x-axis so highest values are in back
        ax: Optional existing axes to plot on
        show_colorbar: Whether to show colorbar
    """
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply mask if provided
    Z = z.copy()
    if mask is not None:
        Z[~mask] = np.nan

    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_zlabel(zlabel, fontsize=10)
    ax.set_title(title, fontsize=12)

    # Invert x-axis if requested (so high values are in back)
    if invert_x:
        ax.invert_xaxis()

    # Add colorbar
    if show_colorbar and standalone:
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)

    if standalone:
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    return fig if standalone else None


def plot_3d_interactive(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    output_path: Path,
    mask: Optional[np.ndarray] = None,
) -> None:
    """
    Create interactive 3D surface plot with plotly.

    Args:
        x: 1D array for x-axis values
        y: 1D array for y-axis values
        z: 2D array of z values (shape: len(x) x len(y))
        title: Plot title
        xlabel, ylabel, zlabel: Axis labels
        output_path: Path to save the HTML file
        mask: Optional boolean mask for valid points
    """
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Apply mask if provided
    Z = z.copy()
    if mask is not None:
        Z[~mask] = np.nan

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title=zlabel),
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
        ),
        width=900,
        height=700,
    )

    fig.write_html(output_path)
    print(f"Saved interactive plot: {output_path}")


def run_scenario_1_equal_groups(
    n_base_comments: int = 1000,
    grid_resolution: int = 50,
    output_dir: Path = None,
):
    """
    Scenario 1: Two equal-sized groups.

    Varies a1, a2 independently and plots bridging surface.
    Uses analytical formula: E[b] = a₁ × a₂ × d_between
    """
    print("\n=== Scenario 1: Two Equal-Sized Groups ===")
    print("Formula: E[b] = a₁ × a₂ × d_between (where d_between ≈ 1)")

    if output_dir is None:
        output_dir = Path(__file__).parent / "plots"

    a1_range = np.linspace(0, 1, grid_resolution)
    a2_range = np.linspace(0, 1, grid_resolution)

    print(f"Computing bridging surface ({grid_resolution}x{grid_resolution} grid)...")
    bridging_surface = compute_bridging_surface_equal_groups(
        n_base_comments=n_base_comments,
        a1_range=a1_range,
        a2_range=a2_range,
    )

    # Verify properties
    print("\nVerifying properties:")
    print(f"  Min bridging: {bridging_surface.min():.4f}")
    print(f"  Max bridging: {bridging_surface.max():.4f}")
    print(f"  Bridging at (0.5, 0.5): {bridging_surface[grid_resolution//2, grid_resolution//2]:.4f}")
    print(f"  Bridging at (1.0, 0.0): {bridging_surface[-1, 0]:.4f}")
    print(f"  Bridging at (0.0, 1.0): {bridging_surface[0, -1]:.4f}")

    # Check symmetry
    symmetry_diff = np.abs(bridging_surface - bridging_surface.T).max()
    print(f"  Symmetry check (max |b(a1,a2) - b(a2,a1)|): {symmetry_diff:.6f}")

    # Static plot with inverted x-axis so highest point (a1=1, a2=1) is in back corner
    plot_3d_surface(
        x=a1_range,
        y=a2_range,
        z=bridging_surface,
        title="Bridging Score: Two Equal-Sized Groups\nE[b] = a₁ × a₂",
        xlabel="Group 1 Approval Rate (a₁)",
        ylabel="Group 2 Approval Rate (a₂)",
        zlabel="Bridging Score",
        output_path=output_dir / "equal_groups_3d.png",
        invert_x=True,
    )

    # Interactive plot
    plot_3d_interactive(
        x=a1_range,
        y=a2_range,
        z=bridging_surface,
        title="Bridging Score: Two Equal-Sized Groups (E[b] = a₁ × a₂)",
        xlabel="Group 1 Approval Rate (a₁)",
        ylabel="Group 2 Approval Rate (a₂)",
        zlabel="Bridging Score",
        output_path=output_dir / "equal_groups_interactive.html",
    )

    return bridging_surface


def compute_bridging_surface_unequal_groups(
    n_base_comments: int,
    w_range: np.ndarray,
    a1_range: np.ndarray,
    total_approval: float = 0.5,
) -> tuple:
    """
    Compute expected bridging scores for two unequal-sized groups with fixed total approval.

    Constraint: w·a₁ + (1-w)·a₂ = total_approval
    So: a₂ = (total_approval - w·a₁) / (1-w)

    Formula: E[b] = 4 × w × (1-w) × a₁ × a₂ × d_between

    Args:
        n_base_comments: Number of base comments (affects d_between)
        w_range: Array of group 1 size fractions
        a1_range: Array of approval rates for group 1
        total_approval: Fixed total approval rate

    Returns:
        (bridging_scores, valid_mask, A2) where:
        - bridging_scores: 2D array of shape (len(w_range), len(a1_range))
        - valid_mask: Boolean mask where a₂ ∈ [0, 1]
        - A2: 2D array of computed a₂ values
    """
    W, A1 = np.meshgrid(w_range, a1_range, indexing='ij')

    # Compute a₂ from constraint: w·a₁ + (1-w)·a₂ = total_approval
    # a₂ = (total_approval - w·a₁) / (1-w)
    # Handle w=1 case to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        A2 = (total_approval - W * A1) / (1 - W)

    # Valid where a₂ ∈ [0, 1] and w < 1
    valid_mask = (A2 >= 0) & (A2 <= 1) & (W < 1)

    # Compute bridging score
    d_between = n_base_comments / (n_base_comments + 1)
    bridging_scores = 4 * W * (1 - W) * A1 * A2 * d_between

    # Zero out invalid regions
    bridging_scores[~valid_mask] = np.nan

    return bridging_scores, valid_mask, A2


def run_scenario_2_unequal_groups(
    n_base_comments: int = 1000,
    grid_resolution: int = 50,
    output_dir: Path = None,
):
    """
    Scenario 2: Two unequal-sized groups with fixed total approval.

    Constraint: w·a₁ + (1-w)·a₂ = 0.5
    Varies w and a₁, with a₂ determined by the constraint.
    """
    print("\n=== Scenario 2: Two Unequal-Sized Groups ===")
    print("Constraint: w·a₁ + (1-w)·a₂ = 0.5")
    print("Formula: E[b] = 4 × w × (1-w) × a₁ × a₂ × d_between")

    if output_dir is None:
        output_dir = Path(__file__).parent / "plots"

    # Avoid w=0 and w=1 to prevent division issues
    w_range = np.linspace(0.05, 0.95, grid_resolution)
    a1_range = np.linspace(0, 1, grid_resolution)

    print(f"Computing bridging surface ({grid_resolution}x{grid_resolution} grid)...")
    bridging_scores, valid_mask, A2 = compute_bridging_surface_unequal_groups(
        n_base_comments=n_base_comments,
        w_range=w_range,
        a1_range=a1_range,
        total_approval=0.5,
    )

    valid_scores = bridging_scores[valid_mask]
    print(f"\nVerifying properties:")
    print(f"  Valid points: {valid_mask.sum()} / {valid_mask.size}")
    print(f"  Min bridging: {np.nanmin(valid_scores):.4f}")
    print(f"  Max bridging: {np.nanmax(valid_scores):.4f}")

    # Find maximum location
    max_idx = np.nanargmax(bridging_scores)
    max_i, max_j = np.unravel_index(max_idx, bridging_scores.shape)
    W, A1 = np.meshgrid(w_range, a1_range, indexing='ij')
    print(f"  Max at: w={W[max_i, max_j]:.2f}, a₁={A1[max_i, max_j]:.2f}, a₂={A2[max_i, max_j]:.2f}")

    # Static plot
    plot_3d_surface(
        x=w_range,
        y=a1_range,
        z=bridging_scores,
        title="Bridging Score: Unequal Groups\nConstraint: w·a₁ + (1-w)·a₂ = 0.5",
        xlabel="Group 1 Size Fraction (w)",
        ylabel="Group 1 Approval Rate (a₁)",
        zlabel="Bridging Score",
        output_path=output_dir / "unequal_groups_3d.png",
        mask=valid_mask,
    )

    # Interactive plot
    plot_3d_interactive(
        x=w_range,
        y=a1_range,
        z=bridging_scores,
        title="Bridging Score: Unequal Groups (w·a₁ + (1-w)·a₂ = 0.5)",
        xlabel="Group 1 Size Fraction (w)",
        ylabel="Group 1 Approval Rate (a₁)",
        zlabel="Bridging Score",
        output_path=output_dir / "unequal_groups_interactive.html",
        mask=valid_mask,
    )

    return bridging_scores, valid_mask


def main():
    """Run experimental scenarios."""
    print("=" * 60)
    print("Experiment 4: Synthetic Elections with Group Structure")
    print("=" * 60)

    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Parameters
    n_base_comments = 1000  # Only affects d_between ≈ 1000/1001 ≈ 0.999
    grid_resolution = 50

    # Run scenarios
    run_scenario_1_equal_groups(
        n_base_comments=n_base_comments,
        grid_resolution=grid_resolution,
        output_dir=output_dir,
    )

    run_scenario_2_unequal_groups(
        n_base_comments=n_base_comments,
        grid_resolution=grid_resolution,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print("All scenarios complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
