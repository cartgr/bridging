"""
Experiment 3: Voter PCA Spectrum Visualization

Visualizes which voters (positioned on a left-right political spectrum via PCA)
approve each candidate in the French Election data using ridgeline plots.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from pathlib import Path
from glob import glob
from scipy.stats import gaussian_kde
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_2.bridging import (
    compute_bridging_scores_vectorized,
    compute_bridging_pnorm,
    compute_bridging_harmonic_pd,
)
from experiment_5.polis import polis_consensus_pipeline


# French election 2002 candidate names (00026)
CANDIDATE_NAMES_2002 = [
    'Megret', 'Lepage', 'Gluckstein', 'Bayrou', 'Chirac',
    'LePen', 'Taubira', 'Saint-Josse', 'Mamere', 'Jospin',
    'Boutin', 'Hue', 'Chevenement', 'Madelin', 'Laguiller', 'Besancenot'
]

# French election 2007 candidate names (00071)
CANDIDATE_NAMES_2007 = [
    'Besancenot', 'Buffet', 'Schivardi', 'Bayrou', 'Bove',
    'Voynet', 'Villiers', 'Royal', 'Nihous', 'Le Pen',
    'Sarkozy', 'Laguiller'
]

# Keep backward compatibility
CANDIDATE_NAMES = CANDIDATE_NAMES_2002


def compute_voter_pca_scores(matrix: np.ndarray) -> np.ndarray:
    """
    Compute PC1 scores for each voter.

    Args:
        matrix: (n_candidates, n_voters) approval matrix

    Returns:
        (n_voters,) array of PC1 scores
    """
    # Transpose so voters are rows
    voter_matrix = matrix.T  # (n_voters, n_candidates)
    centered = voter_matrix - voter_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return U[:, 0] * S[0]


def compute_unnormalized_density(
    x_values: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float = None,
) -> np.ndarray:
    """
    Compute unnormalized density (smoothed histogram) that preserves count info.

    Args:
        x_values: Data points
        x_grid: Grid points to evaluate density at
        bandwidth: KDE bandwidth (None for automatic)

    Returns:
        Unnormalized density values at x_grid points
    """
    if len(x_values) < 2:
        return np.zeros_like(x_grid)

    # Use KDE but scale by count instead of normalizing
    kde = gaussian_kde(x_values, bw_method=bandwidth)
    # KDE integrates to 1, multiply by count to get unnormalized density
    density = kde(x_grid) * len(x_values)
    return density


def load_matrix(filepath: Path) -> np.ndarray:
    """Load matrix from npz file."""
    data = np.load(filepath)
    if 'matrix' in data:
        return data['matrix']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise KeyError(f"No 'matrix' or 'arr_0' key in {filepath}")


def plot_voter_spectrum(
    matrix: np.ndarray,
    pc1_scores: np.ndarray,
    bridging_scores: np.ndarray,
    candidate_names: list[str],
    title: str,
    output_path: Path,
    fig_height: float = 12,
    metric_name: str = 'Bridging',
) -> None:
    """
    Create ridgeline visualization for one polling station.

    Each row shows one candidate as a filled density curve where:
    - X position = voter's PC1 score (political spectrum)
    - Height = unnormalized density of approvers (preserves count info)
    - Fill color = gradient from blue (left) to red (right)

    Args:
        matrix: (n_candidates, n_voters) approval matrix
        pc1_scores: (n_voters,) PC1 scores for voter positioning
        bridging_scores: (n_candidates,) bridging scores
        candidate_names: List of candidate/item names
        title: Plot title
        output_path: Where to save the plot
        fig_height: Figure height in inches
        metric_name: Name of the bridging metric for column header
    """
    n_candidates = matrix.shape[0]
    n_voters = matrix.shape[1]

    # Sort candidates by bridging score (descending)
    sorted_indices = np.argsort(bridging_scores)[::-1]

    # Set up x grid for density estimation
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    x_margin = (pc1_max - pc1_min) * 0.1
    x_grid = np.linspace(pc1_min - x_margin, pc1_max + x_margin, 200)

    # Compute all densities first to find global max for scaling
    densities = []
    for cand_idx in range(n_candidates):
        approvers = matrix[cand_idx] == 1.0
        x_approvers = pc1_scores[approvers]
        density = compute_unnormalized_density(x_approvers, x_grid, bandwidth=0.3)
        densities.append(density)

    # Scale factor: max density should take up ~80% of row height
    max_density = max(d.max() for d in densities) if densities else 1.0
    row_height = 1.0
    scale_factor = (row_height * 0.8) / max_density if max_density > 0 else 1.0

    # Create figure
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set up colormap for fill (blue=left, red=right)
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=pc1_min, vmax=pc1_max)

    # Plot each candidate as a ridge
    for row_idx, cand_idx in enumerate(sorted_indices):
        y_base = n_candidates - 1 - row_idx  # Bottom candidate at y=0
        density = densities[cand_idx] * scale_factor

        # Create filled polygon with gradient color
        # We'll approximate the gradient by drawing many thin vertical strips
        for i in range(len(x_grid) - 1):
            x_left, x_right = x_grid[i], x_grid[i + 1]
            y_left, y_right = density[i], density[i + 1]

            if y_left < 0.001 and y_right < 0.001:
                continue

            # Color based on x position (midpoint)
            x_mid = (x_left + x_right) / 2
            color = cmap(norm(x_mid))

            # Draw filled trapezoid
            verts = [
                (x_left, y_base),
                (x_left, y_base + y_left),
                (x_right, y_base + y_right),
                (x_right, y_base),
            ]
            poly = plt.Polygon(verts, facecolor=color, edgecolor='none', alpha=0.7)
            ax.add_patch(poly)

        # Add outline
        ax.plot(x_grid, y_base + density, color='black', linewidth=0.5, alpha=0.5)

        # Add baseline
        ax.axhline(y=y_base, color='grey', linewidth=0.3, alpha=0.3)

        # Add vertical line at x=0 for this row
        ax.plot([0, 0], [y_base, y_base + row_height * 0.8],
                color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

        # Add candidate name on left
        name = candidate_names[cand_idx]
        ax.text(
            pc1_min - x_margin - 0.1, y_base + row_height * 0.3,
            name,
            fontsize=10,
            fontweight='bold',
            ha='right',
            va='center',
        )

        # Add bridging score on right
        score = bridging_scores[cand_idx]
        ax.text(
            pc1_max + x_margin + 0.1, y_base + row_height * 0.3,
            f'{score:.3f}',
            fontsize=10,
            ha='left',
            va='center',
            family='monospace',
        )

        # Add approval fraction
        approval_frac = matrix[cand_idx].mean()
        ax.text(
            pc1_max + x_margin + 0.9, y_base + row_height * 0.3,
            f'{approval_frac:.0%}',
            fontsize=10,
            ha='left',
            va='center',
            family='monospace',
        )

    # Format axes
    ax.set_xlim(pc1_min - x_margin - 1.5, pc1_max + x_margin + 1.5)
    ax.set_ylim(-0.5, n_candidates + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('PC1 Score (Left ← → Right)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add column headers
    # Use "Item" if names look auto-generated, otherwise "Candidate"
    item_label = 'Item' if candidate_names[0].startswith('Item ') else 'Candidate'
    ax.text(pc1_min - x_margin - 0.8, n_candidates + 0.3, item_label,
            fontsize=11, fontweight='bold', ha='center')
    ax.text((pc1_min + pc1_max) / 2, n_candidates + 0.3, 'Voter Approval Distribution',
            fontsize=11, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 0.4, n_candidates + 0.3, metric_name,
            fontsize=11, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 1.1, n_candidates + 0.3, 'Appr',
            fontsize=11, fontweight='bold', ha='center')

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location='right', shrink=0.3, pad=0.12)
    cbar.set_label('PC1 Score', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_voter_spectrum_histogram(
    matrix: np.ndarray,
    pc1_scores: np.ndarray,
    bridging_scores: np.ndarray,
    candidate_names: list[str],
    title: str,
    output_path: Path,
    fig_height: float = 12,
    n_bins: int = 30,
) -> None:
    """
    Create histogram-based visualization showing approval rates across spectrum.

    Each row shows one candidate as a histogram where:
    - Bar height = number of voters in that bin
    - Colored portion = approval rate (blue = approved)
    - Grey portion = disapproval rate

    Args:
        matrix: (n_candidates, n_voters) approval matrix
        pc1_scores: (n_voters,) PC1 scores for voter positioning
        bridging_scores: (n_candidates,) bridging scores
        candidate_names: List of candidate/item names
        title: Plot title
        output_path: Where to save the plot
        fig_height: Figure height in inches
        n_bins: Number of histogram bins
    """
    n_candidates = matrix.shape[0]

    # Sort candidates by bridging score (descending)
    sorted_indices = np.argsort(bridging_scores)[::-1]

    # Set up bins
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    x_margin = (pc1_max - pc1_min) * 0.05
    bin_edges = np.linspace(pc1_min - x_margin, pc1_max + x_margin, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each voter to a bin
    voter_bins = np.digitize(pc1_scores, bin_edges) - 1
    voter_bins = np.clip(voter_bins, 0, n_bins - 1)

    # Count voters per bin (same for all candidates)
    voters_per_bin = np.bincount(voter_bins, minlength=n_bins)

    # Find max for scaling
    max_voters = voters_per_bin.max()
    row_height = 1.0
    scale_factor = (row_height * 0.85) / max_voters if max_voters > 0 else 1.0

    # Create figure
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Plot each candidate as a row of histogram bars
    for row_idx, cand_idx in enumerate(sorted_indices):
        y_base = n_candidates - 1 - row_idx  # Bottom candidate at y=0

        # Count approvers per bin for this candidate
        approvals = matrix[cand_idx]
        approvers_per_bin = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            voters_in_bin = voter_bins == bin_idx
            if voters_in_bin.sum() > 0:
                approvers_per_bin[bin_idx] = approvals[voters_in_bin].sum()

        # Draw bars
        for bin_idx in range(n_bins):
            n_voters_bin = voters_per_bin[bin_idx]
            n_approvers_bin = approvers_per_bin[bin_idx]

            if n_voters_bin == 0:
                continue

            bar_height = n_voters_bin * scale_factor
            approval_rate = n_approvers_bin / n_voters_bin
            approval_height = bar_height * approval_rate

            x_left = bin_edges[bin_idx]

            # Draw grey (disapproval) portion on top
            if approval_height < bar_height:
                ax.add_patch(plt.Rectangle(
                    (x_left, y_base + approval_height),
                    bin_width,
                    bar_height - approval_height,
                    facecolor='lightgrey',
                    edgecolor='none',
                    alpha=0.8,
                ))

            # Draw blue (approval) portion on bottom
            if approval_height > 0:
                ax.add_patch(plt.Rectangle(
                    (x_left, y_base),
                    bin_width,
                    approval_height,
                    facecolor='steelblue',
                    edgecolor='none',
                    alpha=0.8,
                ))

            # Add thin outline for the full bar
            ax.add_patch(plt.Rectangle(
                (x_left, y_base),
                bin_width,
                bar_height,
                facecolor='none',
                edgecolor='grey',
                linewidth=0.3,
                alpha=0.5,
            ))

        # Add baseline
        ax.axhline(y=y_base, color='grey', linewidth=0.3, alpha=0.3)

        # Add vertical line at x=0
        ax.plot([0, 0], [y_base, y_base + row_height * 0.85],
                color='grey', linestyle='--', linewidth=0.5, alpha=0.3)

        # Add candidate name on left
        name = candidate_names[cand_idx]
        ax.text(
            pc1_min - x_margin - 0.1, y_base + row_height * 0.3,
            name,
            fontsize=10,
            fontweight='bold',
            ha='right',
            va='center',
        )

        # Add bridging score on right
        score = bridging_scores[cand_idx]
        ax.text(
            pc1_max + x_margin + 0.1, y_base + row_height * 0.3,
            f'{score:.3f}',
            fontsize=10,
            ha='left',
            va='center',
            family='monospace',
        )

        # Add approval fraction
        approval_frac = matrix[cand_idx].mean()
        ax.text(
            pc1_max + x_margin + 0.9, y_base + row_height * 0.3,
            f'{approval_frac:.0%}',
            fontsize=10,
            ha='left',
            va='center',
            family='monospace',
        )

    # Format axes
    ax.set_xlim(pc1_min - x_margin - 1.5, pc1_max + x_margin + 1.5)
    ax.set_ylim(-0.5, n_candidates + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('PC1 Score (Left ← → Right)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add column headers
    item_label = 'Item' if candidate_names[0].startswith('Item ') else 'Candidate'
    ax.text(pc1_min - x_margin - 0.8, n_candidates + 0.3, item_label,
            fontsize=11, fontweight='bold', ha='center')
    ax.text((pc1_min + pc1_max) / 2, n_candidates + 0.3, 'Approval Rate by Position',
            fontsize=11, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 0.4, n_candidates + 0.3, 'Bridging',
            fontsize=11, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 1.1, n_candidates + 0.3, 'Appr',
            fontsize=11, fontweight='bold', ha='center')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Approved'),
        Patch(facecolor='lightgrey', alpha=0.8, label='Not Approved'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset_histogram(
    filepath: Path,
    output_dir: Path,
    dataset_name: str,
    item_names: Optional[list] = None,
    max_items: Optional[int] = None,
) -> None:
    """
    Process a single dataset file and generate histogram plot.

    Args:
        filepath: Path to the .npz file
        output_dir: Directory to save output plot
        dataset_name: Name for the plot title (e.g., "French Election", "Pol.is")
        item_names: Optional list of item names. If None, uses "Item 1", "Item 2", etc.
        max_items: Maximum number of items to display (top by bridging score). None for all.
    """
    file_id = filepath.stem

    # Load matrix
    matrix = load_matrix(filepath)
    n_items, n_voters = matrix.shape
    print(f"  {file_id}: {n_items} items, {n_voters} voters")

    # Compute PC1 scores for voters
    pc1_scores = compute_voter_pca_scores(matrix)

    # Compute bridging scores
    bridging_scores = compute_bridging_scores_vectorized(matrix)

    # Generate item names if not provided
    if item_names is None:
        item_names = [f'Item {i+1}' for i in range(n_items)]

    # Limit to top N items if specified
    if max_items is not None and n_items > max_items:
        # Get indices of top items by bridging score
        top_indices = np.argsort(bridging_scores)[::-1][:max_items]
        matrix = matrix[top_indices]
        bridging_scores = bridging_scores[top_indices]
        item_names = [item_names[i] for i in top_indices]
        n_items = max_items
        print(f"    (showing top {max_items} items by bridging score)")

    # Adjust figure height based on number of items
    fig_height = max(8, n_items * 0.6)

    # Generate plot
    title = f'{dataset_name}: {file_id}'
    output_path = output_dir / f'{file_id}.png'
    plot_voter_spectrum_histogram(
        matrix,
        pc1_scores,
        bridging_scores,
        item_names,
        title,
        output_path,
        fig_height=fig_height,
    )
    print(f"    Saved: {output_path}")


def process_dataset(
    filepath: Path,
    output_dir: Path,
    dataset_name: str,
    item_names: Optional[list] = None,
    max_items: Optional[int] = None,
) -> None:
    """
    Process a single dataset file and generate ridgeline plots for all metrics.

    Args:
        filepath: Path to the .npz file
        output_dir: Directory to save output plots (will create metric subdirectories)
        dataset_name: Name for the plot title (e.g., "French Election")
        item_names: Optional list of item names. If None, uses "Item 1", "Item 2", etc.
        max_items: Maximum number of items to display (top by bridging score). None for all.
    """
    file_id = filepath.stem

    # Load matrix
    matrix = load_matrix(filepath)
    n_items, n_voters = matrix.shape
    print(f"  {file_id}: {n_items} items, {n_voters} voters")

    # Skip files with NaN values
    if np.isnan(matrix).any():
        print(f"    Skipping: contains NaN values")
        return

    # Compute PC1 scores for voters
    pc1_scores = compute_voter_pca_scores(matrix)

    # Compute all bridging metrics
    pd_bridging = compute_bridging_scores_vectorized(matrix)
    pnorm_min = compute_bridging_pnorm(matrix, p=-10)
    pnorm_geo = compute_bridging_pnorm(matrix, p=0)
    harmonic_pd = compute_bridging_harmonic_pd(matrix)

    # Compute Polis consensus scores
    observed_mask = np.ones_like(matrix, dtype=bool)
    try:
        polis_scores, _ = polis_consensus_pipeline(matrix, observed_mask)
    except Exception as e:
        print(f"    Polis failed: {e}")
        polis_scores = np.full(n_items, np.nan)

    # All metrics to plot
    metrics = [
        ('pd_bridging', pd_bridging, 'PD Bridging'),
        ('pnorm_min', pnorm_min, 'p-norm (p=-10)'),
        ('pnorm_geo', pnorm_geo, 'p-norm (p=0)'),
        ('harmonic_pd', harmonic_pd, 'Harmonic PD'),
        ('polis', polis_scores, 'Polis'),
    ]

    # Generate item names if not provided
    if item_names is None:
        item_names_list = [f'Item {i+1}' for i in range(n_items)]
    else:
        item_names_list = item_names[:n_items]

    # Generate plots for each metric
    for metric_key, scores, metric_label in metrics:
        if np.isnan(scores).all():
            continue

        # Create metric subdirectory
        metric_dir = output_dir / metric_key
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Handle NaN scores by replacing with -inf for sorting
        scores_for_sort = np.where(np.isnan(scores), -np.inf, scores)

        # Limit to top N items if specified
        if max_items is not None and n_items > max_items:
            top_indices = np.argsort(scores_for_sort)[::-1][:max_items]
            matrix_subset = matrix[top_indices]
            scores_subset = scores[top_indices]
            names_subset = [item_names_list[i] for i in top_indices]
            n_display = max_items
        else:
            matrix_subset = matrix
            scores_subset = scores
            names_subset = item_names_list
            n_display = n_items

        # Adjust figure height
        fig_height = max(8, n_display * 0.6)

        # Generate plot
        title = f'{dataset_name}: {file_id}'
        output_path = metric_dir / f'{file_id}.png'
        plot_voter_spectrum(
            matrix_subset,
            pc1_scores,
            scores_subset,
            names_subset,
            title,
            output_path,
            fig_height=fig_height,
            metric_name=metric_label,
        )

    print(f"    Saved plots for {len(metrics)} metrics")


def main():
    """Generate plots for all approval voting datasets (excluding Pol.is 00069)."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    preflib_dir = base_dir / 'data' / 'processed' / 'preflib'

    # Process French election 2002 files (00026)
    french_2002_files = sorted(glob(str(preflib_dir / '00026-*.npz')))
    if french_2002_files:
        print(f"Processing {len(french_2002_files)} French Election 2002 files...")
        for filepath in french_2002_files:
            process_dataset(
                Path(filepath),
                output_dir,
                'French Election 2002',
                item_names=CANDIDATE_NAMES_2002,
            )

    # Process San Sebastian Poster files (00033)
    poster_files = sorted(glob(str(preflib_dir / '00033-*.npz')))
    if poster_files:
        print(f"\nProcessing {len(poster_files)} San Sebastian Poster files...")
        for filepath in poster_files:
            process_dataset(
                Path(filepath),
                output_dir,
                'San Sebastian Poster',
                item_names=None,  # Auto-generate
            )

    # Process CTU Tutorial files (00063)
    ctu_files = sorted(glob(str(preflib_dir / '00063-*.npz')))
    if ctu_files:
        print(f"\nProcessing {len(ctu_files)} CTU Tutorial files...")
        for filepath in ctu_files:
            process_dataset(
                Path(filepath),
                output_dir,
                'CTU Tutorial',
                item_names=None,  # Auto-generate
            )

    # Process French election 2007 files (00071) - only fully observed ones (1-6)
    french_2007_files = []
    for i in range(1, 7):
        f = preflib_dir / f'00071-{i:08d}.npz'
        if f.exists():
            french_2007_files.append(f)
    if french_2007_files:
        print(f"\nProcessing {len(french_2007_files)} French Election 2007 files...")
        for filepath in french_2007_files:
            process_dataset(
                Path(filepath),
                output_dir,
                'French Election 2007',
                item_names=CANDIDATE_NAMES_2007,
            )

    print(f"\nAll plots saved to: {output_dir}")


def main_histogram():
    """Generate histogram plots for all approval voting datasets (excluding Pol.is 00069)."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'plots_histogram'
    output_dir.mkdir(parents=True, exist_ok=True)

    preflib_dir = base_dir / 'data' / 'processed' / 'preflib'

    # Process French election 2002 files (00026)
    french_2002_files = sorted(glob(str(preflib_dir / '00026-*.npz')))
    if french_2002_files:
        print(f"Processing {len(french_2002_files)} French Election 2002 files...")
        for filepath in french_2002_files:
            process_dataset_histogram(
                Path(filepath),
                output_dir,
                'French Election 2002',
                item_names=CANDIDATE_NAMES_2002,
            )

    # Process San Sebastian Poster files (00033)
    poster_files = sorted(glob(str(preflib_dir / '00033-*.npz')))
    if poster_files:
        print(f"\nProcessing {len(poster_files)} San Sebastian Poster files...")
        for filepath in poster_files:
            process_dataset_histogram(
                Path(filepath),
                output_dir,
                'San Sebastian Poster',
                item_names=None,
            )

    # Process CTU Tutorial files (00063)
    ctu_files = sorted(glob(str(preflib_dir / '00063-*.npz')))
    if ctu_files:
        print(f"\nProcessing {len(ctu_files)} CTU Tutorial files...")
        for filepath in ctu_files:
            process_dataset_histogram(
                Path(filepath),
                output_dir,
                'CTU Tutorial',
                item_names=None,
            )

    # Process French election 2007 files (00071) - only fully observed ones (1-6)
    french_2007_files = []
    for i in range(1, 7):
        f = preflib_dir / f'00071-{i:08d}.npz'
        if f.exists():
            french_2007_files.append(f)
    if french_2007_files:
        print(f"\nProcessing {len(french_2007_files)} French Election 2007 files...")
        for filepath in french_2007_files:
            process_dataset_histogram(
                Path(filepath),
                output_dir,
                'French Election 2007',
                item_names=CANDIDATE_NAMES_2007,
            )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--histogram':
        main_histogram()
    else:
        main()
