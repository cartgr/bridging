"""
Experiment 1: Approval Rate vs. Approver Diversity

Plot each comment/item by its approval rate (x-axis) and the diversity of
its approvers (y-axis). Diversity is measured by average pairwise Hamming
distance between approvers' voting vectors.

Two versions:
1. Plain scatter plot
2. Scatter plot colored by PD bridging score (viridis colormap)

Also shows which comment each method (PD vs Polis) ranks as most bridging.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from glob import glob
from tqdm import tqdm

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_5.polis import polis_consensus_pipeline


def compute_approval_rates(matrix: np.ndarray) -> np.ndarray:
    """
    Compute approval rate for each item (row).

    Args:
        matrix: Shape (n_items, n_voters), values 1.0 (approved) or 0.0 (disapproved)

    Returns:
        Array of approval rates for each item
    """
    return np.mean(matrix, axis=1)


def compute_approver_diversity(matrix: np.ndarray, metric: str = 'hamming') -> np.ndarray:
    """
    Compute diversity of approvers for each item.

    For each item, find all approvers and compute the average pairwise
    Hamming distance between their full voting vectors.

    Uses O(m) formula instead of O(n²) pairwise computation:
    Mean Hamming = sum_j[k_j * (n - k_j)] / [n*(n-1)/2 * m]
    where k_j = count of 1s at position j, n = num approvers, m = num items

    Args:
        matrix: Shape (n_items, n_voters), values 1.0 (approved) or 0.0 (disapproved)
        metric: Distance metric to use (currently only 'hamming' supported)

    Returns:
        Array of diversity scores for each item
    """
    n_items = matrix.shape[0]
    diversity = np.zeros(n_items)

    iterator = range(n_items)
    if n_items > 100:
        iterator = tqdm(iterator, desc="Computing diversity", leave=False)

    for i in iterator:
        approvers = np.where(matrix[i, :] == 1.0)[0]
        n = len(approvers)

        if n < 2:
            diversity[i] = np.nan
            continue

        # Get voting vectors for all approvers
        approver_vectors = matrix[:, approvers]  # shape: (n_items, n_approvers)

        # Count 1s at each position
        k = np.sum(approver_vectors, axis=1)  # shape: (n_items,)

        # Total disagreements: sum of k_j * (n - k_j) for each position j
        total_disagreements = np.sum(k * (n - k))

        # Number of pairs and normalize by vector length
        n_pairs = n * (n - 1) / 2
        mean_hamming = total_disagreements / (n_pairs * n_items)

        diversity[i] = mean_hamming

    return diversity


def plot_approval_vs_diversity(
    rates: np.ndarray,
    diversity: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """
    Create scatter plot of approval rate vs approver diversity.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(diversity)
    valid_rates = rates[valid_mask]
    valid_diversity = diversity[valid_mask]

    ax.scatter(valid_diversity, valid_rates, alpha=0.6, edgecolors='none', s=50)

    ax.set_xlabel('Approver Diversity (Hamming Distance)', fontsize=12)
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(title, fontsize=14)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)

    n_valid = len(valid_rates)
    n_total = len(rates)
    stats_text = f'n={n_valid} items'
    if n_valid < n_total:
        stats_text += f' ({n_total - n_valid} excluded: <2 approvers)'
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_approval_vs_diversity_colored(
    rates: np.ndarray,
    diversity: np.ndarray,
    bridging_scores: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """
    Create scatter plot colored by PD bridging score.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filter out NaN values
    valid_mask = ~np.isnan(diversity)
    valid_rates = rates[valid_mask]
    valid_diversity = diversity[valid_mask]
    valid_bridging = bridging_scores[valid_mask]

    # Color by bridging score using viridis
    norm = Normalize(vmin=valid_bridging.min(), vmax=valid_bridging.max())
    scatter = ax.scatter(
        valid_diversity, valid_rates,
        c=valid_bridging, cmap='viridis',
        alpha=0.7, edgecolors='none', s=60,
        norm=norm
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('PD Bridging Score', fontsize=10)

    ax.set_xlabel('Approver Diversity (Hamming Distance)', fontsize=12)
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(f'{title}\n(colored by PD bridging score)', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    n_valid = len(valid_rates)
    ax.annotate(f'n={n_valid} items', xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_comments_comparison(
    rates: np.ndarray,
    diversity: np.ndarray,
    bridging_scores: np.ndarray,
    polis_scores: np.ndarray,
    title: str,
    output_path: Path,
    item_names: list = None,
) -> None:
    """
    Plot showing which comment each method ranks as most bridging.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filter out NaN values
    valid_mask = ~np.isnan(diversity)
    valid_indices = np.where(valid_mask)[0]
    valid_rates = rates[valid_mask]
    valid_diversity = diversity[valid_mask]
    valid_bridging = bridging_scores[valid_mask]
    valid_polis = polis_scores[valid_mask]

    # Plot all points in grey
    ax.scatter(valid_diversity, valid_rates, c='lightgrey', alpha=0.5,
               edgecolors='none', s=40, label='All items')

    # Find top-1 for each method (among valid items)
    pd_top_valid_idx = np.argmax(valid_bridging)
    polis_top_valid_idx = np.argmax(valid_polis)

    # Map back to original indices
    pd_top_idx = valid_indices[pd_top_valid_idx]
    polis_top_idx = valid_indices[polis_top_valid_idx]

    # Get names if available
    if item_names:
        pd_name = item_names[pd_top_idx]
        polis_name = item_names[polis_top_idx]
    else:
        pd_name = f"Item {pd_top_idx}"
        polis_name = f"Item {polis_top_idx}"

    # Plot PD top (green/teal)
    ax.scatter(
        [diversity[pd_top_idx]], [rates[pd_top_idx]],
        c='#1b9e77', s=200, marker='o', edgecolors='black', linewidths=2,
        label=f'PD Top: {pd_name}', zorder=10
    )

    # Plot Polis top (purple)
    ax.scatter(
        [diversity[polis_top_idx]], [rates[polis_top_idx]],
        c='#7570b3', s=200, marker='s', edgecolors='black', linewidths=2,
        label=f'Polis Top: {polis_name}', zorder=10
    )

    # If same comment, add note
    if pd_top_idx == polis_top_idx:
        ax.annotate('(Same comment)', xy=(0.98, 0.02), xycoords='axes fraction',
                    fontsize=10, ha='right', style='italic')

    ax.set_xlabel('Approver Diversity (Hamming Distance)', fontsize=12)
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(f'{title}\nTop Bridging Comment: PD vs Polis', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)

    # Add stats
    stats = (
        f"PD top: approval={rates[pd_top_idx]:.2f}, diversity={diversity[pd_top_idx]:.2f}\n"
        f"Polis top: approval={rates[polis_top_idx]:.2f}, diversity={diversity[polis_top_idx]:.2f}"
    )
    ax.annotate(stats, xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=8, ha='right', va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_matrix(filepath: Path) -> np.ndarray:
    """Load matrix from npz file."""
    data = np.load(filepath)
    if 'matrix' in data:
        return data['matrix']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise KeyError(f"No 'matrix' or 'arr_0' key in {filepath}")


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


def main():
    """Process all datasets and generate plots."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'plots'

    # Create subdirectories
    plain_dir = output_dir / 'plain'
    colored_dir = output_dir / 'colored'
    comparison_dir = output_dir / 'comparison'

    for d in [plain_dir, colored_dir, comparison_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Define data sources
    data_sources = [
        ('data/processed/preflib', '00026-*.npz', 'French Election 2002'),
        ('data/processed/preflib', '00033-*.npz', 'San Sebastian Poster'),
        ('data/processed/preflib', '00063-*.npz', 'CTU Tutorial'),
        ('data/processed/preflib', '00071-*.npz', 'French Election 2007'),
        ('data/completed', '00069-*.npz', 'Pol.is (Completed)'),
    ]

    all_files = []
    for data_path, pattern, dataset_name in data_sources:
        full_pattern = str(base_dir / data_path / pattern)
        files = sorted(glob(full_pattern))
        all_files.extend([(Path(f), dataset_name) for f in files])

    for filepath, dataset_name in tqdm(all_files, desc="Processing files"):
        file_id = filepath.stem

        # Load matrix
        matrix = load_matrix(filepath)
        n_items, n_voters = matrix.shape

        # Skip files with missing values (Polis pipeline requires complete data)
        if np.isnan(matrix).any():
            print(f"  Skipping {file_id}: contains NaN values")
            continue

        # Compute metrics
        rates = compute_approval_rates(matrix)
        diversity = compute_approver_diversity(matrix)

        # Compute PD bridging scores
        bridging_scores = compute_bridging_scores_vectorized(matrix)

        # Compute Polis consensus scores
        observed_mask = np.ones_like(matrix, dtype=bool)  # Fully observed
        polis_scores, _ = polis_consensus_pipeline(matrix, observed_mask)

        # Get item names
        if '00026' in file_id:
            item_names = CANDIDATE_NAMES_2002[:n_items]
        elif '00071' in file_id:
            item_names = CANDIDATE_NAMES_2007[:n_items]
        else:
            item_names = [f'Item {i}' for i in range(n_items)]

        title = f'{dataset_name}: {file_id}'

        # Plot 1: Plain scatter
        plot_approval_vs_diversity(
            rates, diversity, title,
            plain_dir / f'{file_id}.png'
        )

        # Plot 2: Colored by bridging score
        plot_approval_vs_diversity_colored(
            rates, diversity, bridging_scores, title,
            colored_dir / f'{file_id}.png'
        )

        # Plot 3: Top comments comparison
        plot_top_comments_comparison(
            rates, diversity, bridging_scores, polis_scores, title,
            comparison_dir / f'{file_id}.png',
            item_names=item_names
        )

    print(f"\nPlots saved to:")
    print(f"  Plain: {plain_dir}/")
    print(f"  Colored by PD: {colored_dir}/")
    print(f"  Method comparison: {comparison_dir}/")


if __name__ == '__main__':
    main()
