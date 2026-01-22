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

from experiment_2.bridging import (
    compute_bridging_scores_vectorized,
    compute_bridging_pnorm,
    compute_bridging_harmonic_pd,
)
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
    metric_name: str = 'PD Bridging Score',
) -> None:
    """
    Create scatter plot colored by bridging score.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filter out NaN values
    valid_mask = ~np.isnan(diversity) & ~np.isnan(bridging_scores)
    valid_rates = rates[valid_mask]
    valid_diversity = diversity[valid_mask]
    valid_bridging = bridging_scores[valid_mask]

    if len(valid_bridging) == 0:
        plt.close()
        return

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
    cbar.set_label(metric_name, fontsize=10)

    ax.set_xlabel('Approver Diversity (Hamming Distance)', fontsize=12)
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(f'{title}\n(colored by {metric_name})', fontsize=12)

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
    all_scores: dict,
    title: str,
    output_path: Path,
    item_names: list = None,
) -> None:
    """
    Plot showing which comment each method ranks as most bridging.

    Args:
        all_scores: dict mapping metric name to scores array
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter out NaN values
    valid_mask = ~np.isnan(diversity)
    valid_indices = np.where(valid_mask)[0]
    valid_rates = rates[valid_mask]
    valid_diversity = diversity[valid_mask]

    # Plot all points in grey
    ax.scatter(valid_diversity, valid_rates, c='lightgrey', alpha=0.5,
               edgecolors='none', s=40, label='All items')

    # Colors and markers for each metric
    metric_styles = {
        'PD Bridging': {'color': '#1b9e77', 'marker': 'o'},
        'p-norm (p=-10)': {'color': '#d95f02', 'marker': 's'},
        'p-norm (p=0)': {'color': '#7570b3', 'marker': '^'},
        'Harmonic PD': {'color': '#e7298a', 'marker': 'D'},
        'Polis': {'color': '#66a61e', 'marker': 'v'},
    }

    stats_lines = []
    plotted_positions = set()

    for metric_name, scores in all_scores.items():
        valid_scores = scores[valid_mask]
        valid_scores_clean = np.where(np.isnan(valid_scores), -np.inf, valid_scores)

        if np.all(np.isinf(valid_scores_clean)):
            continue

        # Find top-1 for this metric
        top_valid_idx = np.argmax(valid_scores_clean)
        top_idx = valid_indices[top_valid_idx]

        # Get name
        if item_names:
            name = item_names[top_idx]
        else:
            name = f"Item {top_idx}"

        style = metric_styles.get(metric_name, {'color': 'black', 'marker': 'x'})

        # Slight offset if position already plotted
        x_pos = diversity[top_idx]
        y_pos = rates[top_idx]
        pos_key = (round(x_pos, 3), round(y_pos, 3))
        offset = 0
        while pos_key in plotted_positions:
            offset += 0.015
            pos_key = (round(x_pos + offset, 3), round(y_pos, 3))
        plotted_positions.add(pos_key)

        ax.scatter(
            [x_pos + offset], [y_pos],
            c=style['color'], s=150, marker=style['marker'],
            edgecolors='black', linewidths=1.5,
            label=f'{metric_name}: {name}', zorder=10
        )

        stats_lines.append(
            f"{metric_name}: {name} (appr={rates[top_idx]:.2f}, div={diversity[top_idx]:.2f})"
        )

    ax.set_xlabel('Approver Diversity (Hamming Distance)', fontsize=12)
    ax.set_ylabel('Approval Rate', fontsize=12)
    ax.set_title(f'{title}\nTop Bridging Item by Metric', fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    # Add stats box
    stats_text = '\n'.join(stats_lines)
    ax.annotate(stats_text, xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=7, ha='right', va='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

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

    # Define data sources (excluding 00069 Pol.is - uses imputed data)
    data_sources = [
        ('data/processed/preflib', '00026-*.npz', 'French Election 2002'),
        ('data/processed/preflib', '00033-*.npz', 'San Sebastian Poster'),
        ('data/processed/preflib', '00063-*.npz', 'CTU Tutorial'),
        ('data/processed/preflib', '00071-*.npz', 'French Election 2007'),
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

        # Compute all bridging scores
        pd_bridging = compute_bridging_scores_vectorized(matrix)
        pnorm_min = compute_bridging_pnorm(matrix, p=-10)
        pnorm_geo = compute_bridging_pnorm(matrix, p=0)
        harmonic_pd = compute_bridging_harmonic_pd(matrix)

        # Compute Polis consensus scores
        observed_mask = np.ones_like(matrix, dtype=bool)  # Fully observed
        try:
            polis_scores, _ = polis_consensus_pipeline(matrix, observed_mask)
        except Exception as e:
            print(f"  Polis failed for {file_id}: {e}")
            polis_scores = np.full(n_items, np.nan)

        # Collect all scores for comparison plot
        all_scores = {
            'PD Bridging': pd_bridging,
            'p-norm (p=-10)': pnorm_min,
            'p-norm (p=0)': pnorm_geo,
            'Harmonic PD': harmonic_pd,
            'Polis': polis_scores,
        }

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

        # Plot 2: Colored by each metric (create subdirectories)
        metrics_to_plot = [
            ('pd_bridging', pd_bridging, 'PD Bridging'),
            ('pnorm_min', pnorm_min, 'p-norm (p=-10)'),
            ('pnorm_geo', pnorm_geo, 'p-norm (p=0)'),
            ('harmonic_pd', harmonic_pd, 'Harmonic PD'),
            ('polis', polis_scores, 'Polis Consensus'),
        ]

        for metric_key, scores, metric_label in metrics_to_plot:
            metric_dir = colored_dir / metric_key
            metric_dir.mkdir(parents=True, exist_ok=True)
            plot_approval_vs_diversity_colored(
                rates, diversity, scores, title,
                metric_dir / f'{file_id}.png',
                metric_name=metric_label
            )

        # Plot 3: Top comments comparison (all metrics)
        plot_top_comments_comparison(
            rates, diversity, all_scores, title,
            comparison_dir / f'{file_id}.png',
            item_names=item_names
        )

    print(f"\nPlots saved to:")
    print(f"  Plain: {plain_dir}/")
    print(f"  Colored by metric: {colored_dir}/<metric>/")
    print(f"  Method comparison: {comparison_dir}/")


if __name__ == '__main__':
    main()
