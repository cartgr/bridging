"""
Experiment 1: Approval Rate vs. Approver Diversity

Plot each comment/item by its approval rate (x-axis) and the diversity of
its approvers (y-axis). Diversity is measured by average pairwise Hamming
distance between approvers' voting vectors.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from tqdm import tqdm


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

    # Use tqdm for large matrices
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

    Args:
        rates: Approval rates for each item
        diversity: Diversity scores for each item
        title: Plot title
        output_path: Path to save the plot
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

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add stats annotation
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


def load_matrix(filepath: Path) -> np.ndarray:
    """Load matrix from npz file."""
    data = np.load(filepath)
    # Handle both 'matrix' and 'arr_0' keys
    if 'matrix' in data:
        return data['matrix']
    elif 'arr_0' in data:
        return data['arr_0']
    else:
        raise KeyError(f"No 'matrix' or 'arr_0' key in {filepath}")


def main():
    """Process all datasets and generate plots."""
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'plots' / 'hamming_distance'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define data sources
    data_sources = [
        ('data/processed/preflib', '00026-*.npz', 'French Election'),
        ('data/completed', '00069-*.npz', 'Pol.is (Completed)'),
    ]

    all_files = []
    for data_path, pattern, dataset_name in data_sources:
        full_pattern = str(base_dir / data_path / pattern)
        files = sorted(glob(full_pattern))
        all_files.extend([(Path(f), dataset_name) for f in files])

    for filepath, dataset_name in tqdm(all_files, desc="Processing files"):
        file_id = filepath.stem  # e.g., '00026-00000001'

        # Load matrix
        matrix = load_matrix(filepath)

        # Compute metrics
        rates = compute_approval_rates(matrix)
        diversity = compute_approver_diversity(matrix)

        # Generate plot
        title = f'{dataset_name}: {file_id}'
        output_path = output_dir / f'{file_id}.png'
        plot_approval_vs_diversity(rates, diversity, title, output_path)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
