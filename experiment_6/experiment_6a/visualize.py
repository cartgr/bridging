"""
Experiment 6a: Bridging visualization on sparse Polis data (no imputation)

- Uses MDS on pairwise voter similarity to position voters
- Computes bridging scores naively from observed entries only
- Shows approval fraction among voters who actually voted on each item
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform
from scipy.stats import gaussian_kde
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_6.comments import load_all_comments_for_dataset
from experiment_5.polis import polis_consensus_pipeline
from experiment_2.estimation import estimate_bridging_scores_naive
from scipy import stats


def compute_voter_similarity_sparse(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise voter similarity using only shared observed items.

    Similarity = fraction of shared items where they agree.

    Args:
        matrix: (n_items, n_voters) with NaN for missing, 1=approve, 0=disapprove

    Returns:
        (n_voters, n_voters) similarity matrix
    """
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)

    similarity = np.zeros((n_voters, n_voters))

    for i in range(n_voters):
        for j in range(i, n_voters):
            # Items both voters rated
            shared = observed[:, i] & observed[:, j]
            n_shared = shared.sum()

            if n_shared == 0:
                # No shared items - use 0.5 (neutral)
                sim = 0.5
            else:
                # Agreement rate on shared items (compare actual values)
                agree = (matrix[shared, i] == matrix[shared, j]).sum()
                sim = agree / n_shared

            similarity[i, j] = sim
            similarity[j, i] = sim

    return similarity


def compute_voter_positions_mds(similarity: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Use MDS to get 1D voter positions from similarity matrix.

    Args:
        similarity: (n_voters, n_voters) similarity matrix
        seed: random seed

    Returns:
        (n_voters,) array of 1D positions
    """
    # Convert similarity to distance
    distance = 1 - similarity

    # Ensure diagonal is 0
    np.fill_diagonal(distance, 0)

    # MDS to 1D
    mds = MDS(n_components=1, dissimilarity='precomputed', random_state=seed)
    positions = mds.fit_transform(distance)

    return positions.flatten()


def compute_bridging_naive_sparse(matrix: np.ndarray) -> np.ndarray:
    """
    Compute PD bridging scores using only observed entries.

    For each item c, compute average pairwise disagreement among
    voters who approved c (using only their shared observed items).

    Args:
        matrix: (n_items, n_voters) with NaN for missing, 1=approve, 0=disapprove

    Returns:
        (n_items,) bridging scores
    """
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)

    bridging_scores = np.zeros(n_items)

    for c in range(n_items):
        # Voters who approved item c (observed and value == 1)
        approvers = np.where(observed[c, :] & (matrix[c, :] == 1))[0]
        n_approvers = len(approvers)

        if n_approvers < 2:
            bridging_scores[c] = 0.0
            continue

        # Compute pairwise disagreement among approvers
        total_disagreement = 0.0
        n_pairs = 0

        for i_idx in range(n_approvers):
            for j_idx in range(i_idx + 1, n_approvers):
                i, j = approvers[i_idx], approvers[j_idx]

                # Items both rated (excluding c)
                shared = observed[:, i] & observed[:, j]
                shared[c] = False  # Exclude the item itself
                n_shared = shared.sum()

                if n_shared == 0:
                    continue

                # Disagreement on shared items (compare actual values)
                disagree = (matrix[shared, i] != matrix[shared, j]).sum()
                total_disagreement += disagree / n_shared
                n_pairs += 1

        if n_pairs > 0:
            bridging_scores[c] = total_disagreement / n_pairs

    return bridging_scores


def plot_ridgeline_sparse(
    matrix: np.ndarray,
    voter_positions: np.ndarray,
    bridging_scores: np.ndarray,
    polis_scores: np.ndarray,
    title: str,
    output_path: Path,
    comment_texts: dict[int, str] = None,
    max_items: int = 30,
) -> None:
    """
    Create ridgeline plot for sparse data.

    Shows approval distribution and approval fraction among voters who voted.

    Args:
        matrix: (n_items, n_voters) with NaN for missing
        voter_positions: (n_voters,) MDS positions
        bridging_scores: (n_items,) bridging scores
        polis_scores: (n_items,) Polis consensus scores
        title: plot title
        output_path: where to save
        comment_texts: dict mapping item index -> comment text
        max_items: max items to display
    """
    # Compute ranks
    bridging_ranks = stats.rankdata(-bridging_scores, method="average")
    polis_ranks = stats.rankdata(-polis_scores, method="average")
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)

    # Filter out items with NaN bridging scores (no observations)
    valid_items = ~np.isnan(bridging_scores)
    valid_indices = np.where(valid_items)[0]

    # Sort valid items by bridging score descending
    sorted_indices = valid_indices[np.argsort(bridging_scores[valid_indices])[::-1]]

    # Limit items
    if len(sorted_indices) > max_items:
        sorted_indices = sorted_indices[:max_items]
    n_display = len(sorted_indices)

    # Set up x grid
    pos_min, pos_max = voter_positions.min(), voter_positions.max()
    x_margin = (pos_max - pos_min) * 0.1
    x_grid = np.linspace(pos_min - x_margin, pos_max + x_margin, 200)

    # Compute densities
    densities = []
    for idx in sorted_indices:
        # Voters who approved this item (observed and value == 1)
        approvers = observed[idx, :] & (matrix[idx, :] == 1)
        x_approvers = voter_positions[approvers]

        if len(x_approvers) < 2:
            densities.append(np.zeros_like(x_grid))
        else:
            try:
                kde = gaussian_kde(x_approvers, bw_method=0.3)
                density = kde(x_grid) * len(x_approvers)
            except:
                density = np.zeros_like(x_grid)
            densities.append(density)

    # Scale factor - taller rows to accommodate wrapped text
    max_density = max(d.max() for d in densities) if densities else 1.0
    row_height = 1.5
    scale_factor = (row_height * 0.5) / max_density if max_density > 0 else 1.0

    # Create figure
    fig_height = max(10, n_display * 0.8)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Colormap
    cmap = plt.cm.coolwarm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=pos_min, vmax=pos_max)

    # Plot ridges
    for row_idx, item_idx in enumerate(sorted_indices):
        y_base = n_display - 1 - row_idx
        density = densities[row_idx] * scale_factor

        # Draw filled gradient
        for i in range(len(x_grid) - 1):
            x_left, x_right = x_grid[i], x_grid[i + 1]
            y_left, y_right = density[i], density[i + 1]

            if y_left < 0.001 and y_right < 0.001:
                continue

            x_mid = (x_left + x_right) / 2
            color = cmap(norm(x_mid))

            verts = [
                (x_left, y_base),
                (x_left, y_base + y_left),
                (x_right, y_base + y_right),
                (x_right, y_base),
            ]
            poly = plt.Polygon(verts, facecolor=color, edgecolor='none', alpha=0.7)
            ax.add_patch(poly)

        # Outline
        ax.plot(x_grid, y_base + density, color='black', linewidth=0.5, alpha=0.5)
        ax.axhline(y=y_base, color='grey', linewidth=0.3, alpha=0.3)

        # Comment text label (wrapped)
        if comment_texts and item_idx in comment_texts:
            comment = comment_texts[item_idx]
        else:
            comment = f'Item {item_idx + 1}'

        # Wrap text to fixed width
        wrapped = textwrap.fill(comment, width=60)
        ax.text(pos_min - x_margin - 0.05, y_base + row_height * 0.4,
                wrapped, fontsize=6, ha='right', va='center', linespacing=1.2)

        # Approval fraction (among those who voted)
        voted = observed[item_idx, :]
        n_voted = voted.sum()
        if n_voted > 0:
            approved = (matrix[item_idx, :] == 1) & voted
            approval_frac = approved.sum() / n_voted
            ax.text(pos_max + x_margin + 0.05, y_base + row_height * 0.3,
                    f'{approval_frac:.0%} ({n_voted})', fontsize=8, ha='left',
                    va='center', family='monospace')

        # Polis rank and score
        polis_rank = int(polis_ranks[item_idx])
        polis_score = polis_scores[item_idx]
        ax.text(pos_max + x_margin + 0.7, y_base + row_height * 0.3,
                f'{polis_rank:3d}  {polis_score:.3f}', fontsize=8, ha='left',
                va='center', family='monospace')

        # Bridging rank and score
        bridging_rank = int(bridging_ranks[item_idx])
        bridging_score = bridging_scores[item_idx]
        ax.text(pos_max + x_margin + 1.6, y_base + row_height * 0.3,
                f'{bridging_rank:3d}  {bridging_score:.3f}', fontsize=8, ha='left',
                va='center', family='monospace')

    # Format
    ax.set_xlim(pos_min - x_margin - 0.8, pos_max + x_margin + 2.8)
    ax.set_ylim(-0.5, n_display + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Voter Position (MDS)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Headers
    ax.text(pos_min - x_margin - 0.4, n_display + 0.3, 'Comment',
            fontsize=10, fontweight='bold', ha='center')
    ax.text((pos_min + pos_max) / 2, n_display + 0.3, 'Approver Distribution',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pos_max + x_margin + 0.3, n_display + 0.3, 'Appr (n)',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pos_max + x_margin + 1.1, n_display + 0.3, 'Polis',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pos_max + x_margin + 2.0, n_display + 0.3, 'Bridging',
            fontsize=10, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset(filepath: Path, output_dir: Path) -> None:
    """Process a single sparse Polis dataset."""
    file_id = filepath.stem

    # Load data
    data = np.load(filepath)
    matrix = data['matrix']
    n_items, n_voters = matrix.shape

    # Observation rate
    observed = ~np.isnan(matrix)
    obs_rate = observed.sum() / observed.size

    print(f"  {file_id}: {n_items} items, {n_voters} voters, {obs_rate:.1%} observed")

    # Load comment texts
    print("    Loading comments...", end=" ", flush=True)
    try:
        comment_texts = load_all_comments_for_dataset(file_id)
        print(f"loaded {len(comment_texts)} comments")
    except Exception as e:
        print(f"failed ({e})")
        comment_texts = {}

    # Compute voter similarity and MDS positions
    print("    Computing voter similarity...", end=" ", flush=True)
    similarity = compute_voter_similarity_sparse(matrix)
    print("done")

    print("    Computing MDS positions...", end=" ", flush=True)
    positions = compute_voter_positions_mds(similarity)
    print("done")

    # Compute bridging scores using the standard naive estimator
    print("    Computing bridging scores...", end=" ", flush=True)
    bridging_scores = estimate_bridging_scores_naive(matrix, observed)
    print("done")

    # Compute Polis scores
    print("    Computing Polis scores...", end=" ", flush=True)
    try:
        polis_scores, polis_meta = polis_consensus_pipeline(matrix, observed, max_k=5, seed=42)
        print(f"done (k={polis_meta['k_clusters']})")
    except Exception as e:
        print(f"failed ({e})")
        polis_scores = np.zeros(n_items)

    # Generate plot
    title = f'Polis {file_id} (sparse, {obs_rate:.0%} observed)'
    output_path = output_dir / f'{file_id}.png'
    plot_ridgeline_sparse(matrix, positions, bridging_scores, polis_scores, title, output_path,
                          comment_texts=comment_texts)
    print(f"    Saved: {output_path}")


def main():
    """Run experiment 6a on all Polis datasets."""
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'data' / 'processed' / 'preflib'
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all 00069 Polis datasets
    polis_files = sorted(data_dir.glob('00069-*.npz'))

    print(f"Processing {len(polis_files)} Polis datasets...")
    print()

    for filepath in polis_files:
        process_dataset(filepath, output_dir)
        print()


if __name__ == '__main__':
    main()
