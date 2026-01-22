"""
Compute correlations between approval rate, approver diversity, and PD bridging score.

Outputs summary statistics for French Election and Pol.is datasets separately.
"""

import sys
import numpy as np
from pathlib import Path
from glob import glob
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_2.bridging import compute_bridging_scores_vectorized


def load_matrix(filepath):
    """Load matrix from npz file."""
    data = np.load(filepath)
    if 'matrix' in data:
        return data['matrix']
    return data['arr_0']


def compute_approval_rates(matrix):
    """Compute approval rate for each item."""
    return np.mean(matrix, axis=1)


def compute_approver_diversity(matrix):
    """
    Compute diversity of approvers for each item.

    Diversity = average pairwise Hamming distance between approvers' full voting profiles.
    """
    n_items = matrix.shape[0]
    diversity = np.zeros(n_items)

    for i in range(n_items):
        approvers = np.where(matrix[i, :] == 1.0)[0]
        n = len(approvers)

        if n < 2:
            diversity[i] = np.nan
            continue

        approver_vectors = matrix[:, approvers]
        k = np.sum(approver_vectors, axis=1)
        total_disagreements = np.sum(k * (n - k))
        n_pairs = n * (n - 1) / 2
        diversity[i] = total_disagreements / (n_pairs * n_items)

    return diversity


def compute_all_correlations(files):
    """Compute all pairwise correlations for a list of files."""
    # approval vs diversity
    ad_pearson, ad_spearman = [], []
    # PD bridging vs approval
    ba_pearson, ba_spearman = [], []
    # PD bridging vs diversity
    bd_pearson, bd_spearman = [], []

    for f in files:
        matrix = load_matrix(f)
        rates = compute_approval_rates(matrix)
        diversity = compute_approver_diversity(matrix)
        bridging = compute_bridging_scores_vectorized(matrix)

        valid = ~np.isnan(diversity)
        if valid.sum() < 3:
            continue

        # Approval vs Diversity
        r_p, _ = pearsonr(rates[valid], diversity[valid])
        r_s, _ = spearmanr(rates[valid], diversity[valid])
        ad_pearson.append(r_p)
        ad_spearman.append(r_s)

        # Bridging vs Approval
        r_p, _ = pearsonr(bridging[valid], rates[valid])
        r_s, _ = spearmanr(bridging[valid], rates[valid])
        ba_pearson.append(r_p)
        ba_spearman.append(r_s)

        # Bridging vs Diversity
        r_p, _ = pearsonr(bridging[valid], diversity[valid])
        r_s, _ = spearmanr(bridging[valid], diversity[valid])
        bd_pearson.append(r_p)
        bd_spearman.append(r_s)

    return {
        'approval_diversity': (ad_pearson, ad_spearman),
        'bridging_approval': (ba_pearson, ba_spearman),
        'bridging_diversity': (bd_pearson, bd_spearman),
        'n': len(ad_pearson),
    }


def print_results(name, results):
    """Print correlation results for a dataset."""
    print(f"{name}")
    print(f"  n = {results['n']} datasets")
    print()

    labels = [
        ('approval_diversity', 'Approval vs Diversity'),
        ('bridging_approval', 'PD Bridging vs Approval'),
        ('bridging_diversity', 'PD Bridging vs Diversity'),
    ]

    for key, label in labels:
        pearson, spearman = results[key]
        print(f"  {label}:")
        print(f"    Pearson:  {np.mean(pearson):>6.3f} ± {np.std(pearson):.3f}")
        print(f"    Spearman: {np.mean(spearman):>6.3f} ± {np.std(spearman):.3f}")
        print()


def main():
    base_dir = Path(__file__).parent.parent

    # Define all data sources
    datasets = [
        ('French Election 2002 (00026)', 'data/processed/preflib/00026-*.npz'),
        ('San Sebastian Poster (00033)', 'data/processed/preflib/00033-*.npz'),
        ('CTU Tutorial (00063)', 'data/processed/preflib/00063-*.npz'),
        ('French Election 2007 (00071)', 'data/processed/preflib/00071-*.npz'),
        ('Pol.is (00069)', 'data/completed/00069-*.npz'),
    ]

    print("=" * 55)
    print("Correlations: Approval, Diversity, and PD Bridging Score")
    print("=" * 55)
    print()

    for name, pattern in datasets:
        files = sorted(glob(str(base_dir / pattern)))
        if not files:
            print(f"{name}: No files found")
            print()
            continue
        results = compute_all_correlations(files)
        print_results(name, results)
        print("-" * 55)
        print()


if __name__ == '__main__':
    main()
