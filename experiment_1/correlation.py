"""
Compute correlations between approval rate, approver diversity, and bridging scores.

Compares both PD Bridging and Polis Consensus scores against approval and diversity.
"""

import sys
import numpy as np
from pathlib import Path
from glob import glob
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiment_2.bridging import (
    compute_bridging_scores_vectorized,
    compute_bridging_pnorm,
    compute_bridging_harmonic_pd,
)
from experiment_5.polis import polis_consensus_pipeline


def load_matrix(filepath):
    """Load matrix from npz file."""
    data = np.load(filepath)
    if 'matrix' in data:
        return data['matrix']
    return data['arr_0']


def compute_approval_rates(matrix):
    """Compute approval rate for each item."""
    # Use explicit == 1.0 check so 0.5 (pass) values are not counted as approvals
    return (matrix == 1.0).mean(axis=1)


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
        # Count approvals (1.0) only, not passes (0.5)
        k = np.sum(approver_vectors == 1.0, axis=1)
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
    # Polis vs approval
    pa_pearson, pa_spearman = [], []
    # Polis vs diversity
    pd_pearson, pd_spearman = [], []
    # PD bridging vs Polis
    bp_pearson, bp_spearman = [], []
    # p-norm bridging (p=-10, approx min) vs approval
    pn_min_a_pearson, pn_min_a_spearman = [], []
    # p-norm bridging (p=-10) vs diversity
    pn_min_d_pearson, pn_min_d_spearman = [], []
    # p-norm bridging (p=0, geometric mean) vs approval
    pn_geo_a_pearson, pn_geo_a_spearman = [], []
    # p-norm bridging (p=0) vs diversity
    pn_geo_d_pearson, pn_geo_d_spearman = [], []
    # p-norm bridging (p=1, approval) vs approval
    pn_avg_a_pearson, pn_avg_a_spearman = [], []
    # Harmonic PD vs approval
    hpd_a_pearson, hpd_a_spearman = [], []
    # Harmonic PD vs diversity
    hpd_d_pearson, hpd_d_spearman = [], []

    for f in tqdm(files, desc="Processing files", leave=False):
        matrix = load_matrix(f)

        # Skip files with NaN for Polis (needs complete or imputable data)
        if np.isnan(matrix).any():
            # For sparse data, create observed mask
            observed_mask = ~np.isnan(matrix)
            # Fill NaN with 0 for computations
            matrix_filled = np.nan_to_num(matrix, nan=0.0)
        else:
            observed_mask = np.ones_like(matrix, dtype=bool)
            matrix_filled = matrix

        rates = compute_approval_rates(matrix_filled)
        diversity = compute_approver_diversity(matrix_filled)
        bridging = compute_bridging_scores_vectorized(matrix_filled)

        # Compute p-norm bridging scores
        pnorm_min = compute_bridging_pnorm(matrix_filled, p=-10)  # approx min
        pnorm_geo = compute_bridging_pnorm(matrix_filled, p=0)    # geometric mean
        pnorm_avg = compute_bridging_pnorm(matrix_filled, p=1)    # arithmetic mean (= approval)

        # Compute harmonic PD
        harmonic_pd = compute_bridging_harmonic_pd(matrix_filled)

        # Compute Polis consensus scores
        try:
            polis_scores, _ = polis_consensus_pipeline(matrix_filled, observed_mask, seed=42)
        except Exception:
            continue

        valid = ~np.isnan(diversity)
        if valid.sum() < 3:
            continue

        # Approval vs Diversity
        r_p, _ = pearsonr(rates[valid], diversity[valid])
        r_s, _ = spearmanr(rates[valid], diversity[valid])
        ad_pearson.append(r_p)
        ad_spearman.append(r_s)

        # PD Bridging vs Approval
        r_p, _ = pearsonr(bridging[valid], rates[valid])
        r_s, _ = spearmanr(bridging[valid], rates[valid])
        ba_pearson.append(r_p)
        ba_spearman.append(r_s)

        # PD Bridging vs Diversity
        r_p, _ = pearsonr(bridging[valid], diversity[valid])
        r_s, _ = spearmanr(bridging[valid], diversity[valid])
        bd_pearson.append(r_p)
        bd_spearman.append(r_s)

        # Polis vs Approval
        r_p, _ = pearsonr(polis_scores[valid], rates[valid])
        r_s, _ = spearmanr(polis_scores[valid], rates[valid])
        pa_pearson.append(r_p)
        pa_spearman.append(r_s)

        # Polis vs Diversity
        r_p, _ = pearsonr(polis_scores[valid], diversity[valid])
        r_s, _ = spearmanr(polis_scores[valid], diversity[valid])
        pd_pearson.append(r_p)
        pd_spearman.append(r_s)

        # PD Bridging vs Polis
        r_p, _ = pearsonr(bridging[valid], polis_scores[valid])
        r_s, _ = spearmanr(bridging[valid], polis_scores[valid])
        bp_pearson.append(r_p)
        bp_spearman.append(r_s)

        # p-norm (p=-10, min) vs Approval
        r_p, _ = pearsonr(pnorm_min[valid], rates[valid])
        r_s, _ = spearmanr(pnorm_min[valid], rates[valid])
        pn_min_a_pearson.append(r_p)
        pn_min_a_spearman.append(r_s)

        # p-norm (p=-10, min) vs Diversity
        r_p, _ = pearsonr(pnorm_min[valid], diversity[valid])
        r_s, _ = spearmanr(pnorm_min[valid], diversity[valid])
        pn_min_d_pearson.append(r_p)
        pn_min_d_spearman.append(r_s)

        # p-norm (p=0, geo) vs Approval
        r_p, _ = pearsonr(pnorm_geo[valid], rates[valid])
        r_s, _ = spearmanr(pnorm_geo[valid], rates[valid])
        pn_geo_a_pearson.append(r_p)
        pn_geo_a_spearman.append(r_s)

        # p-norm (p=0, geo) vs Diversity
        r_p, _ = pearsonr(pnorm_geo[valid], diversity[valid])
        r_s, _ = spearmanr(pnorm_geo[valid], diversity[valid])
        pn_geo_d_pearson.append(r_p)
        pn_geo_d_spearman.append(r_s)

        # p-norm (p=1, avg) vs Approval (should be ~1.0)
        r_p, _ = pearsonr(pnorm_avg[valid], rates[valid])
        r_s, _ = spearmanr(pnorm_avg[valid], rates[valid])
        pn_avg_a_pearson.append(r_p)
        pn_avg_a_spearman.append(r_s)

        # Harmonic PD vs Approval
        r_p, _ = pearsonr(harmonic_pd[valid], rates[valid])
        r_s, _ = spearmanr(harmonic_pd[valid], rates[valid])
        hpd_a_pearson.append(r_p)
        hpd_a_spearman.append(r_s)

        # Harmonic PD vs Diversity
        r_p, _ = pearsonr(harmonic_pd[valid], diversity[valid])
        r_s, _ = spearmanr(harmonic_pd[valid], diversity[valid])
        hpd_d_pearson.append(r_p)
        hpd_d_spearman.append(r_s)

    return {
        'approval_diversity': (ad_pearson, ad_spearman),
        'bridging_approval': (ba_pearson, ba_spearman),
        'bridging_diversity': (bd_pearson, bd_spearman),
        'polis_approval': (pa_pearson, pa_spearman),
        'polis_diversity': (pd_pearson, pd_spearman),
        'bridging_polis': (bp_pearson, bp_spearman),
        'pnorm_min_approval': (pn_min_a_pearson, pn_min_a_spearman),
        'pnorm_min_diversity': (pn_min_d_pearson, pn_min_d_spearman),
        'pnorm_geo_approval': (pn_geo_a_pearson, pn_geo_a_spearman),
        'pnorm_geo_diversity': (pn_geo_d_pearson, pn_geo_d_spearman),
        'pnorm_avg_approval': (pn_avg_a_pearson, pn_avg_a_spearman),
        'harmonic_pd_approval': (hpd_a_pearson, hpd_a_spearman),
        'harmonic_pd_diversity': (hpd_d_pearson, hpd_d_spearman),
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
        ('polis_approval', 'Polis vs Approval'),
        ('polis_diversity', 'Polis vs Diversity'),
        ('bridging_polis', 'PD Bridging vs Polis'),
        ('pnorm_min_approval', 'p-norm (p=-10, min) vs Approval'),
        ('pnorm_min_diversity', 'p-norm (p=-10, min) vs Diversity'),
        ('pnorm_geo_approval', 'p-norm (p=0, geo) vs Approval'),
        ('pnorm_geo_diversity', 'p-norm (p=0, geo) vs Diversity'),
        ('pnorm_avg_approval', 'p-norm (p=1, avg) vs Approval'),
        ('harmonic_pd_approval', 'Harmonic PD vs Approval'),
        ('harmonic_pd_diversity', 'Harmonic PD vs Diversity'),
    ]

    for key, label in labels:
        pearson, spearman = results[key]
        if len(pearson) == 0:
            print(f"  {label}: N/A")
            continue
        print(f"  {label}:")
        print(f"    Pearson:  {np.mean(pearson):>6.3f} ± {np.std(pearson):.3f}")
        print(f"    Spearman: {np.mean(spearman):>6.3f} ± {np.std(spearman):.3f}")
        print()


def main():
    base_dir = Path(__file__).parent.parent

    # Define all data sources (excluding Pol.is - uses imputed data)
    datasets = [
        ('French Election 2002 (00026)', 'data/processed/preflib/00026-*.npz'),
        ('San Sebastian Poster (00033)', 'data/processed/preflib/00033-*.npz'),
        ('CTU Tutorial (00063)', 'data/processed/preflib/00063-*.npz'),
        ('French Election 2007 (00071)', 'data/processed/preflib/00071-*.npz'),
    ]

    print("=" * 55)
    print("Correlations: Approval, Diversity, and PD Bridging Score")
    print("=" * 55)
    print()

    for name, pattern in tqdm(datasets, desc="Datasets"):
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
