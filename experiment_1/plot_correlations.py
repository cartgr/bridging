"""
Plot correlations between different bridging metrics and approval/diversity.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
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
    return np.mean(matrix, axis=1)


def compute_approver_diversity(matrix):
    """Compute diversity of approvers for each item."""
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


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent / 'plots' / 'correlations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data (excluding Pol.is - uses imputed data)
    datasets = [
        ('French Election 2002', 'data/processed/preflib/00026-*.npz'),
        ('San Sebastian Poster', 'data/processed/preflib/00033-*.npz'),
        ('CTU Tutorial', 'data/processed/preflib/00063-*.npz'),
        ('French Election 2007', 'data/processed/preflib/00071-*.npz'),
    ]

    # Collect all data points
    all_data = {
        'approval': [],
        'diversity': [],
        'pd_bridging': [],
        'pnorm_min': [],
        'pnorm_geo': [],
        'harmonic_pd': [],
        'polis': [],
        'dataset': [],
    }

    for name, pattern in tqdm(datasets, desc="Loading datasets"):
        files = sorted(glob(str(base_dir / pattern)))
        for f in files:
            matrix = load_matrix(f)
            if np.isnan(matrix).any():
                observed_mask = ~np.isnan(matrix)
                matrix_filled = np.nan_to_num(matrix, nan=0.0)
            else:
                observed_mask = np.ones_like(matrix, dtype=bool)
                matrix_filled = matrix

            approval = compute_approval_rates(matrix_filled)
            diversity = compute_approver_diversity(matrix_filled)
            pd_bridging = compute_bridging_scores_vectorized(matrix_filled)
            pnorm_min = compute_bridging_pnorm(matrix_filled, p=-10)
            pnorm_geo = compute_bridging_pnorm(matrix_filled, p=0)
            harmonic_pd = compute_bridging_harmonic_pd(matrix_filled)

            try:
                polis, _ = polis_consensus_pipeline(matrix_filled, observed_mask, seed=42)
            except Exception as e:
                print(f"Polis failed for {f}: {e}")
                polis = np.full_like(approval, np.nan)

            all_data['approval'].extend(approval)
            all_data['diversity'].extend(diversity)
            all_data['pd_bridging'].extend(pd_bridging)
            all_data['pnorm_min'].extend(pnorm_min)
            all_data['pnorm_geo'].extend(pnorm_geo)
            all_data['harmonic_pd'].extend(harmonic_pd)
            all_data['polis'].extend(polis)
            all_data['dataset'].extend([name] * len(approval))

    # Convert to arrays
    for key in all_data:
        all_data[key] = np.array(all_data[key])

    # Create scatter plots
    metrics = [
        ('pd_bridging', 'PD Bridging'),
        ('pnorm_min', 'p-norm (p=-10, min)'),
        ('pnorm_geo', 'p-norm (p=0, geo)'),
        ('harmonic_pd', 'Harmonic PD'),
        ('polis', 'Polis Consensus'),
    ]

    # Plot 1: All metrics vs Approval (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = {'French Election 2002': 'C0', 'San Sebastian Poster': 'C1',
              'CTU Tutorial': 'C2', 'French Election 2007': 'C3'}

    for idx, (key, label) in enumerate(metrics):
        ax = axes[idx]
        for ds_name in colors:
            mask = all_data['dataset'] == ds_name
            valid = ~np.isnan(all_data[key][mask])
            ax.scatter(all_data['approval'][mask][valid], all_data[key][mask][valid],
                      alpha=0.5, label=ds_name, c=colors[ds_name], s=20)
        ax.set_xlabel('Approval Rate')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Approval')
        ax.set_xlim(0, 1)

        # Add correlation
        valid = ~np.isnan(all_data[key])
        from scipy.stats import spearmanr
        rho, _ = spearmanr(all_data['approval'][valid], all_data[key][valid])
        ax.annotate(f'ρ = {rho:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, va='top')

    # Hide last subplot, add legend
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_vs_approval.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'all_metrics_vs_approval.png'}")

    # Plot 1b: All metrics vs Diversity (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(metrics):
        ax = axes[idx]
        for ds_name in colors:
            mask = all_data['dataset'] == ds_name
            valid = ~np.isnan(all_data[key][mask]) & ~np.isnan(all_data['diversity'][mask])
            ax.scatter(all_data['diversity'][mask][valid], all_data[key][mask][valid],
                      alpha=0.5, label=ds_name, c=colors[ds_name], s=20)
        ax.set_xlabel('Approver Diversity')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Diversity')

        # Add correlation
        valid = ~np.isnan(all_data[key]) & ~np.isnan(all_data['diversity'])
        from scipy.stats import spearmanr
        rho, _ = spearmanr(all_data['diversity'][valid], all_data[key][valid])
        ax.annotate(f'ρ = {rho:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=10, va='top')

    # Hide last subplot, add legend
    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_vs_diversity.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'all_metrics_vs_diversity.png'}")

    # Plot 2: Comparison of p=-10 (min) vs PD Bridging
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for ds_name in colors:
        mask = all_data['dataset'] == ds_name
        ax.scatter(all_data['pd_bridging'][mask], all_data['pnorm_min'][mask],
                  alpha=0.5, label=ds_name, c=colors[ds_name], s=20)
    ax.set_xlabel('PD Bridging')
    ax.set_ylabel('p-norm (p=-10, min)')
    ax.set_title('PD Bridging vs p-norm Min')
    ax.plot([0, 0.3], [0, 0.3], 'k--', alpha=0.3)
    ax.legend()

    ax = axes[1]
    for ds_name in colors:
        mask = all_data['dataset'] == ds_name
        ax.scatter(all_data['pd_bridging'][mask], all_data['harmonic_pd'][mask],
                  alpha=0.5, label=ds_name, c=colors[ds_name], s=20)
    ax.set_xlabel('PD Bridging')
    ax.set_ylabel('Harmonic PD')
    ax.set_title('PD Bridging vs Harmonic PD')
    ax.plot([0, 0.3], [0, 0.3], 'k--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparisons.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'metric_comparisons.png'}")

    # Plot 3: Bar chart of correlations with approval by dataset
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    from scipy.stats import spearmanr

    dataset_names = list(colors.keys())
    metric_keys = ['pd_bridging', 'pnorm_min', 'pnorm_geo', 'harmonic_pd', 'polis']
    metric_labels = ['PD Bridging', 'p-norm\n(p=-10)', 'p-norm\n(p=0)', 'Harmonic\nPD', 'Polis']

    x = np.arange(len(dataset_names))
    width = 0.15

    # Left plot: correlation with approval
    ax = axes[0]
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        corrs = []
        for ds_name in dataset_names:
            mask = all_data['dataset'] == ds_name
            valid = ~np.isnan(all_data[key][mask])
            if valid.sum() > 2:
                rho, _ = spearmanr(all_data['approval'][mask][valid], all_data[key][mask][valid])
                corrs.append(rho)
            else:
                corrs.append(np.nan)
        ax.bar(x + i * width, corrs, width, label=label)

    ax.set_ylabel('Spearman Correlation')
    ax.set_xlabel('Dataset')
    ax.set_title('Correlation with Approval Rate')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Right plot: correlation with diversity
    ax = axes[1]
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        corrs = []
        for ds_name in dataset_names:
            mask = all_data['dataset'] == ds_name
            # Need valid for both metric and diversity
            valid = ~np.isnan(all_data[key][mask]) & ~np.isnan(all_data['diversity'][mask])
            if valid.sum() > 2:
                rho, _ = spearmanr(all_data['diversity'][mask][valid], all_data[key][mask][valid])
                corrs.append(rho)
            else:
                corrs.append(np.nan)
        ax.bar(x + i * width, corrs, width, label=label)

    ax.set_ylabel('Spearman Correlation')
    ax.set_xlabel('Dataset')
    ax.set_title('Correlation with Approver Diversity')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=8)
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_bars.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'correlation_bars.png'}")

    print("\nDone!")


if __name__ == '__main__':
    main()
