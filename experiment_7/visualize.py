"""
Visualization for Experiment 7: Polis Seed Stability
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_results():
    """Load experiment results."""
    with open("experiment_7/results/polis_seed_stability.json") as f:
        return json.load(f)


def plot_seed_stability(results: dict, output_dir: Path):
    """Plot seed stability metrics."""
    aggregate = results["aggregate"]

    obs_rates = [r["obs_rate"] for r in aggregate]
    rank_corrs = [r["rank_corr_mean"] for r in aggregate]
    rank_corr_stds = [r["rank_corr_std"] for r in aggregate]
    top1_agrees = [r["top1_agree_mean"] for r in aggregate]
    k_uniques = [r["k_unique_mean"] for r in aggregate]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Rank correlation
    ax = axes[0]
    ax.errorbar(obs_rates, rank_corrs, yerr=rank_corr_stds,
                marker='o', capsize=3, color='#d62728', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect stability')
    ax.set_xlabel('Observation Rate')
    ax.set_ylabel('Rank Correlation (across seeds)')
    ax.set_title('Polis Ranking Stability')
    ax.set_ylim(0.95, 1.01)
    ax.grid(True, alpha=0.3)

    # Plot 2: Top-1 agreement
    ax = axes[1]
    ax.plot(obs_rates, top1_agrees, marker='s', color='#d62728', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect agreement')
    # Add PD Bridging reference (always 1.0)
    ax.axhline(y=1.0, color='#1f77b4', linestyle='-', alpha=0.7,
               label='PD Bridging (deterministic)', linewidth=2)
    ax.set_xlabel('Observation Rate')
    ax.set_ylabel('Top-1 Agreement')
    ax.set_title('Which Item Ranks First?')
    ax.set_ylim(0.85, 1.02)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Plot 3: k-cluster variation
    ax = axes[2]
    ax.plot(obs_rates, k_uniques, marker='^', color='#d62728', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No variation')
    ax.set_xlabel('Observation Rate')
    ax.set_ylabel('Unique k Values (across seeds)')
    ax.set_title('K-means Cluster Instability')
    ax.set_ylim(0.9, 2.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "polis_seed_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_dir / 'polis_seed_stability.png'}")


def plot_worst_cases(results: dict, output_dir: Path):
    """Plot worst-case instability examples."""
    datasets = results["datasets"]

    # Find worst cases
    worst_cases = []
    for name, data in datasets.items():
        for r in data:
            if r["top_1_agreement"] < 1.0:
                worst_cases.append({
                    "name": name,
                    "obs_rate": r["actual_obs_rate"],
                    "top1": r["top_1_agreement"],
                    "rank_corr": r["rank_correlation_mean"],
                    "k_unique": r["k_unique"],
                })

    worst_cases.sort(key=lambda x: x["top1"])

    if not worst_cases:
        print("No instability cases found")
        return

    # Plot top 10 worst cases
    top_n = min(10, len(worst_cases))
    worst = worst_cases[:top_n]

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"{w['name'].split('-')[-1]}\n({w['obs_rate']:.0%} obs)" for w in worst]
    top1s = [w["top1"] for w in worst]
    colors = ['#d62728' if t < 0.7 else '#ff7f0e' if t < 0.9 else '#2ca02c' for t in top1s]

    bars = ax.barh(range(top_n), top1s, color=colors)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Top-1 Agreement (across 20 seeds)')
    ax.set_title('Worst Polis Seed Instability Cases\n(Same masked data, different random seeds)')
    ax.set_xlim(0, 1.05)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top1s)):
        ax.text(val + 0.02, i, f'{val:.0%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "polis_worst_cases.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_dir / 'polis_worst_cases.png'}")


if __name__ == "__main__":
    output_dir = Path("experiment_7/plots")
    output_dir.mkdir(exist_ok=True)

    results = load_results()
    plot_seed_stability(results, output_dir)
    plot_worst_cases(results, output_dir)
