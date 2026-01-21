#!/usr/bin/env python3
"""
Experiment 6: Real Pol.is Ranking Comparison

Compares comment rankings between:
1. Polis Group-Informed Consensus
2. Pairwise Disagreement Bridging Score (Naive)

Uses incomplete Pol.is datasets (00069) for realistic evaluation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_2.bridging import compute_pairwise_disagreement
from experiment_2.estimation import estimate_bridging_scores_naive
from experiment_5.polis import polis_consensus_pipeline
from experiment_6.comments import load_all_comments_for_dataset


def load_polis_data(data_dir: Path) -> list[dict]:
    """Load all Pol.is datasets (00069)."""
    datasets = []
    files = sorted(data_dir.glob("00069-*.npz"))

    for f in files:
        data = np.load(f)
        matrix = data["matrix"]

        # Create observation mask (non-NaN entries)
        observed_mask = ~np.isnan(matrix)

        # Skip if too sparse
        obs_rate = observed_mask.sum() / observed_mask.size
        if obs_rate < 0.05:
            print(f"Skipping {f.stem}: observation rate {obs_rate:.1%} too low")
            continue

        datasets.append({
            "name": f.stem,
            "matrix": matrix,
            "observed_mask": observed_mask,
            "n_items": matrix.shape[0],
            "n_voters": matrix.shape[1],
            "observation_rate": obs_rate,
        })

    return datasets


def compute_rankings(matrix: np.ndarray, observed_mask: np.ndarray, seed: int = 42) -> dict:
    """Compute rankings using both methods."""

    # Polis Group-Informed Consensus
    polis_scores, polis_meta = polis_consensus_pipeline(
        matrix, observed_mask, max_k=5, seed=seed
    )

    # Naive Bridging Score
    bridging_scores = estimate_bridging_scores_naive(matrix, observed_mask)

    # Compute ranks (higher score = lower rank number = better)
    polis_ranks = stats.rankdata(-polis_scores, method="average")
    bridging_ranks = stats.rankdata(-bridging_scores, method="average")

    return {
        "polis_scores": polis_scores,
        "bridging_scores": bridging_scores,
        "polis_ranks": polis_ranks,
        "bridging_ranks": bridging_ranks,
        "polis_k": polis_meta["k_clusters"],
    }


def compare_rankings(rankings: dict) -> dict:
    """Compare the two ranking methods."""
    polis_ranks = rankings["polis_ranks"]
    bridging_ranks = rankings["bridging_ranks"]
    polis_scores = rankings["polis_scores"]
    bridging_scores = rankings["bridging_scores"]

    # Spearman correlation between ranks
    spearman_corr, spearman_p = stats.spearmanr(polis_ranks, bridging_ranks)

    # Kendall tau
    kendall_corr, kendall_p = stats.kendalltau(polis_ranks, bridging_ranks)

    # Top-k overlap
    n_items = len(polis_ranks)
    overlaps = {}
    for k in [1, 3, 5, 10]:
        if k > n_items:
            continue
        polis_top_k = set(np.argsort(polis_ranks)[:k])
        bridging_top_k = set(np.argsort(bridging_ranks)[:k])
        overlap = len(polis_top_k & bridging_top_k) / k
        overlaps[f"top_{k}_overlap"] = overlap

    # Which comment does each method rank #1?
    polis_top1 = int(np.argmin(polis_ranks))
    bridging_top1 = int(np.argmin(bridging_ranks))

    # Top 5 from each method
    polis_top5 = [int(i) for i in np.argsort(polis_ranks)[:5]]
    bridging_top5 = [int(i) for i in np.argsort(bridging_ranks)[:5]]

    return {
        "spearman_corr": spearman_corr,
        "spearman_p": spearman_p,
        "kendall_corr": kendall_corr,
        "kendall_p": kendall_p,
        **overlaps,
        "polis_top1_idx": polis_top1,
        "bridging_top1_idx": bridging_top1,
        "same_top1": polis_top1 == bridging_top1,
        "polis_top5_idx": polis_top5,
        "bridging_top5_idx": bridging_top5,
    }


def plot_ranking_comparison(
    rankings: dict,
    comparison: dict,
    output_path: Path,
    title: str = "",
):
    """Plot ranking comparison scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Score comparison
    ax1 = axes[0]
    ax1.scatter(rankings["polis_scores"], rankings["bridging_scores"], alpha=0.6)
    ax1.set_xlabel("Polis Consensus Score")
    ax1.set_ylabel("Bridging Score (Naive)")
    ax1.set_title("Score Comparison")

    # Highlight top-1 from each method
    polis_top1 = comparison["polis_top1_idx"]
    bridging_top1 = comparison["bridging_top1_idx"]
    ax1.scatter(
        [rankings["polis_scores"][polis_top1]],
        [rankings["bridging_scores"][polis_top1]],
        color="red", s=100, marker="^", label=f"Polis #1 (idx {polis_top1})", zorder=5
    )
    ax1.scatter(
        [rankings["polis_scores"][bridging_top1]],
        [rankings["bridging_scores"][bridging_top1]],
        color="blue", s=100, marker="s", label=f"Bridging #1 (idx {bridging_top1})", zorder=5
    )
    ax1.legend()

    # Rank comparison
    ax2 = axes[1]
    ax2.scatter(rankings["polis_ranks"], rankings["bridging_ranks"], alpha=0.6)
    ax2.plot([1, len(rankings["polis_ranks"])], [1, len(rankings["polis_ranks"])],
             "k--", alpha=0.3, label="Perfect agreement")
    ax2.set_xlabel("Polis Rank")
    ax2.set_ylabel("Bridging Rank")
    ax2.set_title(f"Rank Comparison (Spearman ρ = {comparison['spearman_corr']:.3f})")
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_top_comments_comparison(
    all_results: list[dict],
    output_path: Path,
):
    """Plot summary of top-1 agreement across datasets."""
    names = [r["name"] for r in all_results]
    same_top1 = [r["comparison"]["same_top1"] for r in all_results]
    spearman = [r["comparison"]["spearman_corr"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Top-1 agreement
    ax1 = axes[0]
    colors = ["green" if s else "red" for s in same_top1]
    ax1.bar(range(len(names)), [1 if s else 0 for s in same_top1], color=colors)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.split("-")[1] for n in names], rotation=45, ha="right")
    ax1.set_ylabel("Same Top-1 Comment")
    ax1.set_title(f"Top-1 Agreement: {sum(same_top1)}/{len(same_top1)} datasets")
    ax1.set_ylim(0, 1.2)

    # Spearman correlation distribution
    ax2 = axes[1]
    ax2.hist(spearman, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(np.mean(spearman), color="red", linestyle="--",
                label=f"Mean = {np.mean(spearman):.3f}")
    ax2.set_xlabel("Spearman Correlation")
    ax2.set_ylabel("Count")
    ax2.set_title("Ranking Correlation Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_experiment():
    """Run the full experiment."""
    print("Experiment 6: Real Pol.is Ranking Comparison")
    print("=" * 50)

    # Paths
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir.parent / "data" / "processed" / "preflib"
    plots_dir = experiment_dir / "plots"
    results_dir = experiment_dir / "results"

    plots_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Load data
    print("\nLoading Pol.is datasets...")
    datasets = load_polis_data(data_dir)
    print(f"Loaded {len(datasets)} datasets")

    if not datasets:
        print("No datasets found!")
        return

    # Process each dataset
    all_results = []

    for ds in datasets:
        print(f"\nProcessing {ds['name']}...")
        print(f"  Shape: {ds['n_items']} items × {ds['n_voters']} voters")
        print(f"  Observation rate: {ds['observation_rate']:.1%}")

        # Compute rankings
        rankings = compute_rankings(ds["matrix"], ds["observed_mask"])
        print(f"  Polis k: {rankings['polis_k']}")

        # Compare
        comparison = compare_rankings(rankings)
        print(f"  Spearman ρ: {comparison['spearman_corr']:.3f}")
        print(f"  Same top-1: {comparison['same_top1']}")

        # Load comment texts
        comments = load_all_comments_for_dataset(ds["name"])

        # Get top comment texts
        polis_top1_idx = comparison["polis_top1_idx"]
        bridging_top1_idx = comparison["bridging_top1_idx"]
        polis_top1_text = comments.get(polis_top1_idx, f"[Comment #{polis_top1_idx}]")
        bridging_top1_text = comments.get(bridging_top1_idx, f"[Comment #{bridging_top1_idx}]")

        # Truncate for display
        def truncate(s, n=80):
            return s[:n] + "..." if len(s) > n else s

        print(f"  Polis #1: {truncate(polis_top1_text)}")
        print(f"  Bridging #1: {truncate(bridging_top1_text)}")

        # Build top-5 comment info
        polis_top5_comments = [
            {"idx": idx, "text": comments.get(idx, f"[#{idx}]"), "score": float(rankings["polis_scores"][idx])}
            for idx in comparison["polis_top5_idx"]
        ]
        bridging_top5_comments = [
            {"idx": idx, "text": comments.get(idx, f"[#{idx}]"), "score": float(rankings["bridging_scores"][idx])}
            for idx in comparison["bridging_top5_idx"]
        ]

        # Plot
        plot_ranking_comparison(
            rankings,
            comparison,
            plots_dir / f"{ds['name']}.png",
            title=f"{ds['name']} (obs rate: {ds['observation_rate']:.1%})"
        )

        all_results.append({
            "name": ds["name"],
            "n_items": ds["n_items"],
            "n_voters": ds["n_voters"],
            "observation_rate": ds["observation_rate"],
            "rankings": {
                "polis_scores": rankings["polis_scores"].tolist(),
                "bridging_scores": rankings["bridging_scores"].tolist(),
                "polis_ranks": rankings["polis_ranks"].tolist(),
                "bridging_ranks": rankings["bridging_ranks"].tolist(),
                "polis_k": rankings["polis_k"],
            },
            "comparison": comparison,
            "top_comments": {
                "polis_top5": polis_top5_comments,
                "bridging_top5": bridging_top5_comments,
            },
        })

    # Summary plot
    plot_top_comments_comparison(all_results, plots_dir / "summary.png")

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    spearman_vals = [r["comparison"]["spearman_corr"] for r in all_results]
    same_top1_count = sum(r["comparison"]["same_top1"] for r in all_results)

    print(f"Datasets analyzed: {len(all_results)}")
    print(f"Mean Spearman ρ: {np.mean(spearman_vals):.3f} ± {np.std(spearman_vals):.3f}")
    print(f"Same top-1 comment: {same_top1_count}/{len(all_results)} ({100*same_top1_count/len(all_results):.0f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"ranking_comparison_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump({
            "datasets": all_results,
            "summary": {
                "n_datasets": len(all_results),
                "mean_spearman": np.mean(spearman_vals),
                "std_spearman": np.std(spearman_vals),
                "same_top1_count": same_top1_count,
                "same_top1_rate": same_top1_count / len(all_results),
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Plots saved to: {plots_dir}/")


if __name__ == "__main__":
    run_experiment()
