#!/usr/bin/env python3
"""
Experiment 6: Real Pol.is Ranking Comparison

Compares comment rankings between:
1. Polis Group-Informed Consensus
2. Pairwise Disagreement Bridging Score (Naive)

Uses incomplete Pol.is datasets (00069) for realistic evaluation.
"""

import csv
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
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


def save_rankings_csv(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    rankings: dict,
    comments: dict[int, str],
    output_path: Path,
    sort_by: str = "polis",
):
    """Save all comments with their ranks to CSV, sorted by rank."""
    n_items = len(rankings["polis_ranks"])

    # Build rows
    rows = []
    for idx in range(n_items):
        # Compute approval rate
        obs_votes = observed_mask[idx]
        if obs_votes.sum() > 0:
            approval_rate = (matrix[idx, obs_votes] == 1.0).sum() / obs_votes.sum()
        else:
            approval_rate = 0.0

        rows.append({
            "polis_rank": int(rankings["polis_ranks"][idx]),
            "bridging_rank": int(rankings["bridging_ranks"][idx]),
            "approval_rate": f"{approval_rate:.1%}",
            "polis_score": rankings["polis_scores"][idx],
            "bridging_score": rankings["bridging_scores"][idx],
            "comment": comments.get(idx, f"[Comment #{idx}]"),
        })

    # Sort by specified rank (1 to n)
    sort_key = "polis_rank" if sort_by == "polis" else "bridging_rank"
    rows.sort(key=lambda r: r[sort_key])

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["polis_rank", "bridging_rank", "approval_rate", "polis_score", "bridging_score", "comment"])
        writer.writeheader()
        writer.writerows(rows)


def compute_voter_pca_scores(matrix: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Compute PC1 scores for voters using imputed matrix."""
    # Impute missing values with column means
    imputed = matrix.copy()
    for c in range(matrix.shape[0]):
        col_mask = observed_mask[c]
        if col_mask.sum() > 0:
            col_mean = matrix[c, col_mask].mean()
            imputed[c, ~col_mask] = col_mean
        else:
            imputed[c, :] = 0

    # PCA on voters
    voter_matrix = imputed.T  # (n_voters, n_items)
    centered = voter_matrix - np.nanmean(voter_matrix, axis=0)
    centered = np.nan_to_num(centered, 0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return U[:, 0] * S[0]


def wrap_comment(text: str, max_width: int = 60) -> list[str]:
    """Wrap comment text into multiple lines."""
    text = text.replace('\n', ' ').strip()
    if not text:
        return ["[No text]"]
    return textwrap.wrap(text, width=max_width) or [text]


def plot_ridgeline_rankings(
    matrix: np.ndarray,
    observed_mask: np.ndarray,
    rankings: dict,
    comments: dict[int, str],
    output_path: Path,
    title: str = "",
    max_items: int = 30,
    sort_by: str = "bridging",
):
    """
    Create ridgeline visualization showing:
    - Full comment text on left (wrapped)
    - Voter approval distribution in middle
    - Both Polis and Bridging ranks/scores on right

    Sorted by bridging score by default.
    """
    n_items = matrix.shape[0]

    # Get PC1 scores for voters
    pc1_scores = compute_voter_pca_scores(matrix, observed_mask)

    # Sort by specified ranking
    if sort_by == "polis":
        sorted_indices = np.argsort(rankings["polis_ranks"])
    else:
        sorted_indices = np.argsort(rankings["bridging_ranks"])

    # Limit to top N
    if n_items > max_items:
        sorted_indices = sorted_indices[:max_items]

    n_display = len(sorted_indices)

    # Set up x grid for density
    pc1_min, pc1_max = pc1_scores.min(), pc1_scores.max()
    x_margin = (pc1_max - pc1_min) * 0.1
    x_grid = np.linspace(pc1_min - x_margin, pc1_max + x_margin, 100)

    # Compute densities and wrapped comments for displayed items
    densities = []
    wrapped_comments = []
    for idx in sorted_indices:
        # Get voters who approved this item
        approvers = (matrix[idx] == 1.0) & observed_mask[idx]
        x_approvers = pc1_scores[approvers]
        if len(x_approvers) >= 2:
            kde = gaussian_kde(x_approvers, bw_method=0.3)
            density = kde(x_grid) * len(x_approvers)
        else:
            density = np.zeros_like(x_grid)
        densities.append(density)
        wrapped_comments.append(wrap_comment(comments.get(idx, f"[#{idx}]"), max_width=55))

    # Calculate row heights based on number of comment lines
    line_height = 0.18  # Height per line of text
    min_row_height = 1.0
    row_heights = [max(min_row_height, len(wc) * line_height + 0.3) for wc in wrapped_comments]

    # Scale factor for density (relative to minimum row height)
    max_density = max(d.max() for d in densities) if densities else 1.0
    scale_factor = (min_row_height * 0.75) / max_density if max_density > 0 else 1.0

    # Calculate cumulative y positions (bottom to top)
    y_positions = []
    y_current = 0
    for i in range(n_display - 1, -1, -1):
        y_positions.insert(0, y_current)
        y_current += row_heights[i]
    total_height = y_current

    # Create figure
    fig_height = max(10, total_height * 0.6)
    fig, ax = plt.subplots(figsize=(18, fig_height))

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Colormap for gradient fill
    cmap = plt.cm.coolwarm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=pc1_min, vmax=pc1_max)

    # Plot each item
    for row_idx, item_idx in enumerate(sorted_indices):
        y_base = y_positions[row_idx]
        density = densities[row_idx] * scale_factor
        row_height = row_heights[row_idx]

        # Draw filled density with gradient
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

        # Baseline
        ax.axhline(y=y_base, color='grey', linewidth=0.3, alpha=0.3)

        # Compute approval rate for this item
        obs_votes = observed_mask[item_idx]
        if obs_votes.sum() > 0:
            approval_rate = (matrix[item_idx, obs_votes] == 1.0).sum() / obs_votes.sum()
        else:
            approval_rate = 0.0

        # Comment text on left (wrapped, multiple lines)
        comment_lines = wrapped_comments[row_idx]
        y_text_center = y_base + row_height * 0.5
        y_text_start = y_text_center + (len(comment_lines) - 1) * line_height / 2
        for line_idx, line in enumerate(comment_lines):
            ax.text(
                pc1_min - x_margin - 0.3, y_text_start - line_idx * line_height,
                line,
                fontsize=7,
                ha='right',
                va='center',
                family='sans-serif',
            )

        # Metrics on right side (vertically centered)
        y_metrics = y_base + row_height * 0.4

        # Approval rate
        ax.text(
            pc1_max + x_margin + 0.2, y_metrics,
            f'{approval_rate:.0%}',
            fontsize=8,
            ha='left',
            va='center',
            family='monospace',
        )

        # Polis rank and score
        polis_rank = int(rankings["polis_ranks"][item_idx])
        polis_score = rankings["polis_scores"][item_idx]
        ax.text(
            pc1_max + x_margin + 1.0, y_metrics,
            f'{polis_rank:3d}  {polis_score:.3f}',
            fontsize=8,
            ha='left',
            va='center',
            family='monospace',
        )

        # Bridging rank and score
        bridging_rank = int(rankings["bridging_ranks"][item_idx])
        bridging_score = rankings["bridging_scores"][item_idx]
        ax.text(
            pc1_max + x_margin + 2.8, y_metrics,
            f'{bridging_rank:3d}  {bridging_score:.3f}',
            fontsize=8,
            ha='left',
            va='center',
            family='monospace',
        )

    # Format axes
    ax.set_xlim(pc1_min - x_margin - 7.0, pc1_max + x_margin + 5.5)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_yticks([])
    ax.set_xlabel('PC1 Score (Voter Spectrum)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Column headers
    ax.text(pc1_min - x_margin - 3.5, total_height + 0.3, 'Comment',
            fontsize=10, fontweight='bold', ha='center')
    ax.text((pc1_min + pc1_max) / 2, total_height + 0.3, 'Approval Distribution',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 0.4, total_height + 0.3, 'Appr',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 1.6, total_height + 0.3, 'Polis',
            fontsize=10, fontweight='bold', ha='center')
    ax.text(pc1_max + x_margin + 3.4, total_height + 0.3, 'Bridging',
            fontsize=10, fontweight='bold', ha='center')

    # Sub-headers for rank/score
    ax.text(pc1_max + x_margin + 1.0, total_height + 0.0, 'Rank  Score',
            fontsize=7, ha='left', color='grey', family='monospace')
    ax.text(pc1_max + x_margin + 2.8, total_height + 0.0, 'Rank  Score',
            fontsize=7, ha='left', color='grey', family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

        # Save CSV with rankings
        save_rankings_csv(ds["matrix"], ds["observed_mask"], rankings, comments, results_dir / f"{ds['name']}_rankings.csv")

        # Ridgeline visualization (sorted by bridging score)
        plot_ridgeline_rankings(
            ds["matrix"],
            ds["observed_mask"],
            rankings,
            comments,
            plots_dir / f"{ds['name']}_ridgeline.png",
            title=f"{ds['name']} - Top Comments by Bridging Score",
            max_items=30,
            sort_by="bridging",
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
