#!/usr/bin/env python3
"""
Experiment 5: Robustness Comparison Under Random Masking

Compares robustness of Polis Group-Informed Consensus vs our Bridging Score
using French election data (00026) with random MCAR masking.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_5.robustness import run_masking_experiment, run_experiment_on_datasets
from experiment_5.evaluate import aggregate_by_mask_rate, compute_summary_statistics
from experiment_5.visualize import (
    plot_robustness_comparison,
    plot_variance_comparison,
    plot_multi_metric,
    plot_top_k_precision,
    plot_ranking_stability,
    create_summary_table,
)
from experiment_5.run_simulation_experiment import run_full_simulation_experiment


def find_data_files() -> list[Path]:
    """Find French election data files (00026)."""
    data_dir = Path(__file__).parent.parent / "data" / "processed" / "preflib"
    files = sorted(data_dir.glob("00026-*.npz"))
    return files


def save_results(results: dict, output_dir: Path, name: str) -> Path:
    """Save results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{name}_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return output_path


def run_single_dataset_experiment(
    data_path: Path,
    mask_rates: list[float],
    n_trials: int,
    base_seed: int,
    output_dir: Path,
    plots_dir: Path,
) -> dict:
    """Run experiment on a single dataset and generate outputs."""
    name = data_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # Load data
    data = np.load(data_path)
    matrix = data["matrix"]
    print(f"Matrix shape: {matrix.shape}")

    # Run experiment
    results = run_masking_experiment(
        matrix,
        mask_rates=mask_rates,
        n_trials=n_trials,
        base_seed=base_seed,
        polis_max_k=5,
        show_progress=True,
    )

    # Aggregate results
    aggregated = aggregate_by_mask_rate(results)
    summary = compute_summary_statistics(aggregated)

    # Print summary
    print(f"\nSummary for {name}:")
    print(f"  Bridging wins (Spearman): {summary['bridging_wins_spearman']}")
    print(f"  Polis wins (Spearman): {summary['polis_wins_spearman']}")
    print(f"  Ties (Spearman): {summary['ties_spearman']}")

    # Generate plots
    dataset_plots_dir = plots_dir / name
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    # Spearman correlation plot
    plot_robustness_comparison(
        aggregated,
        dataset_plots_dir / "spearman_comparison.png",
        metric="spearman",
        title=f"{name}: Spearman Correlation vs Observation Rate",
    )

    # RMSE plot
    plot_robustness_comparison(
        aggregated,
        dataset_plots_dir / "rmse_comparison.png",
        metric="rmse",
        title=f"{name}: RMSE vs Observation Rate",
    )

    # Variance plot
    plot_variance_comparison(
        aggregated,
        dataset_plots_dir / "variance_comparison.png",
        title=f"{name}: Estimate Variance vs Observation Rate",
    )

    # Multi-metric plot
    plot_multi_metric(
        aggregated,
        dataset_plots_dir / "multi_metric.png",
        metrics=["spearman", "rmse"],
        title=f"{name}: Robustness Comparison",
    )

    # Top-3 precision plot
    plot_top_k_precision(
        aggregated,
        dataset_plots_dir / "top_3_precision.png",
        k=3,
        title=f"{name}: Top-3 Precision vs Observation Rate",
    )

    # Ranking stability plot
    plot_ranking_stability(
        aggregated,
        dataset_plots_dir / "ranking_stability.png",
        title=f"{name}: Ranking Stability",
    )

    # Summary table
    create_summary_table(
        aggregated,
        dataset_plots_dir / "summary_table.md",
    )

    # Prepare output
    output = {
        "name": name,
        "matrix_shape": matrix.shape,
        "mask_rates": mask_rates,
        "n_trials": n_trials,
        "aggregated": aggregated,
        "summary": summary,
        "gt_bridging": results["gt_bridging"],
        "gt_polis": results["gt_polis"],
    }

    return output


def run_full_experiment():
    """Run the full experiment on all French election datasets."""
    print("Experiment 5: Robustness Comparison Under Random Masking")
    print("=" * 60)

    # Configuration
    mask_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    n_trials = 50
    base_seed = 42

    # Output directories
    experiment_dir = Path(__file__).parent
    output_dir = experiment_dir / "results"
    plots_dir = experiment_dir / "plots"

    # Find data files
    data_files = find_data_files()
    if not data_files:
        print("ERROR: No data files found!")
        print("Expected: data/processed/preflib/00026-*.npz")
        return

    print(f"Found {len(data_files)} dataset(s)")
    print(f"Mask rates: {mask_rates}")
    print(f"Trials per rate: {n_trials}")
    print(f"Base seed: {base_seed}")

    # Run experiments
    all_results = {}

    for data_path in tqdm(data_files, desc="Datasets (MCAR)"):
        result = run_single_dataset_experiment(
            data_path,
            mask_rates=mask_rates,
            n_trials=n_trials,
            base_seed=base_seed,
            output_dir=output_dir,
            plots_dir=plots_dir,
        )
        all_results[result["name"]] = result

    # Compute aggregate statistics across all datasets
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total_bridging_wins = sum(r["summary"]["bridging_wins_spearman"] for r in all_results.values())
    total_polis_wins = sum(r["summary"]["polis_wins_spearman"] for r in all_results.values())
    total_ties = sum(r["summary"]["ties_spearman"] for r in all_results.values())

    print(f"Across all datasets and mask rates:")
    print(f"  Bridging wins (Spearman): {total_bridging_wins}")
    print(f"  Polis wins (Spearman): {total_polis_wins}")
    print(f"  Ties: {total_ties}")

    # Create combined aggregation for visualization
    combined_aggregated = _combine_aggregations(all_results, mask_rates)

    # Generate combined plots
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_robustness_comparison(
        combined_aggregated,
        plots_dir / "combined_spearman.png",
        metric="spearman",
        title="Combined: Spearman Correlation vs Observation Rate",
    )

    plot_multi_metric(
        combined_aggregated,
        plots_dir / "combined_multi_metric.png",
        metrics=["spearman", "rmse"],
        title="Combined: Robustness Comparison (Mean Across Datasets)",
    )

    plot_ranking_stability(
        combined_aggregated,
        plots_dir / "combined_ranking_stability.png",
        title="Combined: Ranking Stability",
    )

    create_summary_table(
        combined_aggregated,
        plots_dir / "combined_summary.md",
    )

    # Save all results
    save_path = save_results(
        {
            "datasets": all_results,
            "combined": combined_aggregated,
            "config": {
                "mask_rates": mask_rates,
                "n_trials": n_trials,
                "base_seed": base_seed,
            },
        },
        output_dir,
        "robustness_experiment",
    )
    print(f"\nResults saved to: {save_path}")
    print(f"Plots saved to: {plots_dir}/")


def _combine_aggregations(all_results: dict, mask_rates: list[float]) -> dict:
    """Combine aggregated results across datasets by averaging."""
    combined = {
        "mask_rates": mask_rates,
        "bridging": {},
        "polis": {},
    }

    for mask_rate in mask_rates:
        bridging_metrics = {}
        polis_metrics = {}

        for key in ["spearman_mean", "spearman_std", "rmse_mean", "rmse_std",
                    "mae_mean", "mae_std", "mean_estimate_variance",
                    "top_1_precision_mean", "top_1_precision_std",
                    "top_3_precision_mean", "top_3_precision_std",
                    "top_5_precision_mean", "top_5_precision_std",
                    # Ranking stability metrics
                    "stability_top_1_frequency", "stability_rank_correlation_mean",
                    "stability_top_1_jaccard", "stability_top_3_jaccard", "stability_top_5_jaccard"]:
            b_vals = []
            p_vals = []
            for r in all_results.values():
                agg = r["aggregated"]
                if mask_rate in agg["bridging"]:
                    if key in agg["bridging"][mask_rate]:
                        b_vals.append(agg["bridging"][mask_rate][key])
                    if key in agg["polis"][mask_rate]:
                        p_vals.append(agg["polis"][mask_rate][key])

            bridging_metrics[key] = np.mean(b_vals) if b_vals else np.nan
            polis_metrics[key] = np.mean(p_vals) if p_vals else np.nan

        # Handle k stats for polis
        k_vals = []
        for r in all_results.values():
            agg = r["aggregated"]
            if mask_rate in agg["polis"] and "k_mean" in agg["polis"][mask_rate]:
                k_vals.append(agg["polis"][mask_rate]["k_mean"])
        polis_metrics["k_mean"] = np.mean(k_vals) if k_vals else np.nan

        combined["bridging"][mask_rate] = bridging_metrics
        combined["polis"][mask_rate] = polis_metrics

    return combined


if __name__ == "__main__":
    # Run MCAR masking experiment
    run_full_experiment()

    # Run simulated Polis routing experiment
    print("\n" + "=" * 60)
    print("Now running simulated Polis routing experiment...")
    print("=" * 60 + "\n")
    run_full_simulation_experiment()
