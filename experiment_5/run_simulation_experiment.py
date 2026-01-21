#!/usr/bin/env python3
"""
Experiment 5b: Robustness Comparison Under Simulated Polis Routing

Compares robustness of three methods under realistic informative missingness:
1. Bridging Score (Naive) - no IPW correction
2. Bridging Score (IPW) - with Inverse Probability Weighting
3. Polis Group-Informed Consensus

Uses French election data (00026) with simulated Polis routing.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_5.robustness_simulated import (
    run_simulation_experiment,
    create_votes_distributions,
)
from experiment_5.evaluate import (
    aggregate_simulation_results,
    compute_simulation_summary,
)
from experiment_5.visualize import (
    plot_simulation_comparison,
    plot_simulation_multi_metric,
    create_simulation_summary_table,
)


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


def run_single_dataset_simulation(
    data_path: Path,
    n_trials: int,
    base_seed: int,
    output_dir: Path,
    plots_dir: Path,
) -> dict:
    """Run simulation experiment on a single dataset."""
    name = data_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    # Load data
    data = np.load(data_path)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    print(f"Matrix shape: {matrix.shape}")

    # Create votes distributions for different observation rates
    votes_distributions = create_votes_distributions(n_items)
    print(f"Testing {len(votes_distributions)} observation rate targets")

    # Run experiment
    results = run_simulation_experiment(
        matrix,
        votes_distributions,
        n_trials=n_trials,
        base_seed=base_seed,
        polis_max_k=5,
        mc_samples=100,
        show_progress=True,
    )

    # Aggregate results
    aggregated = aggregate_simulation_results(results)
    summary = compute_simulation_summary(aggregated)

    # Print summary
    print(f"\nSummary for {name}:")
    print(f"  Naive vs Polis: Naive wins {summary['naive_wins_vs_polis']}, "
          f"Polis wins {summary['polis_wins_vs_naive']}, Ties {summary['ties_naive_polis']}")
    print(f"  IPW vs Polis: IPW wins {summary['ipw_wins_vs_polis']}, "
          f"Polis wins {summary['polis_wins_vs_ipw']}, Ties {summary['ties_ipw_polis']}")
    print(f"  IPW vs Naive: IPW wins {summary['ipw_wins_vs_naive']}, "
          f"Naive wins {summary['naive_wins_vs_ipw']}, Ties {summary['ties_ipw_naive']}")

    # Generate plots
    dataset_plots_dir = plots_dir / f"{name}_simulated"
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)

    # Spearman comparison
    plot_simulation_comparison(
        aggregated,
        dataset_plots_dir / "spearman_comparison.png",
        metric="spearman",
        title=f"{name}: Spearman Correlation (Simulated Routing)",
    )

    # RMSE comparison
    plot_simulation_comparison(
        aggregated,
        dataset_plots_dir / "rmse_comparison.png",
        metric="rmse",
        title=f"{name}: RMSE (Simulated Routing)",
    )

    # Multi-metric
    plot_simulation_multi_metric(
        aggregated,
        dataset_plots_dir / "multi_metric.png",
        metrics=["spearman", "rmse"],
        title=f"{name}: Robustness Under Simulated Routing",
    )

    # Summary table
    create_simulation_summary_table(
        aggregated,
        dataset_plots_dir / "summary_table.md",
    )

    return {
        "name": name,
        "matrix_shape": [n_items, n_voters],
        "aggregated": aggregated,
        "summary": summary,
        "gt_bridging": results["gt_bridging"],
        "gt_polis": results["gt_polis"],
    }


def run_full_simulation_experiment():
    """Run the full simulation experiment on all French election datasets."""
    print("Experiment 5b: Robustness Comparison Under Simulated Polis Routing")
    print("=" * 60)

    # Configuration - reduced for speed
    n_trials = 15  # Reduced for faster results
    base_seed = 42
    max_datasets = 2  # Only run on first 2 datasets for speed

    # Output directories
    experiment_dir = Path(__file__).parent
    output_dir = experiment_dir / "results"
    plots_dir = experiment_dir / "plots"

    # Find data files
    data_files = find_data_files()
    if not data_files:
        print("ERROR: No data files found!")
        return

    # Limit datasets for speed
    data_files = data_files[:max_datasets]

    print(f"Running on {len(data_files)} dataset(s)")
    print(f"Trials per observation rate: {n_trials}")
    print(f"Base seed: {base_seed}")

    # Run experiments
    all_results = {}

    for data_path in data_files:
        result = run_single_dataset_simulation(
            data_path,
            n_trials=n_trials,
            base_seed=base_seed,
            output_dir=output_dir,
            plots_dir=plots_dir,
        )
        all_results[result["name"]] = result

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY (Simulated Routing)")
    print(f"{'='*60}")

    total_naive_wins = sum(r["summary"]["naive_wins_vs_polis"] for r in all_results.values())
    total_ipw_wins = sum(r["summary"]["ipw_wins_vs_polis"] for r in all_results.values())
    total_polis_wins_naive = sum(r["summary"]["polis_wins_vs_naive"] for r in all_results.values())
    total_polis_wins_ipw = sum(r["summary"]["polis_wins_vs_ipw"] for r in all_results.values())
    total_ipw_beats_naive = sum(r["summary"]["ipw_wins_vs_naive"] for r in all_results.values())
    total_naive_beats_ipw = sum(r["summary"]["naive_wins_vs_ipw"] for r in all_results.values())

    print("Across all datasets and observation rates:")
    print(f"  Naive vs Polis: Naive {total_naive_wins}, Polis {total_polis_wins_naive}")
    print(f"  IPW vs Polis: IPW {total_ipw_wins}, Polis {total_polis_wins_ipw}")
    print(f"  IPW vs Naive: IPW {total_ipw_beats_naive}, Naive {total_naive_beats_ipw}")

    # Combined results
    combined = _combine_simulation_results(all_results)

    # Combined plots
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_simulation_comparison(
        combined,
        plots_dir / "combined_simulated_spearman.png",
        metric="spearman",
        title="Combined: Spearman Correlation (Simulated Routing)",
    )

    plot_simulation_multi_metric(
        combined,
        plots_dir / "combined_simulated_multi_metric.png",
        metrics=["spearman", "rmse"],
        title="Combined: Robustness Under Simulated Routing",
    )

    create_simulation_summary_table(
        combined,
        plots_dir / "combined_simulated_summary.md",
    )

    # Save results
    save_path = save_results(
        {
            "datasets": all_results,
            "combined": combined,
            "config": {
                "n_trials": n_trials,
                "base_seed": base_seed,
            },
        },
        output_dir,
        "simulation_experiment",
    )
    print(f"\nResults saved to: {save_path}")
    print(f"Plots saved to: {plots_dir}/")


def _combine_simulation_results(all_results: dict) -> dict:
    """Combine results across datasets by averaging."""
    # Get all distribution names (should be the same across datasets)
    first_result = next(iter(all_results.values()))
    distribution_names = first_result["aggregated"]["distribution_names"]

    combined = {
        "distribution_names": distribution_names,
        "bridging_naive": {},
        "bridging_ipw": {},
        "polis": {},
    }

    metric_keys = [
        "spearman_mean", "spearman_std", "rmse_mean", "rmse_std",
        "mae_mean", "mae_std", "mean_estimate_variance",
        "observation_rate_mean", "observation_rate_std",
    ]

    for dist in distribution_names:
        for method in ["bridging_naive", "bridging_ipw", "polis"]:
            metrics = {}
            for key in metric_keys:
                vals = []
                for r in all_results.values():
                    if dist in r["aggregated"][method]:
                        if key in r["aggregated"][method][dist]:
                            vals.append(r["aggregated"][method][dist][key])
                metrics[key] = np.mean(vals) if vals else np.nan
            combined[method][dist] = metrics

    return combined


if __name__ == "__main__":
    run_full_simulation_experiment()
