"""
Experiment D: Approval vs Approver Heterogeneity Scatter Plot.

For each French election dataset, compute:
1. Approval rate per candidate
2. Approver heterogeneity (average pairwise Hamming distance between approvers)
3. Bridging scores from PD, Pol.is GIC, and p-mean

Identifies the top candidate by each method.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_1.analyze import compute_approver_diversity
from experiment_2.bridging import compute_bridging_scores_vectorized, compute_bridging_pnorm
from experiment_3.visualize import CANDIDATE_NAMES_2002, CANDIDATE_NAMES_2007
from experiment_5.polis import polis_consensus_pipeline


DATASETS = {
    "00026-combined": {
        "path": "data/processed/preflib/00026-combined.npz",
        "label": "French Election 2002",
        "candidate_names": CANDIDATE_NAMES_2002,
    },
    "00071-combined": {
        "path": "data/processed/preflib/00071-combined.npz",
        "label": "French Election 2007",
        "candidate_names": CANDIDATE_NAMES_2007,
    },
}


def compute_approval_rates(matrix: np.ndarray) -> np.ndarray:
    """Compute approval rate for each candidate (row)."""
    return (matrix == 1.0).mean(axis=1)


def run_dataset(dataset_id: str, config: dict, base_dir: Path) -> dict:
    """Compute all metrics for one dataset."""
    data = np.load(base_dir / config["path"])
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape

    print(f"  {dataset_id}: {n_items} items, {n_voters} voters")

    # Compute approval rates
    approval_rates = compute_approval_rates(matrix)

    # Compute approver heterogeneity (diversity)
    heterogeneity = compute_approver_diversity(matrix)

    # Compute bridging scores
    pd_scores = compute_bridging_scores_vectorized(matrix)

    # Polis GIC
    observed_mask = np.ones_like(matrix, dtype=bool)
    try:
        polis_scores, _ = polis_consensus_pipeline(matrix, observed_mask)
    except Exception as e:
        print(f"    Polis failed: {e}")
        polis_scores = np.full(n_items, np.nan)

    # p-mean (p=-10)
    pmean_scores = compute_bridging_pnorm(matrix, p=-10)

    # Find top candidates by each method (handling NaN)
    def safe_argmax(scores):
        valid = ~np.isnan(scores)
        if not valid.any():
            return -1
        scores_safe = np.where(valid, scores, -np.inf)
        return int(np.argmax(scores_safe))

    top_pd = safe_argmax(pd_scores)
    top_polis = safe_argmax(polis_scores)
    top_pmean = safe_argmax(pmean_scores)

    return {
        "label": config["label"],
        "n_items": n_items,
        "n_voters": n_voters,
        "candidate_names": config["candidate_names"][:n_items],
        "approval_rates": approval_rates.tolist(),
        "heterogeneity": heterogeneity.tolist(),
        "pd_scores": pd_scores.tolist(),
        "polis_scores": polis_scores.tolist(),
        "pmean_scores": pmean_scores.tolist(),
        "top_pd": top_pd,
        "top_polis": top_polis,
        "top_pmean": top_pmean,
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, config in DATASETS.items():
        print(f"Processing {dataset_id}...")
        all_results[dataset_id] = run_dataset(dataset_id, config, base_dir)

    output_path = results_dir / "experiment_d.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
