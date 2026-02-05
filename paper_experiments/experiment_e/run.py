"""
Experiment E: p-mean Parameter Sensitivity Analysis.

For each French election dataset, compute p-mean bridging scores
for various values of p to analyze how the choice of p affects
which candidate is selected as "most bridging".

Test values: p = 1, 0, -1, -2, -5, -10, -inf
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_1.analyze import compute_approver_diversity
from experiment_2.bridging import compute_bridging_pnorm
from experiment_3.visualize import CANDIDATE_NAMES_2002, CANDIDATE_NAMES_2007


# P values to test (from high to low)
P_VALUES = [1, 0, -1, -2, -5, -10, float('-inf')]
P_LABELS = ["p=1", "p=0", "p=-1", "p=-2", "p=-5", "p=-10", r"p=-\infty"]

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


def safe_argmax(scores: np.ndarray) -> int:
    """Find argmax handling NaN values."""
    valid = ~np.isnan(scores)
    if not valid.any():
        return -1
    scores_safe = np.where(valid, scores, -np.inf)
    return int(np.argmax(scores_safe))


def p_to_key(p: float) -> str:
    """Convert p value to JSON-safe string key."""
    if p == float('-inf'):
        return "-inf"
    elif p == float('inf'):
        return "inf"
    else:
        return str(int(p)) if p == int(p) else str(p)


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

    # Compute p-mean scores for each p value
    pmean_scores = {}
    top_by_p = {}

    for p in P_VALUES:
        p_key = p_to_key(p)
        scores = compute_bridging_pnorm(matrix, p=p)
        pmean_scores[p_key] = scores.tolist()
        top_by_p[p_key] = safe_argmax(scores)
        print(f"    p={p_key}: top candidate = {top_by_p[p_key]} ({config['candidate_names'][top_by_p[p_key]]})")

    return {
        "label": config["label"],
        "n_items": n_items,
        "n_voters": n_voters,
        "candidate_names": config["candidate_names"][:n_items],
        "approval_rates": approval_rates.tolist(),
        "heterogeneity": heterogeneity.tolist(),
        "pmean_scores": pmean_scores,
        "top_by_p": top_by_p,
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, config in DATASETS.items():
        print(f"Processing {dataset_id}...")
        all_results[dataset_id] = run_dataset(dataset_id, config, base_dir)

    output_path = results_dir / "experiment_e.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
