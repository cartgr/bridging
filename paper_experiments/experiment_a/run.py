"""
Experiment A: Qualitative rankings on French election data.

Computes PD, Pol.is GIC, and p-mean (p=-10) on fully-observed French
approval voting data. Saves scores and PCA voter positions to JSON.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_2.bridging import compute_bridging_scores_vectorized, compute_bridging_pnorm
from experiment_5.polis import polis_consensus_pipeline
from experiment_3.visualize import CANDIDATE_NAMES_2002, CANDIDATE_NAMES_2007


DATASETS = {
    "00026-combined": {
        "path": "data/processed/preflib/00026-combined.npz",
        "names": CANDIDATE_NAMES_2002,
        "label": "French Election 2002",
    },
    "00071-combined": {
        "path": "data/processed/preflib/00071-combined.npz",
        "names": CANDIDATE_NAMES_2007,
        "label": "French Election 2007",
    },
}


def compute_voter_pca(matrix: np.ndarray) -> np.ndarray:
    """Compute PC1 scores for voters."""
    voter_matrix = matrix.T
    centered = voter_matrix - voter_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return (U[:, 0] * S[0]).tolist()


def run_dataset(dataset_id: str, info: dict, base_dir: Path) -> dict:
    filepath = base_dir / info["path"]
    data = np.load(filepath)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    print(f"  {dataset_id}: {n_items} items, {n_voters} voters")

    # PD scores
    pd_scores = compute_bridging_scores_vectorized(matrix)

    # Pol.is GIC
    observed_mask = np.ones_like(matrix, dtype=bool)
    polis_scores, polis_meta = polis_consensus_pipeline(matrix, observed_mask)

    # p-mean (p=-10)
    pmean_scores = compute_bridging_pnorm(matrix, p=-10)

    # PCA voter positions
    pc1_scores = compute_voter_pca(matrix)

    # Approval fractions
    approval_fracs = matrix.mean(axis=1).tolist()

    return {
        "dataset_id": dataset_id,
        "label": info["label"],
        "n_items": n_items,
        "n_voters": n_voters,
        "candidate_names": info["names"],
        "pd_scores": pd_scores.tolist(),
        "polis_scores": polis_scores.tolist(),
        "pmean_scores": pmean_scores.tolist(),
        "approval_fracs": approval_fracs,
        "pc1_scores": pc1_scores,
        "polis_k": polis_meta["k_clusters"],
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, info in DATASETS.items():
        print(f"Processing {info['label']}...")
        all_results[dataset_id] = run_dataset(dataset_id, info, base_dir)

    output_path = results_dir / "experiment_a.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
