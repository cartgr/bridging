"""
Experiment C: Real Polis data (00069).

Computes PD (naive), Pol.is GIC, and p-mean (p=-10, naive) on all
00069 sparse Polis datasets. Uses MDS on pairwise voter similarity
for voter positioning. Saves results to JSON.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.manifold import MDS
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_2.estimation import estimate_bridging_scores_naive, estimate_pnorm_naive
from experiment_5.polis import polis_consensus_pipeline
from experiment_6.comments import load_all_comments_for_dataset


def compute_voter_similarity_sparse(matrix, show_progress=True):
    """Pairwise voter similarity from co-observed agreement."""
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)
    similarity = np.zeros((n_voters, n_voters))
    voter_iter = range(n_voters)
    if show_progress:
        voter_iter = tqdm(voter_iter, desc="      Voter similarity", leave=False)
    for i in voter_iter:
        for j in range(i, n_voters):
            shared = observed[:, i] & observed[:, j]
            n_shared = shared.sum()
            if n_shared == 0:
                sim = 0.5
            else:
                agree = (matrix[shared, i] == matrix[shared, j]).sum()
                sim = agree / n_shared
            similarity[i, j] = sim
            similarity[j, i] = sim
    return similarity


def compute_mds_positions(similarity, seed=42):
    """MDS to 1D from similarity matrix."""
    distance = 1 - similarity
    np.fill_diagonal(distance, 0)
    mds = MDS(n_components=1, dissimilarity="precomputed", random_state=seed)
    return mds.fit_transform(distance).flatten()


def process_dataset(filepath, base_dir):
    """Process one 00069 dataset."""
    file_id = filepath.stem
    data = np.load(filepath)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)
    obs_rate = observed.sum() / observed.size
    print(f"  {file_id}: {n_items} items, {n_voters} voters, {obs_rate:.1%} observed")

    # Voter similarity and MDS
    print("    Computing voter positions...", end=" ", flush=True)
    similarity = compute_voter_similarity_sparse(matrix)
    positions = compute_mds_positions(similarity)
    print("done")

    # PD naive
    print("    Computing PD (naive)...", end=" ", flush=True)
    pd_scores = estimate_bridging_scores_naive(matrix, observed)
    print("done")

    # Polis GIC
    print("    Computing Pol.is GIC...", end=" ", flush=True)
    try:
        polis_scores, polis_meta = polis_consensus_pipeline(matrix, observed)
        polis_k = polis_meta["k_clusters"]
    except Exception as e:
        print(f"failed ({e})")
        polis_scores = np.zeros(n_items)
        polis_k = 0
    else:
        print(f"done (k={polis_k})")

    # p-mean naive
    print("    Computing p-mean (naive)...", end=" ", flush=True)
    pmean_scores = estimate_pnorm_naive(matrix, observed, p=-10)
    print("done")

    # Comment texts
    print("    Loading comments...", end=" ", flush=True)
    try:
        comment_texts = load_all_comments_for_dataset(file_id)
        print(f"loaded {len(comment_texts)}")
    except Exception as e:
        print(f"failed ({e})")
        comment_texts = {}

    # Approval fractions (among voters who voted)
    approval_fracs = []
    n_voted_list = []
    for c in range(n_items):
        voted = observed[c, :]
        n_voted = int(voted.sum())
        n_voted_list.append(n_voted)
        if n_voted > 0:
            approval_fracs.append(float((matrix[c, :] == 1)[voted].sum() / n_voted))
        else:
            approval_fracs.append(0.0)

    return {
        "file_id": file_id,
        "n_items": n_items,
        "n_voters": n_voters,
        "obs_rate": obs_rate,
        "voter_positions": positions.tolist(),
        "pd_scores": pd_scores.tolist(),
        "polis_scores": polis_scores.tolist(),
        "pmean_scores": pmean_scores.tolist(),
        "approval_fracs": approval_fracs,
        "n_voted": n_voted_list,
        "comment_texts": {str(k): v for k, v in comment_texts.items()},
        "polis_k": polis_k,
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "processed" / "preflib"
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    polis_files = sorted(data_dir.glob("00069-*.npz"))
    print(f"Processing {len(polis_files)} Polis datasets...")

    all_results = {}
    for i, filepath in enumerate(polis_files, 1):
        print(f"\n[{i}/{len(polis_files)}]")
        result = process_dataset(filepath, base_dir)
        all_results[result["file_id"]] = result

    output_path = results_dir / "experiment_c.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
