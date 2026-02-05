"""
Complete sparse matrices using KNNBaseline collaborative filtering.

Usage:
    python data/complete.py
    python data/complete.py --k 40
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from surprise import KNNBaseline, Dataset, Reader

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed" / "preflib"
COMPLETED_DIR = DATA_DIR / "completed"


def complete_matrix(matrix: np.ndarray, k: int = 40) -> np.ndarray:
    """Complete a sparse matrix using KNNBaseline.

    Pass votes (0.5) are preserved as-is since they represent "saw but chose not to vote".
    Only truly unobserved entries (NaN) are imputed.
    The CF model is trained only on agree (1.0) and disagree (0.0) votes.
    """
    n_items, n_voters = matrix.shape

    # Only use agree (1.0) and disagree (0.0) for training
    # Pass votes (0.5) are excluded from training since CF expects binary data
    binary_mask = (matrix == 1.0) | (matrix == 0.0)
    rows, cols = np.where(binary_mask)
    ratings = matrix[binary_mask]

    df = pd.DataFrame({"voter": cols, "item": rows, "rating": ratings})
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["voter", "item", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = KNNBaseline(k=k, verbose=False)
    algo.fit(trainset)

    # Start with original matrix
    completed = matrix.copy()

    # Only impute truly unobserved entries (NaN)
    # Pass votes (0.5) are preserved
    for i in range(n_items):
        for j in range(n_voters):
            if np.isnan(matrix[i, j]):
                pred = algo.predict(j, i, verbose=False)
                completed[i, j] = 1.0 if pred.est >= 0.5 else 0.0

    return completed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Complete sparse matrices")
    parser.add_argument("--k", type=int, default=40, help="Number of neighbors")
    args = parser.parse_args()

    polis_files = sorted(PROCESSED_DIR.glob("00069-*.npz"))
    if not polis_files:
        print("No Pol.is datasets found. Run process.py first.")
        return

    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Completing {len(polis_files)} Pol.is datasets (k={args.k})...")

    for path in tqdm(polis_files, desc="Completing"):
        data = np.load(path, allow_pickle=True)
        matrix = data["matrix"]
        completed = complete_matrix(matrix, k=args.k)

        np.savez(
            COMPLETED_DIR / path.name,
            matrix=completed,
            n_items=matrix.shape[0],
            n_voters=matrix.shape[1],
            original_sparsity=float(data["sparsity"]),
        )

    print(f"Output: {COMPLETED_DIR}/")


if __name__ == "__main__":
    main()
