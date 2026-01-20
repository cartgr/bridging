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
    """Complete a sparse matrix using KNNBaseline."""
    n_items, n_voters = matrix.shape
    observed = ~np.isnan(matrix)
    rows, cols = np.where(observed)
    ratings = matrix[observed]

    df = pd.DataFrame({"voter": cols, "item": rows, "rating": ratings})
    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df[["voter", "item", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = KNNBaseline(k=k, verbose=False)
    algo.fit(trainset)

    completed = np.zeros_like(matrix)
    for i in range(n_items):
        for j in range(n_voters):
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
