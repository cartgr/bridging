"""
Experiment F: p-mean Robustness under MCAR.

Tests robustness of various p-mean values under Missing Completely At Random
(MCAR) masking. Shows how different values of p perform when data is sparse.

P values: 1, 0, -1, -2, -5, -10, -inf
Observation rates: 5%, 25%, 50%, 75%, 95%
Trials: 20 per observation rate
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_2.bridging import compute_bridging_pnorm
from experiment_2.estimation import estimate_pnorm_naive
from experiment_5.masking import apply_random_mask


DATASETS = {
    "00026-combined": "data/processed/preflib/00026-combined.npz",
    "00071-combined": "data/processed/preflib/00071-combined.npz",
}

# P values to test (from high to low)
P_VALUES = [1, 0, -1, -2, -5, -10, float('-inf')]

# Observation rates
OBS_RATES = [0.05, 0.25, 0.50, 0.75, 0.95]
MASK_RATES = [1.0 - r for r in OBS_RATES]

N_TRIALS = 20


def p_to_key(p: float) -> str:
    """Convert p value to JSON-safe string key."""
    if p == float('-inf'):
        return "-inf"
    elif p == float('inf'):
        return "inf"
    else:
        return str(int(p)) if p == int(p) else str(p)


def safe_spearman(estimated, ground_truth):
    """Spearman correlation handling NaN values."""
    valid = ~np.isnan(estimated) & ~np.isnan(ground_truth)
    if valid.sum() < 3:
        return np.nan
    rho, _ = spearmanr(estimated[valid], ground_truth[valid])
    return rho


def safe_kendall(estimated, ground_truth):
    """Kendall tau handling NaN values."""
    valid = ~np.isnan(estimated) & ~np.isnan(ground_truth)
    if valid.sum() < 3:
        return np.nan
    tau, _ = kendalltau(estimated[valid], ground_truth[valid])
    return tau


def top1_match(estimated, ground_truth):
    """Check if the top-1 item matches between estimated and ground truth."""
    valid = ~np.isnan(estimated) & ~np.isnan(ground_truth)
    if valid.sum() < 2:
        return np.nan
    est_valid = np.where(valid, estimated, -np.inf)
    gt_valid = np.where(valid, ground_truth, -np.inf)
    return float(np.argmax(est_valid) == np.argmax(gt_valid))


def compute_ground_truth(matrix: np.ndarray) -> dict:
    """Compute ground truth p-mean scores for all p values."""
    gt = {}
    for p in P_VALUES:
        p_key = p_to_key(p)
        gt[p_key] = compute_bridging_pnorm(matrix, p=p)
    return gt


def run_mcar_trial(matrix, mask_rate, seed, gt):
    """Run one MCAR trial for all p values."""
    masked_matrix, observed_mask = apply_random_mask(matrix, mask_rate, seed=seed)
    obs_rate = observed_mask.sum() / observed_mask.size

    result = {
        "obs_rate": obs_rate,
    }

    for p in P_VALUES:
        p_key = p_to_key(p)
        gt_scores = gt[p_key]

        # Estimate p-mean with naive estimator
        est_scores = estimate_pnorm_naive(masked_matrix, observed_mask, p=p)

        # Compute metrics
        result[f"spearman_{p_key}"] = safe_spearman(est_scores, gt_scores)
        result[f"kendall_{p_key}"] = safe_kendall(est_scores, gt_scores)
        result[f"top1_{p_key}"] = top1_match(est_scores, gt_scores)

    return result


def run_dataset(dataset_id, filepath, base_dir):
    """Run all MCAR trials for one dataset."""
    data = np.load(base_dir / filepath)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    print(f"  {dataset_id}: {n_items} items, {n_voters} voters")

    print("  Computing ground truth for all p values...")
    gt = compute_ground_truth(matrix)

    # Convert ground truth to serializable format
    gt_serializable = {k: v.tolist() for k, v in gt.items()}

    print("  Running MCAR trials...")
    mcar_results = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=N_TRIALS)

    total_trials = len(MASK_RATES) * N_TRIALS
    pbar = tqdm(total=total_trials, desc="    MCAR trials")

    for mask_rate in MASK_RATES:
        for trial in range(N_TRIALS):
            result = run_mcar_trial(matrix, mask_rate, int(seeds[trial]), gt)
            result["mask_rate"] = mask_rate
            result["trial"] = trial
            mcar_results.append(result)
            pbar.update(1)
            pbar.set_postfix(mask=f"{mask_rate:.2f}", trial=trial)

    pbar.close()

    return {
        "dataset_id": dataset_id,
        "n_items": n_items,
        "n_voters": n_voters,
        "ground_truth": gt_serializable,
        "mcar_results": mcar_results,
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, filepath in DATASETS.items():
        print(f"Processing {dataset_id}...")
        all_results[dataset_id] = run_dataset(dataset_id, filepath, base_dir)

    output_path = results_dir / "experiment_f.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
