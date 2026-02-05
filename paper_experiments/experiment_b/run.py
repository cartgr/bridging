"""
Experiment B: Robustness under missing data.

B1: MCAR masking at various rates, 50 trials each.
B2: Simulated Polis routing, 30 trials each.

Computes Spearman rank correlation vs ground truth for:
- PD (naive)
- Pol.is GIC
- p-mean (p=-10, naive)

No IPW correction used.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_2.bridging import compute_bridging_scores_vectorized, compute_bridging_pnorm
from experiment_2.estimation import estimate_bridging_scores_naive, estimate_pnorm_naive
from experiment_5.polis import polis_consensus_pipeline
from experiment_5.masking import apply_random_mask
from experiment_2.priority import compute_priorities, compute_pca_extremeness, compute_sampling_probabilities


DATASETS = {
    "00026-combined": "data/processed/preflib/00026-combined.npz",
    "00071-combined": "data/processed/preflib/00071-combined.npz",
}

# Observation rates from 5% to 95% in 5% increments
OBS_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MCAR_MASK_RATES = [1.0 - r for r in OBS_RATES]  # mask_rate = 1 - obs_rate
MCAR_N_TRIALS = 20
ROUTING_N_TRIALS = 20
ROUTING_TARGET_RATES = OBS_RATES


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


def compute_ground_truth(matrix: np.ndarray) -> Dict:
    """Compute ground truth scores on full matrix."""
    full_mask = np.ones_like(matrix, dtype=bool)
    gt_pd = compute_bridging_scores_vectorized(matrix)
    gt_polis, _ = polis_consensus_pipeline(matrix, full_mask)
    gt_pmean = compute_bridging_pnorm(matrix, p=-10)
    return {"pd": gt_pd, "polis": gt_polis, "pmean": gt_pmean}


def run_mcar_trial(matrix, mask_rate, seed, gt):
    """Run one MCAR trial."""
    masked_matrix, observed_mask = apply_random_mask(matrix, mask_rate, seed=seed)

    # PD naive
    pd_est = estimate_bridging_scores_naive(masked_matrix, observed_mask)
    # Polis GIC
    try:
        polis_est, _ = polis_consensus_pipeline(masked_matrix, observed_mask)
    except Exception:
        polis_est = np.full(matrix.shape[0], np.nan)
    # p-mean naive
    pmean_est = estimate_pnorm_naive(masked_matrix, observed_mask, p=-10)

    obs_rate = observed_mask.sum() / observed_mask.size

    return {
        "obs_rate": obs_rate,
        "spearman_pd": safe_spearman(pd_est, gt["pd"]),
        "spearman_polis": safe_spearman(polis_est, gt["polis"]),
        "spearman_pmean": safe_spearman(pmean_est, gt["pmean"]),
        "kendall_pd": safe_kendall(pd_est, gt["pd"]),
        "kendall_polis": safe_kendall(polis_est, gt["polis"]),
        "kendall_pmean": safe_kendall(pmean_est, gt["pmean"]),
        "top1_pd": top1_match(pd_est, gt["pd"]),
        "top1_polis": top1_match(polis_est, gt["polis"]),
        "top1_pmean": top1_match(pmean_est, gt["pmean"]),
    }


def simulate_polis_routing_naive(ground_truth, target_obs_rate, seed):
    """Simulate Polis routing and compute naive scores (no IPW)."""
    rng = np.random.default_rng(seed)
    n_items, n_voters = ground_truth.shape

    target_k = max(1, int(n_items * target_obs_rate))
    votes_dist = np.arange(max(1, target_k - 2), min(n_items, target_k + 2) + 1)

    observed_matrix = np.full((n_items, n_voters), np.nan)
    observed_mask = np.zeros((n_items, n_voters), dtype=bool)

    for voter_idx in tqdm(range(n_voters), desc="        Voters", leave=False):
        extremeness = compute_pca_extremeness(observed_matrix, observed_mask)
        priorities = compute_priorities(observed_matrix, observed_mask, extremeness)
        k_votes = min(rng.choice(votes_dist), n_items)

        eligible = np.ones(n_items, dtype=bool)
        for _ in range(k_votes):
            p = compute_sampling_probabilities(priorities, eligible)
            if p.sum() == 0:
                break
            sampled = rng.choice(n_items, p=p)
            eligible[sampled] = False
            observed_mask[sampled, voter_idx] = True
            observed_matrix[sampled, voter_idx] = ground_truth[sampled, voter_idx]

    return observed_matrix, observed_mask


def run_routing_trial(matrix, target_obs_rate, seed, gt):
    """Run one simulated routing trial."""
    observed_matrix, observed_mask = simulate_polis_routing_naive(
        matrix, target_obs_rate, seed
    )

    obs_rate = observed_mask.sum() / observed_mask.size

    # PD naive
    pd_est = estimate_bridging_scores_naive(observed_matrix, observed_mask)
    # Polis GIC
    try:
        polis_est, _ = polis_consensus_pipeline(observed_matrix, observed_mask)
    except Exception:
        polis_est = np.full(matrix.shape[0], np.nan)
    # p-mean naive
    pmean_est = estimate_pnorm_naive(observed_matrix, observed_mask, p=-10)

    return {
        "target_obs_rate": target_obs_rate,
        "actual_obs_rate": obs_rate,
        "spearman_pd": safe_spearman(pd_est, gt["pd"]),
        "spearman_polis": safe_spearman(polis_est, gt["polis"]),
        "spearman_pmean": safe_spearman(pmean_est, gt["pmean"]),
        "kendall_pd": safe_kendall(pd_est, gt["pd"]),
        "kendall_polis": safe_kendall(polis_est, gt["polis"]),
        "kendall_pmean": safe_kendall(pmean_est, gt["pmean"]),
        "top1_pd": top1_match(pd_est, gt["pd"]),
        "top1_polis": top1_match(polis_est, gt["polis"]),
        "top1_pmean": top1_match(pmean_est, gt["pmean"]),
    }


def run_dataset(dataset_id, filepath, base_dir):
    """Run all experiments for one dataset."""
    data = np.load(base_dir / filepath)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    print(f"  {dataset_id}: {n_items} items, {n_voters} voters")

    print("  Computing ground truth...")
    gt = compute_ground_truth(matrix)

    # B1: MCAR
    print("  Running MCAR trials...")
    mcar_results = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=MCAR_N_TRIALS)

    total_mcar = len(MCAR_MASK_RATES) * MCAR_N_TRIALS
    pbar = tqdm(total=total_mcar, desc="    MCAR trials")
    for mask_rate in MCAR_MASK_RATES:
        for trial in range(MCAR_N_TRIALS):
            result = run_mcar_trial(matrix, mask_rate, int(seeds[trial]), gt)
            result["mask_rate"] = mask_rate
            result["trial"] = trial
            mcar_results.append(result)
            pbar.update(1)
            pbar.set_postfix(mask=f"{mask_rate:.1f}", trial=trial)
    pbar.close()

    # B2: Simulated routing
    print("  Running routing trials...")
    routing_results = []
    seeds = rng.integers(0, 2**31, size=ROUTING_N_TRIALS)

    total_routing = len(ROUTING_TARGET_RATES) * ROUTING_N_TRIALS
    pbar = tqdm(total=total_routing, desc="    Routing trials")
    for target_rate in ROUTING_TARGET_RATES:
        for trial in range(ROUTING_N_TRIALS):
            result = run_routing_trial(matrix, target_rate, int(seeds[trial]), gt)
            result["trial"] = trial
            routing_results.append(result)
            pbar.update(1)
            pbar.set_postfix(rate=f"{target_rate:.1f}", trial=trial)
    pbar.close()

    return {
        "dataset_id": dataset_id,
        "n_items": n_items,
        "n_voters": n_voters,
        "gt_pd": gt["pd"].tolist(),
        "gt_polis": gt["polis"].tolist(),
        "gt_pmean": gt["pmean"].tolist(),
        "mcar_results": mcar_results,
        "routing_results": routing_results,
    }


def main():
    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, filepath in DATASETS.items():
        print(f"Processing {dataset_id}...")
        all_results[dataset_id] = run_dataset(dataset_id, filepath, base_dir)

    output_path = results_dir / "experiment_b.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
