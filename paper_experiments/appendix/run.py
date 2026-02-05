"""
Appendix Experiment: Naive PD vs IPW-Corrected PD under MCAR.

Compares the naive PD estimator against a properly IPW-corrected PD
estimator when observation rates are known (MCAR assumption).

Mathematical Background:
The full PD formula is:
    theta_PD(x) = (4/mn^2) * sum_y sum_{i<j} 1[A_{i,x}=1, A_{j,x}=1] * 1[A_{i,y}!=A_{j,y}]

Under missing data, each term requires observing 4 entries: (i,x), (j,x), (i,y), (j,y).
Under MCAR with observation rate q, the IPW weight is 1/q^4.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_2.bridging import compute_bridging_scores_vectorized
from experiment_2.estimation import estimate_bridging_scores_naive
from experiment_5.masking import apply_random_mask


DATASETS = {
    "00026-combined": "data/processed/preflib/00026-combined.npz",
    "00071-combined": "data/processed/preflib/00071-combined.npz",
}

# 5 observation rates as specified in the plan
OBS_RATES = [0.05, 0.275, 0.50, 0.725, 0.95]
N_TRIALS = 20


def estimate_bridging_scores_ipw_mcar_slow(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    obs_rate: float,
) -> np.ndarray:
    """
    IPW-corrected PD under MCAR assumption (SLOW reference implementation).

    O(n_items * n_approvers^2 * n_items) - use only for testing.
    """
    n_items, n_voters = observed_matrix.shape
    weight = 1.0 / (obs_rate ** 4)
    normalization = 4.0 / (n_items * n_voters ** 2)

    bridging_scores = np.zeros(n_items)

    for x in range(n_items):
        total = 0.0
        approvers_x = np.where(observed_mask[x, :] & (observed_matrix[x, :] == 1.0))[0]

        if len(approvers_x) < 2:
            continue

        for idx_i in range(len(approvers_x)):
            for idx_j in range(idx_i + 1, len(approvers_x)):
                i, j = approvers_x[idx_i], approvers_x[idx_j]

                for y in range(n_items):
                    if y == x:
                        continue
                    if not (observed_mask[y, i] and observed_mask[y, j]):
                        continue
                    v_i_y = observed_matrix[y, i]
                    v_j_y = observed_matrix[y, j]
                    if (v_i_y == 1.0 and v_j_y == 0.0) or (v_i_y == 0.0 and v_j_y == 1.0):
                        total += weight

        bridging_scores[x] = normalization * total

    return bridging_scores


def estimate_bridging_scores_ipw_mcar(
    observed_matrix: np.ndarray,
    observed_mask: np.ndarray,
    obs_rate: float,
) -> np.ndarray:
    """
    IPW-corrected PD under MCAR assumption (FAST vectorized implementation).

    For each comment x:
        theta_PD(x) = (4/mn^2) * sum_y sum_{i<j: all 4 observed}
                      1[A_{i,x}=1, A_{j,x}=1] * 1[A_{i,y}!=A_{j,y}] / q^4

    Key insight: The number of disagreeing pairs on y among approvers of x equals:
        |{i in approvers_x : obs[y,i] AND matrix[y,i]=1}| *
        |{j in approvers_x : obs[y,j] AND matrix[y,j]=0}|

    This allows vectorization over voter pairs.

    Complexity: O(n_items^2 * n_voters) with vectorized numpy operations.

    Args:
        observed_matrix: (n_items, n_voters) array with observed votes (NaN for missing)
        observed_mask: (n_items, n_voters) boolean array, True where observed
        obs_rate: known observation rate under MCAR

    Returns:
        (n_items,) array of IPW-corrected bridging scores
    """
    n_items, n_voters = observed_matrix.shape
    weight = 1.0 / (obs_rate ** 4)
    normalization = 4.0 / (n_items * n_voters ** 2)

    bridging_scores = np.zeros(n_items)

    # Precompute: where voters approved (observed AND voted 1)
    approved = observed_mask & (observed_matrix == 1.0)  # (n_items, n_voters)
    # Precompute: where voters disapproved (observed AND voted 0)
    disapproved = observed_mask & (observed_matrix == 0.0)  # (n_items, n_voters)

    for x in range(n_items):
        # Boolean mask of approvers of x
        approvers_x = approved[x, :]  # (n_voters,)

        if approvers_x.sum() < 2:
            continue

        # For each y: which approvers of x also observed y?
        # observed_mask[y, :] & approvers_x gives voters who: approved x AND observed y
        approvers_x_obs_y = observed_mask & approvers_x  # (n_items, n_voters)

        # Among approvers of x who observed y: how many approved y?
        n_approve_y = (approvers_x_obs_y & approved).sum(axis=1)  # (n_items,)

        # Among approvers of x who observed y: how many disapproved y?
        n_disapprove_y = (approvers_x_obs_y & disapproved).sum(axis=1)  # (n_items,)

        # Number of disagreeing pairs on y = |approve| * |disapprove|
        disagreeing_pairs = n_approve_y * n_disapprove_y  # (n_items,)

        # Exclude y == x
        disagreeing_pairs[x] = 0

        # Sum over all y, apply weight and normalization
        bridging_scores[x] = normalization * weight * disagreeing_pairs.sum()

    return bridging_scores


def test_ipw_implementations():
    """Test that slow and fast IPW implementations produce identical results."""
    print("Testing IPW implementations...")

    # Create small test matrix
    np.random.seed(42)
    n_items, n_voters = 8, 50
    matrix = np.random.choice([0.0, 1.0], size=(n_items, n_voters))

    # Create random mask (50% observed)
    mask = np.random.random((n_items, n_voters)) < 0.5
    # Ensure minimum observations
    for i in range(n_items):
        if mask[i].sum() < 2:
            mask[i, :2] = True
    for j in range(n_voters):
        if mask[:, j].sum() < 1:
            mask[0, j] = True

    masked_matrix = np.where(mask, matrix, np.nan)

    # Test with different observation rates
    for obs_rate in [0.3, 0.5, 0.7]:
        slow_scores = estimate_bridging_scores_ipw_mcar_slow(masked_matrix, mask, obs_rate)
        fast_scores = estimate_bridging_scores_ipw_mcar(masked_matrix, mask, obs_rate)

        max_diff = np.abs(slow_scores - fast_scores).max()
        assert max_diff < 1e-10, f"Mismatch at obs_rate={obs_rate}: max_diff={max_diff}"
        print(f"  obs_rate={obs_rate}: max_diff={max_diff:.2e} OK")

    # Test on slightly larger matrix
    n_items, n_voters = 12, 100
    matrix = np.random.choice([0.0, 1.0], size=(n_items, n_voters))
    mask = np.random.random((n_items, n_voters)) < 0.6
    for i in range(n_items):
        if mask[i].sum() < 2:
            mask[i, :2] = True
    masked_matrix = np.where(mask, matrix, np.nan)

    slow_scores = estimate_bridging_scores_ipw_mcar_slow(masked_matrix, mask, 0.6)
    fast_scores = estimate_bridging_scores_ipw_mcar(masked_matrix, mask, 0.6)
    max_diff = np.abs(slow_scores - fast_scores).max()
    assert max_diff < 1e-10, f"Mismatch on larger matrix: max_diff={max_diff}"
    print(f"  larger matrix (12x100): max_diff={max_diff:.2e} OK")

    print("All tests passed!")
    return True


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
    """Check if the top-1 item matches."""
    valid = ~np.isnan(estimated) & ~np.isnan(ground_truth)
    if valid.sum() < 2:
        return np.nan
    est_valid = np.where(valid, estimated, -np.inf)
    gt_valid = np.where(valid, ground_truth, -np.inf)
    return float(np.argmax(est_valid) == np.argmax(gt_valid))


def run_trial(matrix, obs_rate, seed, gt_pd):
    """Run one MCAR trial comparing naive vs IPW."""
    mask_rate = 1.0 - obs_rate
    masked_matrix, observed_mask = apply_random_mask(matrix, mask_rate, seed=seed)

    # Actual observation rate (may differ slightly due to constraints)
    actual_obs_rate = observed_mask.sum() / observed_mask.size

    # Naive PD
    pd_naive = estimate_bridging_scores_naive(masked_matrix, observed_mask)

    # IPW PD (using the target obs_rate as the known probability)
    pd_ipw = estimate_bridging_scores_ipw_mcar(masked_matrix, observed_mask, obs_rate)

    return {
        "target_obs_rate": obs_rate,
        "actual_obs_rate": actual_obs_rate,
        "spearman_naive": safe_spearman(pd_naive, gt_pd),
        "spearman_ipw": safe_spearman(pd_ipw, gt_pd),
        "kendall_naive": safe_kendall(pd_naive, gt_pd),
        "kendall_ipw": safe_kendall(pd_ipw, gt_pd),
        "top1_naive": top1_match(pd_naive, gt_pd),
        "top1_ipw": top1_match(pd_ipw, gt_pd),
    }


def run_dataset(dataset_id, filepath, base_dir):
    """Run appendix experiment for one dataset."""
    data = np.load(base_dir / filepath)
    matrix = data["matrix"]
    n_items, n_voters = matrix.shape
    print(f"  {dataset_id}: {n_items} items, {n_voters} voters")

    # Ground truth PD
    print("  Computing ground truth PD...")
    gt_pd = compute_bridging_scores_vectorized(matrix)

    # Run trials
    print("  Running MCAR trials (naive vs IPW)...")
    results = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=N_TRIALS)

    total_trials = len(OBS_RATES) * N_TRIALS
    pbar = tqdm(total=total_trials, desc="    Trials")
    for obs_rate in OBS_RATES:
        for trial in range(N_TRIALS):
            result = run_trial(matrix, obs_rate, int(seeds[trial]), gt_pd)
            result["trial"] = trial
            results.append(result)
            pbar.update(1)
            pbar.set_postfix(rate=f"{obs_rate:.2f}", trial=trial)
    pbar.close()

    return {
        "dataset_id": dataset_id,
        "n_items": n_items,
        "n_voters": n_voters,
        "gt_pd": gt_pd.tolist(),
        "results": results,
    }


def main():
    # Run tests first to verify implementations match
    test_ipw_implementations()
    print()

    base_dir = Path(__file__).parent.parent.parent
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for dataset_id, filepath in DATASETS.items():
        print(f"Processing {dataset_id}...")
        all_results[dataset_id] = run_dataset(dataset_id, filepath, base_dir)

    output_path = results_dir / "appendix.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
