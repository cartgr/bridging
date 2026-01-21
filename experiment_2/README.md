# Experiment 2: Bridging Score Estimation Under Pol.is Sampling

This experiment tests how well we can estimate bridging scores from partially observed Pol.is data using inverse-probability weighting (IPW).

## Overview

Pol.is uses an adaptive routing algorithm to decide which comments to show each voter. This creates **informative missingness**: the probability of observing a vote depends on the current state of the vote matrix (through the priority formula). Naive estimation methods that ignore this can produce biased results.

We simulate the Pol.is routing process on complete (ground truth) data, then estimate bridging scores using IPW to correct for the sampling bias.

## Key Formulas

### Bridging Score (Pairwise Disagreement)

For comment $c$ with approvers $N_c$:

$$b^{PD}(c) = \frac{4}{n^2} \sum_{i<j, \, i,j \in N_c} d_{ij}$$

where $d_{ij} = \frac{1}{|C|} \sum_{c' \in C} \mathbf{1}[i \text{ and } j \text{ disagree on } c']$

The $4/n^2$ normalization ensures $b^{PD}(c) \in [0, 1]$.

A high bridging score indicates that the comment is approved by voters who otherwise disagree with each other.

### Pol.is Priority Formula

From the [Pol.is source code](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj):

```
importance = (1 - p) × (E + 1) × a
priority = [importance × (1 + 8 × 2^(-S/5))]²
```

Where (with Laplace smoothing):
- `a = (A + 1) / (S + 2)`: smoothed agree rate
- `p = (P + 1) / (S + 2)`: smoothed pass rate (P=0 in our binary setting)
- `E`: extremeness in PCA space (L2 norm of loadings in first 2 PCs)
- `S`: total votes on comment
- `A`: agree votes on comment

### PCA Extremeness

Following the Pol.is implementation:
1. **Participant filtering**: Include voters with at least `min(7, n_comments)` votes. If fewer than 15 qualify, greedily add top contributors.
2. **Imputation**: Missing values imputed with per-comment (column) mean
3. **PCA method**: Power iteration (matching `pca.clj`)
4. **Extremeness**: L2 norm of each comment's loading on the first 2 principal components

### IPW Estimation

The IPW estimator corrects for biased sampling:

$$\hat{d}_{ij} = \frac{1}{|C|} \sum_{c' \text{ observed by both}} \frac{\mathbf{1}[\text{disagree}]}{\pi_{i,c'} \times \pi_{j,c'}}$$

where $\pi_{i,c}$ is the inclusion probability (probability that voter $i$ sees comment $c$).

## Data Sources

- **Ground truth**: `data/completed/00069-*.npz` (20 files, fully observed)
- **Vote distribution**: `data/processed/preflib/00069-*.npz` (per-dataset votes-per-voter)

Each dataset uses its own vote distribution from the corresponding processed file.

Matrix format: `(n_items, n_voters)`, values `1.0` (approved), `0.0` (disapproved), `NaN` (missing)

## Module Structure

```
experiment_2/
├── README.md              # This file
├── __init__.py            # Package exports
├── bridging.py            # Ground truth bridging score computation
├── priority.py            # Pol.is priority formula and PCA extremeness
├── simulation.py          # Pol.is routing simulation with exact probability tracking
├── estimation.py          # IPW bridging score estimation
├── evaluate.py            # Comparison metrics
├── run_experiment.py      # Main experiment runner
├── tests/
│   ├── test_bridging.py
│   ├── test_priority.py
│   ├── test_simulation.py
│   ├── test_estimation.py
│   └── test_integration.py
└── results/               # Output files
```

## Usage

### Run the Full Experiment

```bash
python -m experiment_2.run_experiment
```

Options:
- `--completed-dir`: Ground truth data directory (default: `data/completed`)
- `--processed-dir`: Processed data directory for vote distributions (default: `data/processed/preflib`)
- `--output-dir`: Output directory (default: `experiment_2/results`)
- `--seed`: Random seed (default: `42`)
- `--quiet`: Suppress progress output

### Run Tests

```bash
# All tests
pytest experiment_2/tests/ -v

# Skip slow Monte Carlo tests
pytest experiment_2/tests/ -v -m "not slow"

# Specific module
pytest experiment_2/tests/test_bridging.py -v
```

### Use Individual Modules

```python
import numpy as np
from experiment_2 import (
    compute_bridging_scores_vectorized,
    simulate_polis_routing,
    estimate_bridging_scores_ipw,
    evaluate_estimation,
)

# Load data
completed = np.load("data/completed/00069-00000001.npz")
processed = np.load("data/processed/preflib/00069-00000001.npz")

ground_truth = completed["matrix"]
votes_dist = (~np.isnan(processed["matrix"])).sum(axis=0)  # Votes per voter

# Compute ground truth bridging scores
true_scores = compute_bridging_scores_vectorized(ground_truth)

# Simulate Pol.is routing
observed_mask, inclusion_probs = simulate_polis_routing(
    ground_truth, votes_dist, seed=42
)

# Create observed matrix
observed_matrix = np.where(observed_mask, ground_truth, np.nan)

# Estimate bridging scores with IPW
ipw_scores = estimate_bridging_scores_ipw(
    observed_matrix, observed_mask, inclusion_probs
)

# Evaluate
metrics = evaluate_estimation(true_scores, ipw_scores)
print(f"Spearman correlation: {metrics['spearman_correlation']:.3f}")
```

## Evaluation Metrics

- **Spearman correlation**: Rank correlation between true and estimated scores
- **Kendall's tau**: Alternative rank correlation
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **Top-k precision/recall**: Recovery of the truly highest bridging comments

## Technical Details

### Exact Inclusion Probability Computation

For PPS (probability proportional to size) sampling without replacement, inclusion probabilities are computed using the recursive formula:

```
π_c(S, k) = p_c(S) + Σ_{j∈S, j≠c} p_j(S) × π_c(S\{j}, k-1)
```

where `p_i(S) = Priority[i] / Σ_{j∈S} Priority[j]`

This gives the **true marginal probability** P(comment c is shown) before knowing which comments are actually sampled - which is what IPW requires.

**Complexity:** O(Σ C(n,j) for j=0..k) states, which is exponential. For tractability:
- **Exact computation** when estimated states ≤ 500k (e.g., n≤25, k≤12)
- **Monte Carlo fallback** (2000 samples) for larger problems

### Assumptions and Simplifications

| Assumption | Description | Impact |
|------------|-------------|--------|
| **Independence** | $\pi_{ij,c} = \pi_{i,c} \times \pi_{j,c}$ | Approximation; voters are processed sequentially so matrix state differs |
| **Voter ordering** | Process voters in index order (0, 1, 2, ...) | Real Pol.is has asynchronous arrivals |
| **No skip behavior** | All shown comments receive a vote | Real Pol.is allows pass/skip |
| **Binary votes** | Only agree (1) / disagree (0), no pass | Real Pol.is has agree/disagree/pass |
| **Ground truth as oracle** | Revealed votes come from completed matrix | Assumes matrix completion is accurate |
| **Probability clipping** | Clip $\pi$ to min $10^{-6}$ for IPW stability | Prevents extreme weights |
| **Monte Carlo fallback** | Use 2000 samples when exact computation infeasible | Small approximation error for large n×k |

### What We Match from Pol.is

- Laplace smoothing: `a = (A+1)/(S+2)`, `p = (P+1)/(S+2)`
- Vote factor: `1 + 8 × 2^(-S/5)`
- Squared priority
- Per-comment mean imputation for PCA
- Participant filtering (min 7 votes, min 15 participants)
- Power iteration PCA

## Output Format

Results are saved as JSON:
- `results_YYYYMMDD_HHMMSS.json`: Per-dataset metrics and aggregates
- `full_results_YYYYMMDD_HHMMSS.json`: Includes full bridging score arrays

## References

- [Pol.is math source code](https://github.com/compdemocracy/polis/tree/edge/math/src/polismath/math)
- Priority formula: `conversation.clj`
- PCA implementation: `pca.clj`
