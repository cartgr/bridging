# Experiment 2: Bridging Score Estimation Under Pol.is Sampling

> **TODO:**
> - [ ] Do a full run of the experiment
> - [ ] Change exact inclusion probability computation to Monte Carlo

Tests whether IPW can accurately estimate bridging scores from partially observed Pol.is data.

## The Problem

Pol.is uses adaptive routing that creates **informative missingness**: comments with higher agreement and more extreme PCA positions are shown more often. Naive estimators are biased.

## Usage

```bash
python experiment_2/run_experiment.py
```

## Output

Results in `results/*.json` with metrics: Spearman, Kendall, RMSE, top-k precision/recall.

## Method

1. Simulate Pol.is routing on complete (ground truth) data
2. Track exact inclusion probabilities for each voter-comment pair
3. Estimate bridging scores using IPW correction
4. Compare to ground truth

### Bridging Score

```
b(c) = (4/n²) × Σᵢ＜ⱼ dᵢⱼ   where i,j ∈ approvers of c
```

### Pol.is Priority Formula

Implemented in [`priority.py`](priority.py), based on [conversation.clj](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj):

```
priority = [importance × vote_factor]²
importance = (1 - pass_rate) × (extremeness + 1) × agree_rate
vote_factor = 1 + 8 × 2^(-votes/5)
```

With Laplace smoothing:
- `agree_rate = (A + 1) / (S + 2)`
- `pass_rate = (P + 1) / (S + 2)`

### PCA Extremeness

Implemented in [`priority.py:compute_pca_extremeness()`](priority.py), based on [pca.clj](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/pca.clj):

1. **Participant filtering:** Include voters with ≥ min(7, n_comments) votes; if < 15 qualify, greedily add top contributors
2. **Imputation:** Missing values → per-comment mean
3. **PCA:** Power iteration (matching pca.clj)
4. **Extremeness:** L2 norm of each comment's loading on first 2 PCs

### Inclusion Probability Computation

Implemented in [`simulation.py`](simulation.py). For PPS sampling without replacement:

```
π_c(S, k) = p_c(S) + Σⱼ p_j(S) × π_c(S\{j}, k-1)
```

Exact computation when feasible (≤500k states), Monte Carlo fallback otherwise.

### IPW Estimator

Implemented in [`estimation.py`](estimation.py):

```
d̂ᵢⱼ = Σᶜ 𝟙[disagree] / (πᵢᶜ × πⱼᶜ)
```

## Data

- Ground truth: `data/completed/00069-*.npz`
- Vote distributions: `data/processed/preflib/00069-*.npz`

## Tests

```bash
pytest experiment_2/tests/ -v
pytest experiment_2/tests/ -v -m "not slow"  # Skip slow tests
```
