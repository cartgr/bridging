# Experiment 5: Robustness Comparison

> **TODO:**
> - [ ] Do a full run of the simulation experiment (all 6 datasets, 30+ trials)

Compares robustness of bridging scores vs Pol.is Group-Informed Consensus under missing data.

## Usage

```bash
python experiment_5/run_experiment.py              # MCAR masking
python experiment_5/run_simulation_experiment.py   # Simulated Polis routing
```

## Results

| Condition | Missingness | Result |
|-----------|-------------|--------|
| MCAR | Uniform random | Bridging wins 39, Polis wins 0, Ties 3 |
| Simulated | Polis routing | In progress |

## Methods Compared

| Method | Description |
|--------|-------------|
| **Polis Consensus** | Product of per-group Laplace-smoothed approval rates |
| **Bridging (Naive)** | Pairwise disagreement, no correction |
| **Bridging (IPW)** | With inverse probability weighting |

## Output

- Plots: `plots/*.png`
- Results: `results/*.json`

## Polis Group-Informed Consensus

Implemented in [`polis.py`](polis.py), based on [conversation.clj](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj) and the [official documentation](https://compdemocracy.org/group-informed-consensus/).

### Pipeline

1. **Imputation** ([`impute_column_mean()`](polis.py)): Replace missing votes with per-comment mean

2. **PCA** ([`compute_voter_pca_projections()`](polis.py)): Project voters onto 2 components using power iteration

3. **Clustering** ([`cluster_voters_kmeans()`](polis.py)): K-means with silhouette-based k selection
   - Try k ∈ {2, ..., min(5, 2 + n_voters/12)}
   - Select k with highest silhouette score

4. **Consensus Score** ([`compute_group_informed_consensus()`](polis.py)):
   ```
   P_g(c) = (A_g + 1) / (S_g + 2)    # Laplace-smoothed approval
   consensus(c) = Π_g P_g(c)          # Product over all groups
   ```

The product formula means one dissenting group kills the score—preventing "tyranny of the majority" but sensitive to clustering instability.

### Numerical Stability

Uses log-sum-exp to avoid underflow: `log(consensus) = Σ_g log(P_g)`

## Simulated Polis Routing

Implemented in [`robustness_simulated.py`](robustness_simulated.py).

For each voter:
1. Compute PCA extremeness on current observed matrix
2. Compute priorities using Pol.is formula (reuses [`experiment_2/priority.py`](../experiment_2/priority.py))
3. Sample k comments via PPS (probability proportional to priority) without replacement
4. Estimate inclusion probabilities via Monte Carlo (100 samples)

This creates realistic informative missingness matching actual Pol.is behavior.

## Data

French Election (00026): `data/processed/preflib/00026-*.npz`

## Tests

```bash
pytest experiment_5/tests/ -v
```
