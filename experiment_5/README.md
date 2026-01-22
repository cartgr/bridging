# Experiment 5: Robustness Comparison

> **TODO:**
> - [ ] Do a full run of the simulation experiment (all 6 datasets, 30+ trials)
> - [ ] Add variance analysis (estimate variance across trials for each method)

Compares robustness of bridging scores vs Pol.is Group-Informed Consensus under missing data.

## Usage

```bash
python experiment_5/run_experiment.py              # MCAR masking
python experiment_5/run_simulation_experiment.py   # Simulated Polis routing
```

## Results

| Condition | Missingness | Result |
|-----------|-------------|--------|
| MCAR | Uniform random | Bridging wins all 7 mask rates (ρ 0.93–0.99 vs 0.73–0.97) |
| Simulated | Polis routing | Bridging maintains ρ > 0.83 at 30% obs; Polis drops to ρ ≈ 0.17 |

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

0. **Participant Filtering** ([`filter_participants_by_votes()`](polis.py)): Remove low-activity voters
   - Require min(7, n_comments) votes per participant
   - If < 15 participants remain, greedily add top contributors

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

### Differences from Production Polis

- **No k-smoothing buffer**: Production Polis requires seeing the same optimal k for 4 consecutive updates before switching (for streaming stability). We select k once per analysis since our data is static.

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
