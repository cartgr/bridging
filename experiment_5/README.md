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

### What's Real vs Synthetic

| Component | Source | Notes |
|-----------|--------|-------|
| **Priority formula** | Real Polis | Exact formula from [conversation.clj](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj) |
| **PCA extremeness** | Real Polis | Power iteration PCA matching [pca.clj](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/pca.clj) |
| **Sampling mechanism** | Real Polis | PPS without replacement |
| **Session lengths** | Synthetic | Uniform distribution around target observation rate |

### Polis Priority Formula

From [`experiment_2/priority.py`](../experiment_2/priority.py), matching the production Polis source:

```
importance = (1 - p) × (E + 1) × a
priority = [importance × (1 + 8 × 2^(-S/5))]²
```

Where (with Laplace smoothing):
- `a = (A + 1) / (S + 2)` — smoothed agree rate
- `p = (P + 1) / (S + 2)` — smoothed pass rate (P=0 in binary setting)
- `E` — PCA extremeness (L2 norm of comment's loading on first 2 PCs)
- `S` — total votes on this comment

The `(1 + 8 × 2^(-S/5))` term is an exploration bonus that favors comments with fewer votes.

### Simulation Loop

For each voter in sequence:

1. **Compute PCA extremeness** on the current observed matrix (updates as data accumulates)
2. **Compute priorities** for all comments using the formula above
3. **Sample k comments** via PPS (probability proportional to priority) without replacement
4. **Estimate inclusion probabilities** via Monte Carlo (500 samples) for IPW correction
5. **Record votes** and update observed matrix for next voter

### Session Length Distribution

Session lengths (how many comments each voter sees) are **synthetic**, not from real Polis data:

```python
# For target observation rate (e.g., 50%):
target_k = int(n_items * 0.5)  # e.g., 8 for 16 items
distribution = [target_k - 2, ..., target_k + 2]  # e.g., [6, 7, 8, 9, 10]
```

Each voter's session length is drawn uniformly from this range.

### Sources of Variance Between Trials

1. **Stochastic PPS sampling**: Even with identical priorities, different comments are sampled each draw
2. **Random session lengths**: k_votes varies per voter within the target range
3. **Sequential dependency**: Early voters' responses change PCA extremeness for later voters, creating cascading effects

### Differences from Production Polis

| Aspect | Production Polis | Our Simulation |
|--------|-----------------|----------------|
| Session lengths | User-controlled (vote until bored) | Synthetic distribution |
| Priority updates | Real-time streaming | After each simulated voter |
| Pass votes | Supported (P > 0) | Binary only (P = 0) |
| k-smoothing | Buffer of 4 before switching k | Single k selection |

## Data

French Election (00026): `data/processed/preflib/00026-*.npz`

## Tests

```bash
pytest experiment_5/tests/ -v
```
