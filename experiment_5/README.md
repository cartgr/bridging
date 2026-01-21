# Experiment 5: Robustness Comparison

This experiment compares the robustness of bridging/consensus metrics under different missingness conditions.

## Methods Compared

1. **Polis Group-Informed Consensus** - The metric used by [Pol.is](https://pol.is)
2. **Pairwise Disagreement Bridging Score (Naive)** - Our metric without correction
3. **Pairwise Disagreement Bridging Score (IPW)** - Our metric with Inverse Probability Weighting

## Two Experimental Conditions

### 1. MCAR Masking (`run_experiment.py`)
- **Missingness**: Missing Completely At Random (uniform random)
- **Methods compared**: Bridging (Naive) vs Polis
- **Purpose**: Baseline comparison under ideal conditions
- **Result**: Bridging wins 39, Polis wins 0, Ties 3

### 2. Simulated Polis Routing (`run_simulation_experiment.py`)
- **Missingness**: Informative (via simulated Polis routing with PCA-based priorities)
- **Methods compared**: Bridging (Naive) vs Bridging (IPW) vs Polis
- **Purpose**: Realistic comparison under actual Polis observation patterns

## Methods Compared

### Polis Group-Informed Consensus

Based on the [Polis documentation](https://compdemocracy.org/group-informed-consensus/) and [source code](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj).

**Pipeline:**
1. **Imputation**: Replace missing votes with per-comment mean
2. **PCA**: Compute voter projections (2 components) using power iteration
3. **K-means clustering**: Cluster voters with silhouette-based k selection
   - Try k from 2 to min(5, 2 + n_voters/12)
   - Select k with highest silhouette score
4. **Group-Informed Consensus Score**:
   ```
   P_g(c) = (A_g + 1) / (S_g + 2)   # Laplace-smoothed approval rate
   consensus(c) = Π_g P_g(c)         # Product over all groups
   ```

The product formula means one dissenting group kills the score - preventing "tyranny of the majority" but potentially sensitive to clustering instability.

### Pairwise Disagreement Bridging Score

Our metric from Experiment 2:
```
b^PD(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij
```

Where d_ij is the pairwise disagreement between voters i and j.

**Two estimation methods:**
- **Naive**: Compute on observed data without correction (treats missing as random)
- **IPW**: Inverse Probability Weighting correction using tracked inclusion probabilities

## Experiment Setup

- **Data**: French election 00026 (6 matrices, ~16 items × ~400 voters each)
- **MCAR Mask rates**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
- **Simulated observation rates**: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
- **Trials**: 50 (MCAR) / configurable (Simulated, default 15)
- **MC samples**: 100 (for inclusion probability estimation in simulation)
- **Random seed**: 42

## Evaluation Metrics

1. **Spearman rank correlation** with ground truth (primary metric)
2. **Kendall tau correlation**
3. **RMSE** between estimated and true scores
4. **Estimate variance** across trials (stability measure)
5. **Top-k precision** (correctly identifying high-bridging items)

## Running the Experiments

```bash
# MCAR experiment (faster)
python experiment_5/run_experiment.py

# Simulation experiment (slower, more realistic)
python experiment_5/run_simulation_experiment.py
```

## Outputs

### Results Directory
- `results/robustness_experiment_TIMESTAMP.json` - MCAR experiment results
- `results/simulation_experiment_TIMESTAMP.json` - Simulation experiment results

### Plots Directory

**MCAR experiment:**
- `plots/{dataset}/spearman_comparison.png` - Spearman correlation vs observation rate
- `plots/{dataset}/rmse_comparison.png` - RMSE vs observation rate
- `plots/{dataset}/variance_comparison.png` - Estimate variance comparison
- `plots/{dataset}/multi_metric.png` - Side-by-side metric comparison
- `plots/combined_*.png` - Combined results across all datasets

**Simulation experiment:**
- `plots/{dataset}_simulated/spearman_comparison.png` - 3-way comparison
- `plots/{dataset}_simulated/multi_metric.png` - Spearman + RMSE
- `plots/combined_simulated_*.png` - Combined simulation results

## Expected Behavior

**MCAR experiment:**
- Both methods should degrade as mask rate increases
- Bridging should be more stable (no clustering step)

**Simulation experiment:**
- IPW should improve over Naive under informative missingness
- Polis may be more sensitive to its own routing patterns
- The benefit of IPW should increase at higher missingness rates

## Module Structure

```
experiment_5/
├── __init__.py                    # Package exports
├── polis.py                       # Polis Group-Informed Consensus
├── masking.py                     # Random masking utilities
├── robustness.py                  # MCAR experiment logic
├── robustness_simulated.py        # Simulation experiment logic
├── evaluate.py                    # Robustness metrics
├── visualize.py                   # Result visualization
├── run_experiment.py              # MCAR runner
├── run_simulation_experiment.py   # Simulation runner
├── README.md                      # This file
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_polis.py
│   └── test_masking.py
├── results/                       # JSON output
└── plots/                         # Visualization output
```

## Dependencies

- numpy
- scipy
- matplotlib
- scikit-learn (for KMeans and silhouette_score)
- tqdm

## Key Implementation Details

### Numerical Stability
- Group-informed consensus uses log-sum-exp to avoid underflow
- Inclusion probabilities clipped at 1e-6 to avoid division issues

### Simulation Details
- Simulates Polis priority-based routing for each voter
- PCA extremeness recomputed for every voter (matching real Polis behavior)
- Uses Monte Carlo estimation for inclusion probabilities (faster than exact computation)
- Tracks per-voter, per-comment inclusion probabilities for IPW

### Edge Cases Handled
- High mask rates: minimum observation constraints per item/voter
- Small clusters: graceful fallback when k-means fails
- Degenerate PCA: return zeros when variance is zero
