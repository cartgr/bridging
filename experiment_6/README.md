# Experiment 6: Real Pol.is Ranking Comparison

> **TODO:**
> - [ ] Decide how to handle estimation without true inclusion probabilities (naive only, or try heuristic IPW?)

Compares comment rankings between Polis consensus and bridging scores on real (incomplete) Pol.is data.

## Usage

```bash
python experiment_6/analyze.py
```

## Output

- Per-dataset plots: `plots/{dataset}.png`
- Summary plot: `plots/summary.png`
- Results: `results/ranking_comparison_*.json`

## Methods Compared

| Method | Description |
|--------|-------------|
| **Polis Consensus** | Group-informed consensus from [`experiment_5/polis.py`](../experiment_5/polis.py) |
| **Bridging (Naive)** | Pairwise disagreement from [`experiment_2/estimation.py`](../experiment_2/estimation.py) |

## Metrics

- **Spearman correlation** between rankings
- **Top-k overlap** (do methods agree on top comments?)
- **Same top-1** (do they pick the same best comment?)

## Data

Incomplete Pol.is datasets (00069): `data/processed/preflib/00069-*.npz`

## Limitation

**No IPW correction.** On real Pol.is data, we don't have access to the true inclusion probabilities needed for IPW. We use naive estimation for both methods, which may be biased by informative missingness. This is a fair comparison (same conditions for both methods) but neither estimate is necessarily unbiased.

Estimating inclusion probabilities from observation rates (π_c ≈ shown_count / total_voters) is possible but makes strong assumptions about stationarity that may not hold.
