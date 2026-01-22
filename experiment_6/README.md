# Experiment 6: Real Pol.is Ranking Comparison

> **TODO:**
> - [ ] Decide how to handle estimation without true inclusion probabilities (naive only, or try heuristic IPW?)

Compares comment rankings between Polis consensus and bridging scores on real (incomplete) Pol.is data.

## Usage

```bash
python experiment_6/analyze.py
```

## Output

- **Ridgeline plots**: `plots/{dataset}_ridgeline.png` - Comments sorted by bridging score with voter approval distribution
- **Scatter plots**: `plots/{dataset}.png` - Score/rank comparison between methods
- **Summary plot**: `plots/summary.png` - Aggregate statistics across datasets
- **Rankings CSV**: `results/{dataset}_rankings.csv` - All comments with both rankings and approval rates
- **Results JSON**: `results/ranking_comparison_*.json`

## Methods Compared

| Method | Description |
|--------|-------------|
| **Polis Consensus** | Group-informed consensus from [`experiment_5/polis.py`](../experiment_5/polis.py) |
| **Bridging (Naive)** | Pairwise disagreement from [`experiment_2/estimation.py`](../experiment_2/estimation.py) |

## Metrics

- **Spearman correlation** between rankings
- **Top-k overlap** (do methods agree on top comments?)
- **Same top-1** (do they pick the same best comment?)

## Results

- 16 datasets analyzed
- Mean Spearman ρ: 0.439 ± 0.159
- Same top-1 comment: 1/16 (6%)

## Data

- Vote matrices: `data/processed/preflib/00069-*.npz`
- Comment text: `data/polis_comments/*.csv` (from [compdemocracy/openData](https://github.com/compdemocracy/openData))
- Dataset mapping: `data/polis_comments/dataset_mapping.json`

## Limitation

**No IPW correction.** On real Pol.is data, we don't have access to the true inclusion probabilities needed for IPW. We use naive estimation for both methods, which may be biased by informative missingness. This is a fair comparison (same conditions for both methods) but neither estimate is necessarily unbiased.
