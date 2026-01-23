# Experiment 7: Polis Ranking Stability Across Random Seeds

## Question

Does the Polis consensus ranking change when only the random seed changes, given the exact same masked data?

This isolates **algorithmic instability** (k-means initialization) from **data variation**.

## Method

1. Take the 00026 French election dataset (6 matrices, ~16 items × ~400-500 voters)
2. Apply MCAR random masking at various rates (10%-70%) with a **fixed** seed
3. Run Polis consensus algorithm 20 times with **different** random seeds
4. Measure ranking stability across runs

## Metrics

- **Rank Correlation**: Spearman correlation between rankings from different seeds
- **Top-1 Agreement**: How often the "best" item is the same across seeds
- **Top-3 Overlap**: Jaccard overlap of top-3 items
- **k-unique**: Number of different cluster counts selected

## Results

| Obs Rate | Rank Corr (mean±std) | Top-1 Agree | k unique |
|----------|---------------------|-------------|----------|
| 90%      | 0.999 ± 0.002       | 100%        | 1.0      |
| 80%      | 1.000 ± 0.000       | 100%        | 1.0      |
| 70%      | 0.991 ± 0.019       | 92%         | 1.2      |
| 60%      | 1.000 ± 0.000       | 100%        | 1.0      |
| 50%      | 0.998 ± 0.004       | 98%         | 1.2      |
| 40%      | 0.998 ± 0.003       | 99%         | 1.2      |
| 30%      | 0.988 ± 0.014       | 93%         | 1.7      |

### Notable Instability Cases

| Dataset | Obs Rate | Rank Corr | Top-1 Agree | k unique |
|---------|----------|-----------|-------------|----------|
| 00026-00000001 | 70% | 0.948 | **55%** | 2 |
| 00026-00000003 | 30% | 0.976 | 80% | **3** |
| 00026-00000006 | 50% | 0.990 | 90% | 2 |

## Key Findings

1. **Polis is mostly stable** - Average rank correlation ≥ 0.988 across all observation rates

2. **Algorithmic instability exists** - Even with identical masked data, the "best" item changes in 7-8% of cases at 30% observation

3. **k-means cluster instability** - At 30% observation, the number of clusters varies (k_unique=1.7), causing ranking changes

4. **Worst case: 55% top-1 agreement** - In one dataset at 70% observation, the winning item changed 45% of the time just due to random seed

## Comparison with PD Bridging

Our PD Bridging score is **deterministic** given the same masked data - it has no random component. This means:

- PD Bridging: Top-1 Agreement = **100%** (always, by construction)
- Polis: Top-1 Agreement = 93-100% (varies with observation rate)

This is an additional advantage of our metric beyond the robustness results from Experiment 5.

## Implications

The Polis consensus algorithm's reliance on k-means clustering introduces algorithmic instability. While this is usually minor (rank correlation > 0.98), it can be substantial in specific cases (top-1 agreement as low as 55%).

For applications where reproducibility matters, a deterministic metric like PD Bridging is preferable.
