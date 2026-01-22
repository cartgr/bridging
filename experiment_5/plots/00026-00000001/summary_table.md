# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.968 (0.016) | 0.973 (0.017) | 0.0049 | 0.0010 | 3.0 |
| 80% | 0.957 (0.023) | 0.946 (0.031) | 0.0095 | 0.0016 | 3.0 |
| 70% | 0.950 (0.025) | 0.913 (0.046) | 0.0131 | 0.0023 | 3.2 |
| 60% | 0.933 (0.037) | 0.859 (0.058) | 0.0163 | 0.0027 | 3.3 |
| 50% | 0.920 (0.036) | 0.798 (0.082) | 0.0192 | 0.0035 | 3.2 |
| 40% | 0.900 (0.048) | 0.689 (0.116) | 0.0214 | 0.0060 | 3.2 |
| 30% | 0.842 (0.070) | 0.610 (0.159) | 0.0229 | 0.0091 | 3.7 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
