# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.994 (0.004) | 0.961 (0.039) | 0.0081 | 0.0077 | 2.1 |
| 80% | 0.990 (0.007) | 0.953 (0.040) | 0.0149 | 0.0140 | 2.3 |
| 70% | 0.984 (0.010) | 0.931 (0.045) | 0.0211 | 0.0254 | 2.6 |
| 60% | 0.981 (0.010) | 0.901 (0.054) | 0.0263 | 0.0305 | 2.7 |
| 50% | 0.969 (0.013) | 0.874 (0.064) | 0.0311 | 0.0354 | 2.9 |
| 40% | 0.960 (0.021) | 0.810 (0.084) | 0.0346 | 0.0405 | 3.3 |
| 30% | 0.936 (0.028) | 0.792 (0.097) | 0.0370 | 0.0448 | 4.5 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
