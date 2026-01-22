# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.994 (0.004) | 0.962 (0.039) | 0.0081 | 0.0078 | 2.1 |
| 80% | 0.990 (0.007) | 0.953 (0.038) | 0.0149 | 0.0140 | 2.3 |
| 70% | 0.984 (0.010) | 0.936 (0.040) | 0.0211 | 0.0253 | 2.6 |
| 60% | 0.981 (0.010) | 0.899 (0.055) | 0.0263 | 0.0286 | 2.6 |
| 50% | 0.969 (0.013) | 0.883 (0.061) | 0.0311 | 0.0335 | 2.8 |
| 40% | 0.960 (0.021) | 0.843 (0.083) | 0.0346 | 0.0357 | 2.9 |
| 30% | 0.936 (0.028) | 0.754 (0.104) | 0.0370 | 0.0378 | 3.4 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
