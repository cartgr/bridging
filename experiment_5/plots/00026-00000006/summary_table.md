# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.986 (0.008) | 0.960 (0.022) | 0.0073 | 0.0047 | 3.5 |
| 80% | 0.980 (0.011) | 0.934 (0.034) | 0.0133 | 0.0134 | 3.1 |
| 70% | 0.977 (0.014) | 0.929 (0.028) | 0.0188 | 0.0160 | 2.8 |
| 60% | 0.966 (0.017) | 0.882 (0.054) | 0.0235 | 0.0126 | 2.8 |
| 50% | 0.959 (0.020) | 0.828 (0.079) | 0.0278 | 0.0084 | 3.0 |
| 40% | 0.951 (0.024) | 0.782 (0.091) | 0.0308 | 0.0069 | 3.2 |
| 30% | 0.932 (0.033) | 0.686 (0.098) | 0.0330 | 0.0043 | 4.0 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
