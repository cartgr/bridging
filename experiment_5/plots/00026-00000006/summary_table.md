# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.986 (0.008) | 0.963 (0.021) | 0.0073 | 0.0048 | 3.5 |
| 80% | 0.980 (0.011) | 0.938 (0.033) | 0.0133 | 0.0134 | 3.1 |
| 70% | 0.977 (0.014) | 0.931 (0.027) | 0.0188 | 0.0153 | 2.8 |
| 60% | 0.966 (0.017) | 0.890 (0.060) | 0.0235 | 0.0150 | 2.8 |
| 50% | 0.959 (0.020) | 0.857 (0.062) | 0.0278 | 0.0117 | 2.9 |
| 40% | 0.951 (0.024) | 0.808 (0.091) | 0.0308 | 0.0122 | 2.9 |
| 30% | 0.932 (0.033) | 0.718 (0.113) | 0.0330 | 0.0143 | 3.5 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
