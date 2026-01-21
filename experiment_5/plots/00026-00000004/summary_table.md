# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.992 (0.006) | 0.970 (0.034) | 0.0081 | 0.0090 | 2.2 |
| 80% | 0.989 (0.007) | 0.939 (0.051) | 0.0150 | 0.0202 | 2.6 |
| 70% | 0.985 (0.010) | 0.928 (0.050) | 0.0214 | 0.0191 | 2.5 |
| 60% | 0.979 (0.010) | 0.911 (0.054) | 0.0269 | 0.0270 | 2.6 |
| 50% | 0.976 (0.011) | 0.868 (0.074) | 0.0313 | 0.0326 | 2.8 |
| 40% | 0.969 (0.014) | 0.834 (0.066) | 0.0349 | 0.0397 | 3.3 |
| 30% | 0.954 (0.021) | 0.766 (0.093) | 0.0374 | 0.0443 | 4.5 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
