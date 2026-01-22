# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.992 (0.006) | 0.955 (0.021) | 0.0077 | 0.0012 | 4.4 |
| 80% | 0.990 (0.006) | 0.900 (0.076) | 0.0146 | 0.0121 | 3.6 |
| 70% | 0.987 (0.006) | 0.856 (0.075) | 0.0205 | 0.0162 | 3.0 |
| 60% | 0.981 (0.010) | 0.852 (0.091) | 0.0259 | 0.0144 | 2.9 |
| 50% | 0.975 (0.014) | 0.852 (0.076) | 0.0302 | 0.0096 | 3.1 |
| 40% | 0.967 (0.014) | 0.764 (0.097) | 0.0337 | 0.0146 | 3.0 |
| 30% | 0.957 (0.018) | 0.677 (0.083) | 0.0361 | 0.0177 | 3.3 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
