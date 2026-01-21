# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.992 (0.006) | 0.955 (0.021) | 0.0077 | 0.0012 | 4.4 |
| 80% | 0.990 (0.006) | 0.900 (0.076) | 0.0146 | 0.0121 | 3.6 |
| 70% | 0.987 (0.006) | 0.860 (0.074) | 0.0205 | 0.0160 | 3.1 |
| 60% | 0.981 (0.010) | 0.850 (0.084) | 0.0259 | 0.0140 | 2.9 |
| 50% | 0.975 (0.014) | 0.865 (0.081) | 0.0302 | 0.0099 | 3.0 |
| 40% | 0.967 (0.014) | 0.811 (0.105) | 0.0337 | 0.0050 | 3.6 |
| 30% | 0.957 (0.018) | 0.770 (0.118) | 0.0361 | 0.0032 | 4.4 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
