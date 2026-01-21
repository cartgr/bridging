# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.991 (0.007) | 0.989 (0.008) | 0.0071 | 0.0035 | 2.0 |
| 80% | 0.988 (0.006) | 0.979 (0.022) | 0.0135 | 0.0058 | 2.0 |
| 70% | 0.985 (0.007) | 0.956 (0.044) | 0.0185 | 0.0133 | 2.3 |
| 60% | 0.982 (0.008) | 0.916 (0.056) | 0.0233 | 0.0244 | 2.6 |
| 50% | 0.974 (0.011) | 0.864 (0.064) | 0.0276 | 0.0323 | 2.8 |
| 40% | 0.972 (0.012) | 0.854 (0.051) | 0.0307 | 0.0368 | 3.3 |
| 30% | 0.957 (0.020) | 0.808 (0.059) | 0.0328 | 0.0394 | 4.4 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
