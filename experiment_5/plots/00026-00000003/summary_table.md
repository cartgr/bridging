# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.991 (0.007) | 0.989 (0.008) | 0.0071 | 0.0035 | 2.0 |
| 80% | 0.988 (0.006) | 0.979 (0.022) | 0.0135 | 0.0058 | 2.0 |
| 70% | 0.985 (0.007) | 0.958 (0.043) | 0.0185 | 0.0128 | 2.2 |
| 60% | 0.982 (0.008) | 0.913 (0.057) | 0.0233 | 0.0252 | 2.6 |
| 50% | 0.974 (0.011) | 0.887 (0.056) | 0.0276 | 0.0307 | 2.8 |
| 40% | 0.972 (0.012) | 0.858 (0.054) | 0.0307 | 0.0325 | 3.0 |
| 30% | 0.957 (0.020) | 0.795 (0.074) | 0.0328 | 0.0349 | 3.7 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
