# Robustness Comparison: PD Bridging vs p-norm vs Polis Consensus

## Results by Observation Rate

| Obs Rate | PD Spearman | p-norm Spearman | Polis Spearman | PD RMSE | p-norm RMSE | Polis RMSE | Polis k |
|----------|-------------|-----------------|----------------|---------|-------------|------------|---------|
| 90% | 0.994 (0.004) | 0.992 (0.005) | 0.961 (0.039) | 0.0081 | 0.0074 | 0.0077 | 2.1 |
| 80% | 0.990 (0.007) | 0.989 (0.009) | 0.953 (0.040) | 0.0149 | 0.0105 | 0.0139 | 2.3 |
| 70% | 0.984 (0.010) | 0.976 (0.014) | 0.935 (0.040) | 0.0211 | 0.0161 | 0.0253 | 2.6 |
| 60% | 0.981 (0.010) | 0.972 (0.015) | 0.898 (0.057) | 0.0263 | 0.0190 | 0.0286 | 2.6 |
| 50% | 0.969 (0.013) | 0.961 (0.020) | 0.882 (0.063) | 0.0311 | 0.0264 | 0.0336 | 2.9 |
| 40% | 0.960 (0.021) | 0.943 (0.028) | 0.843 (0.083) | 0.0346 | 0.0353 | 0.0357 | 2.9 |
| 30% | 0.936 (0.028) | 0.924 (0.031) | 0.760 (0.096) | 0.0370 | 0.0506 | 0.0372 | 3.4 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
