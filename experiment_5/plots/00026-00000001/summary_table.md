# Robustness Comparison: PD Bridging vs p-norm vs Polis Consensus

## Results by Observation Rate

| Obs Rate | PD Spearman | p-norm Spearman | Polis Spearman | PD RMSE | p-norm RMSE | Polis RMSE | Polis k |
|----------|-------------|-----------------|----------------|---------|-------------|------------|---------|
| 90% | 0.968 (0.016) | 0.987 (0.007) | 0.973 (0.017) | 0.0049 | 0.0072 | 0.0010 | 3.0 |
| 80% | 0.957 (0.023) | 0.978 (0.015) | 0.946 (0.031) | 0.0095 | 0.0114 | 0.0016 | 3.0 |
| 70% | 0.950 (0.025) | 0.963 (0.019) | 0.916 (0.045) | 0.0131 | 0.0145 | 0.0022 | 3.2 |
| 60% | 0.933 (0.037) | 0.935 (0.039) | 0.859 (0.059) | 0.0163 | 0.0202 | 0.0027 | 3.3 |
| 50% | 0.920 (0.036) | 0.922 (0.038) | 0.800 (0.081) | 0.0192 | 0.0268 | 0.0035 | 3.2 |
| 40% | 0.900 (0.048) | 0.892 (0.055) | 0.689 (0.115) | 0.0214 | 0.0377 | 0.0060 | 3.2 |
| 30% | 0.842 (0.070) | 0.837 (0.073) | 0.611 (0.156) | 0.0229 | 0.0549 | 0.0107 | 3.8 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
