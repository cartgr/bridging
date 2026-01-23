# Robustness Comparison: PD Bridging vs p-norm vs Polis Consensus

## Results by Observation Rate

| Obs Rate | PD Spearman | p-norm Spearman | Polis Spearman | PD RMSE | p-norm RMSE | Polis RMSE | Polis k |
|----------|-------------|-----------------|----------------|---------|-------------|------------|---------|
| 90% | 0.986 (0.008) | 0.981 (0.009) | 0.960 (0.022) | 0.0073 | 0.0072 | 0.0047 | 3.5 |
| 80% | 0.980 (0.011) | 0.969 (0.016) | 0.934 (0.034) | 0.0133 | 0.0109 | 0.0134 | 3.1 |
| 70% | 0.977 (0.014) | 0.968 (0.017) | 0.929 (0.027) | 0.0188 | 0.0149 | 0.0153 | 2.8 |
| 60% | 0.966 (0.017) | 0.953 (0.019) | 0.887 (0.060) | 0.0235 | 0.0197 | 0.0151 | 2.8 |
| 50% | 0.959 (0.020) | 0.948 (0.020) | 0.852 (0.064) | 0.0278 | 0.0251 | 0.0118 | 2.9 |
| 40% | 0.951 (0.024) | 0.929 (0.027) | 0.810 (0.091) | 0.0308 | 0.0368 | 0.0130 | 2.9 |
| 30% | 0.932 (0.033) | 0.916 (0.033) | 0.721 (0.115) | 0.0330 | 0.0512 | 0.0158 | 3.4 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
