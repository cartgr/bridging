# Robustness Comparison: PD Bridging vs p-norm vs Polis Consensus

## Results by Observation Rate

| Obs Rate | PD Spearman | p-norm Spearman | Polis Spearman | PD RMSE | p-norm RMSE | Polis RMSE | Polis k |
|----------|-------------|-----------------|----------------|---------|-------------|------------|---------|
| 90% | 0.992 (0.006) | 0.993 (0.003) | 0.970 (0.034) | 0.0081 | 0.0061 | 0.0090 | 2.2 |
| 80% | 0.989 (0.007) | 0.992 (0.004) | 0.941 (0.051) | 0.0150 | 0.0098 | 0.0195 | 2.5 |
| 70% | 0.985 (0.010) | 0.985 (0.008) | 0.928 (0.053) | 0.0214 | 0.0139 | 0.0196 | 2.5 |
| 60% | 0.979 (0.010) | 0.984 (0.008) | 0.908 (0.056) | 0.0269 | 0.0168 | 0.0268 | 2.6 |
| 50% | 0.976 (0.011) | 0.977 (0.009) | 0.885 (0.068) | 0.0313 | 0.0227 | 0.0300 | 2.7 |
| 40% | 0.969 (0.014) | 0.965 (0.014) | 0.864 (0.057) | 0.0349 | 0.0306 | 0.0311 | 2.7 |
| 30% | 0.954 (0.021) | 0.951 (0.026) | 0.804 (0.084) | 0.0374 | 0.0422 | 0.0380 | 3.5 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
