# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.987 (0.008) | 0.968 (0.023) | 0.0072 | 0.0045 | 2.9 |
| 80% | 0.982 (0.010) | 0.942 (0.042) | 0.0135 | 0.0112 | 2.8 |
| 70% | 0.978 (0.012) | 0.919 (0.047) | 0.0189 | 0.0154 | 2.8 |
| 60% | 0.970 (0.015) | 0.885 (0.060) | 0.0237 | 0.0186 | 2.8 |
| 50% | 0.962 (0.017) | 0.845 (0.078) | 0.0279 | 0.0205 | 3.0 |
| 40% | 0.953 (0.022) | 0.794 (0.083) | 0.0310 | 0.0228 | 3.4 |
| 30% | 0.930 (0.032) | 0.743 (0.098) | 0.0332 | 0.0246 | 4.2 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
