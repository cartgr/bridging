# Robustness Comparison: Bridging Score vs Polis Consensus

## Results by Observation Rate

| Obs Rate | Bridging Spearman | Polis Spearman | Bridging RMSE | Polis RMSE | Polis k |
|----------|-------------------|----------------|---------------|------------|---------|
| 90% | 0.987 (0.008) | 0.968 (0.023) | 0.0072 | 0.0045 | 2.9 |
| 80% | 0.982 (0.010) | 0.942 (0.042) | 0.0135 | 0.0111 | 2.8 |
| 70% | 0.978 (0.012) | 0.920 (0.047) | 0.0189 | 0.0153 | 2.7 |
| 60% | 0.970 (0.015) | 0.886 (0.063) | 0.0237 | 0.0188 | 2.8 |
| 50% | 0.962 (0.017) | 0.860 (0.068) | 0.0279 | 0.0199 | 2.9 |
| 40% | 0.953 (0.022) | 0.805 (0.083) | 0.0310 | 0.0221 | 3.0 |
| 30% | 0.930 (0.032) | 0.728 (0.101) | 0.0332 | 0.0257 | 3.5 |

## Legend
- **Obs Rate**: Fraction of votes observed (1 - mask_rate)
- **Spearman**: Spearman rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
- **Polis k**: Average number of clusters selected by Polis
