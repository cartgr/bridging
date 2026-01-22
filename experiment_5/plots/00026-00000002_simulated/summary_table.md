# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.987 (0.006) | 0.989 (0.012) | 0.977 (0.014) | 0.0039 | 0.0025 | 0.0027 |
| 75% | 0.976 (0.015) | 0.977 (0.016) | 0.948 (0.020) | 0.0068 | 0.0046 | 0.0047 |
| 69% | 0.968 (0.021) | 0.969 (0.020) | 0.819 (0.088) | 0.0082 | 0.0062 | 0.0311 |
| 56% | 0.939 (0.040) | 0.936 (0.041) | 0.636 (0.080) | 0.0097 | 0.0108 | 0.0442 |
| 50% | 0.940 (0.022) | 0.937 (0.025) | 0.616 (0.161) | 0.0100 | 7.8854 | 0.0360 |
| 37% | 0.905 (0.047) | 0.905 (0.048) | 0.392 (0.324) | 0.0123 | 0.0171 | 0.0378 |
| 25% | 0.899 (0.048) | 0.903 (0.046) | -0.397 (0.230) | 0.0262 | 0.4896 | 0.0793 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
