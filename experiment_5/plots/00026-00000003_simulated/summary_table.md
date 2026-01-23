# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.989 (0.004) | 0.987 (0.008) | 0.990 (0.004) | 0.0043 | 0.0017 | 0.0021 |
| 75% | 0.978 (0.010) | 0.972 (0.017) | 0.970 (0.012) | 0.0076 | 1.0777 | 0.0043 |
| 69% | 0.967 (0.018) | 0.964 (0.019) | 0.969 (0.018) | 0.0089 | 0.0052 | 0.0051 |
| 56% | 0.956 (0.020) | 0.948 (0.021) | 0.905 (0.073) | 0.0102 | 0.0086 | 0.0144 |
| 50% | 0.930 (0.042) | 0.918 (0.045) | 0.829 (0.122) | 0.0108 | 0.1107 | 0.0201 |
| 37% | 0.920 (0.039) | 0.915 (0.041) | 0.431 (0.249) | 0.0110 | 0.9607 | 0.0349 |
| 25% | 0.878 (0.060) | 0.874 (0.059) | -0.364 (0.220) | 0.0207 | 1.6567 | 0.0784 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
