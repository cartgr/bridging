# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.990 (0.006) | 0.989 (0.006) | 0.983 (0.010) | 0.0055 | 0.0018 | 0.0000 |
| 75% | 0.979 (0.010) | 0.978 (0.012) | 0.959 (0.018) | 0.0092 | 0.1308 | 0.0002 |
| 69% | 0.975 (0.009) | 0.969 (0.021) | 0.947 (0.018) | 0.0103 | 0.0054 | 0.0004 |
| 56% | 0.949 (0.030) | 0.940 (0.042) | 0.865 (0.082) | 0.0110 | 4.2175 | 0.0013 |
| 50% | 0.951 (0.023) | 0.943 (0.028) | 0.829 (0.068) | 0.0106 | 1.9966 | 0.0030 |
| 37% | 0.917 (0.042) | 0.911 (0.046) | 0.459 (0.287) | 0.0126 | 0.6111 | 0.0151 |
| 25% | 0.901 (0.044) | 0.895 (0.048) | -0.356 (0.216) | 0.0246 | 0.0258 | 0.0869 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
