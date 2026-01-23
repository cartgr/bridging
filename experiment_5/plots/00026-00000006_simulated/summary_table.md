# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.993 (0.006) | 0.989 (0.009) | 0.979 (0.013) | 0.0038 | 0.0023 | 0.0005 |
| 75% | 0.964 (0.019) | 0.957 (0.026) | 0.943 (0.026) | 0.0060 | 0.0059 | 0.0027 |
| 69% | 0.955 (0.029) | 0.949 (0.031) | 0.925 (0.040) | 0.0071 | 0.0118 | 0.0087 |
| 56% | 0.921 (0.043) | 0.915 (0.044) | 0.878 (0.066) | 0.0091 | 0.0101 | 0.0111 |
| 50% | 0.910 (0.056) | 0.902 (0.054) | 0.799 (0.120) | 0.0098 | 0.0112 | 0.0098 |
| 37% | 0.894 (0.048) | 0.889 (0.049) | 0.337 (0.260) | 0.0108 | 0.6193 | 0.0070 |
| 25% | 0.858 (0.049) | 0.853 (0.052) | -0.327 (0.255) | 0.0227 | 0.0228 | 0.0904 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
