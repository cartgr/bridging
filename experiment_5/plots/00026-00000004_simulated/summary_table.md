# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.991 (0.004) | 0.990 (0.008) | 0.932 (0.076) | 0.0054 | 0.0018 | 0.0174 |
| 75% | 0.988 (0.006) | 0.979 (0.029) | 0.806 (0.048) | 0.0085 | 0.0795 | 0.0439 |
| 69% | 0.985 (0.012) | 0.980 (0.013) | 0.778 (0.046) | 0.0098 | 0.0060 | 0.0440 |
| 56% | 0.956 (0.024) | 0.953 (0.025) | 0.639 (0.124) | 0.0114 | 0.0099 | 0.0439 |
| 50% | 0.955 (0.031) | 0.950 (0.029) | 0.633 (0.140) | 0.0113 | 3.1721 | 0.0416 |
| 37% | 0.924 (0.030) | 0.919 (0.032) | 0.405 (0.317) | 0.0106 | 0.8913 | 0.0389 |
| 25% | 0.894 (0.044) | 0.888 (0.044) | -0.387 (0.195) | 0.0247 | 0.0263 | 0.0772 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
