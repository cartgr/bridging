# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.988 (0.004) | 0.957 (0.092) | 0.978 (0.011) | 0.0040 | 0.2280 | 0.0025 |
| 75% | 0.980 (0.011) | 0.962 (0.072) | 0.945 (0.032) | 0.0069 | 8.0010 | 0.0074 |
| 69% | 0.956 (0.023) | 0.913 (0.103) | 0.816 (0.097) | 0.0081 | 24926.9554 | 0.0282 |
| 56% | 0.946 (0.025) | 0.925 (0.036) | 0.616 (0.066) | 0.0097 | 36.2572 | 0.0442 |
| 50% | 0.920 (0.033) | 0.894 (0.060) | 0.578 (0.161) | 0.0097 | 50.7472 | 0.0370 |
| 37% | 0.912 (0.036) | 0.904 (0.039) | 0.312 (0.210) | 0.0125 | 17.7114 | 0.0437 |
| 25% | 0.879 (0.063) | 0.872 (0.060) | -0.391 (0.188) | 0.0264 | 16.7890 | 0.0758 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
