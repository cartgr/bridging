# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.969 (0.012) | 0.971 (0.014) | 0.965 (0.016) | 0.0030 | 0.0021 | 0.0008 |
| 75% | 0.946 (0.018) | 0.953 (0.023) | 0.942 (0.023) | 0.0058 | 0.0045 | 0.0018 |
| 69% | 0.927 (0.032) | 0.926 (0.039) | 0.905 (0.046) | 0.0073 | 0.0074 | 0.0019 |
| 56% | 0.906 (0.041) | 0.911 (0.049) | 0.777 (0.095) | 0.0093 | 0.0093 | 0.0033 |
| 50% | 0.871 (0.051) | 0.865 (0.056) | 0.665 (0.149) | 0.0104 | 0.0109 | 0.0038 |
| 37% | 0.823 (0.067) | 0.820 (0.068) | 0.218 (0.233) | 0.0122 | 0.0155 | 0.0046 |
| 25% | 0.813 (0.096) | 0.814 (0.092) | -0.216 (0.193) | 0.0159 | 0.0165 | 0.0723 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
