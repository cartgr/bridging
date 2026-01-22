# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 88% | 0.976 (0.008) | 0.980 (0.012) | 0.971 (0.009) | 0.0030 | 0.0019 | 0.0007 |
| 75% | 0.942 (0.022) | 0.922 (0.049) | 0.935 (0.026) | 0.0058 | 2.3223 | 0.0017 |
| 69% | 0.924 (0.020) | 0.922 (0.036) | 0.918 (0.032) | 0.0071 | 13.8077 | 0.0016 |
| 56% | 0.887 (0.035) | 0.855 (0.054) | 0.756 (0.109) | 0.0093 | 8.3065 | 0.0036 |
| 50% | 0.892 (0.048) | 0.875 (0.061) | 0.692 (0.167) | 0.0105 | 8.8533 | 0.0039 |
| 37% | 0.812 (0.050) | 0.823 (0.059) | 0.176 (0.224) | 0.0123 | 12.2349 | 0.0044 |
| 25% | 0.792 (0.078) | 0.796 (0.087) | -0.268 (0.166) | 0.0157 | 3.0609 | 0.0767 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
