# Robustness Comparison: Simulated Polis Routing

## Results by Observation Rate

| Obs Rate | Naive Spearman | IPW Spearman | Polis Spearman | Naive RMSE | IPW RMSE | Polis RMSE |
|----------|----------------|--------------|----------------|------------|----------|------------|
| 87% | 0.982 (0.006) | 0.968 (0.052) | 0.974 (0.010) | 0.0035 | 0.1150 | 0.0016 |
| 75% | 0.961 (0.016) | 0.942 (0.060) | 0.940 (0.029) | 0.0064 | 5.1617 | 0.0045 |
| 69% | 0.940 (0.021) | 0.917 (0.070) | 0.867 (0.064) | 0.0076 | 12470.3815 | 0.0149 |
| 56% | 0.917 (0.030) | 0.890 (0.045) | 0.686 (0.087) | 0.0095 | 22.2819 | 0.0239 |
| 50% | 0.906 (0.040) | 0.885 (0.060) | 0.635 (0.164) | 0.0101 | 29.8003 | 0.0205 |
| 37% | 0.862 (0.043) | 0.863 (0.049) | 0.244 (0.217) | 0.0124 | 14.9731 | 0.0240 |
| 25% | 0.836 (0.070) | 0.834 (0.074) | -0.329 (0.177) | 0.0211 | 9.9249 | 0.0762 |

## Legend
- **Obs Rate**: Actual observation rate achieved under simulated routing
- **Naive**: Bridging score without IPW correction
- **IPW**: Bridging score with Inverse Probability Weighting
- **Polis**: Polis Group-Informed Consensus
- **Spearman**: Rank correlation with ground truth (mean, std in parentheses)
- **RMSE**: Root mean squared error from ground truth
