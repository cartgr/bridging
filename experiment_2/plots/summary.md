# Experiment 2: IPW vs Naive Estimation on Pol.is Data

## Aggregate Results

| Metric | Naive | IPW |
|--------|-------|-----|
| Spearman (mean ± std) | 0.711 ± 0.191 | 0.573 ± 0.184 |
| Kendall (mean ± std) | 0.581 ± 0.169 | 0.451 ± 0.157 |
| RMSE (mean ± std) | 0.112 ± 0.056 | 3.517 ± 9.524 |

**Observation Rate**: 18.5% (range: 2.5% - 55.5%)

## Per-Dataset Results

| Dataset | Items | Voters | Obs Rate | Naive Spearman | IPW Spearman | Winner |
|---------|-------|--------|----------|----------------|--------------|--------|
| 00000005 | 2138 | 3142 | 3.5% | 0.932 | 0.670 | Naive |
| 00000009 | 50 | 204 | 46.6% | 0.910 | 0.900 | Naive |
| 00000002 | 896 | 2031 | 8.3% | 0.900 | 0.629 | Naive |
| 00000004 | 522 | 1116 | 8.1% | 0.872 | 0.773 | Naive |
| 00000019 | 297 | 404 | 23.3% | 0.869 | 0.723 | Naive |
| 00000007 | 1452 | 3616 | 3.6% | 0.862 | 0.466 | Naive |
| 00000006 | 613 | 1503 | 6.1% | 0.828 | 0.693 | Naive |
| 00000017 | 148 | 334 | 30.4% | 0.822 | 0.680 | Naive |
| 00000003 | 1039 | 1756 | 5.0% | 0.782 | 0.587 | Naive |
| 00000020 | 197 | 1921 | 11.5% | 0.767 | 0.682 | Naive |
| 00000010 | 174 | 448 | 13.0% | 0.764 | 0.660 | Naive |
| 00000018 | 71 | 234 | 40.1% | 0.749 | 0.504 | Naive |
| 00000012 | 39 | 26 | 55.5% | 0.745 | 0.736 | Naive |
| 00000014 | 165 | 381 | 18.1% | 0.654 | 0.590 | Naive |
| 00000001 | 54 | 339 | 11.1% | 0.626 | 0.542 | Naive |
| 00000015 | 316 | 536 | 14.3% | 0.574 | 0.384 | Naive |
| 00000011 | 298 | 1487 | 2.5% | 0.504 | 0.339 | Naive |
| 00000016 | 80 | 117 | 38.1% | 0.499 | 0.494 | Naive |
| 00000008 | 371 | 126 | 26.7% | 0.315 | 0.330 | IPW |
| 00000013 | 2162 | 6289 | 3.5% | 0.244 | 0.074 | Naive |

## Summary

- **Naive wins**: 19 datasets
- **IPW wins**: 1 datasets
- **Ties**: 0 datasets

**Conclusion**: The naive estimator outperforms IPW on real Pol.is data, likely because:
1. The missing data mechanism is not purely MAR (Missing At Random)
2. IPW can have high variance with estimated propensity scores
3. The observation patterns in Pol.is may be closer to MCAR than expected
