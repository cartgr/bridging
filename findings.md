## TLDR
- Qualitatively the dynamic groups with p = -10 seems to pick the "bridging" candidates most consistently. See [Experiment 3](#experiment-3).
- All bridging metrics are highly correlated with approval. See [Experiment 1](#experiment-1).
- Pol.is rankings are more unstable than PD and dynamic groups when subject to sparse matrices. At 30% sparsity Pol.is is very unstable. Empirically the Pol.is datasets have observation rates between 2.5% and 57.6% (mean 18.5%) so there is reason to be concerned about the ranking obtained from Pol.is' group informed consensus. PD is much more stable. See [Experiment 5](#experiment-5).

## Experiment 1
- All metrics are highly correlated with approval:

![all_metrics_vs_approval](experiment_1/plots/correlations/all_metrics_vs_approval.png)

- Correlation with approver diversity (average pairwise hamming among approvers) is low. Harmonic-PD is the most correlated at ρ = 0.277:

![all_metrics_vs_diversity](experiment_1/plots/correlations/all_metrics_vs_diversity.png)

## Experiment 2
- Naively computing PD using whatever entries in the matrix exist performs better than IPW:

![multi_metric_comparison](experiment_2/plots/multi_metric_comparison.png)

## Experiment 3
- It seems as though the dynamic groups bridging metric with p = -10 and the Pol.is group informed consensus method are the only two methods that pick qualitatively bridging candidates (when looking at voters through the lens of the first component given by PCA).
- In the 2002 French election data from the second district (00026-00000002) dynamic groups and Polis select Chevenement which looks like the "bridging option." See [the plots](https://github.com/cartgr/bridging/tree/main/experiment_3#example-french-election-2002-00026-00000002).
- In the 2007 French election data from the sixth district (00071-00000006) dynamic groups is the only method to select Bayrou who is the centrist candidate and also qualitatively looks to be the most bridging:

![pnorm_min_00071](experiment_3/plots/pnorm_min/00071-00000006.png)

## Experiment 5
- The 2002 French election data has no missing entries. If we randomly mask entries and get the bridging ranking we can see that PD (not using IPW) is more stable than the dynamic groups which in turn is more stable than Pol.is group aware consensus both in terms of the overall correlation between ranking of different trials with different masked entries and in terms of consistently identifying the same winner:

![combined_ranking_stability](experiment_5/plots/combined_ranking_stability.png)

- If we mask according the Pol.is routing algorithm instead of uniformly at random, we observe that PD is still more stable than Pol.is' group informed consensus and that naively estimating PD without IPW is better than using IPW. Note, the green in this plot represents PD estimated with IPW instead of dynamic groups like the above plot:

![combined_simulated_ranking_stability](experiment_5/plots/combined_simulated_ranking_stability.png)

## Experiment 6
- Overall, I am suspicious of the Pol.is data that has been completed with collaborative filtering and I should probably rerun this on the sparse original Pol.is data:

![00069_ridgeline](experiment_6/plots/00069-00000001_ridgeline.png)

## Experiment 7
- Changing the random seed appears to occasionally change the Pol.is group aware consensus ranking:

![polis_seed_stability](experiment_7/plots/polis_seed_stability.png)
