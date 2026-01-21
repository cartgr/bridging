# Bridging Score Experiments

**Identifying consensus in polarized discussions.**

Bridging comments are approved by voters who otherwise disagree. This repository explores how to measure and estimate bridging scores from voting data, with a focus on [Pol.is](https://pol.is)-style platforms.

## Experiments

| Experiment | Question | Run |
|------------|----------|-----|
| [1. Approval vs Diversity](experiment_1/) | How do bridging metrics balance approval vs approver diversity? | `python experiment_1/analyze.py` |
| [2. IPW Estimation](experiment_2/) | Can we estimate bridging under informative missingness? | `python experiment_2/run_experiment.py` |
| [3. PCA Spectrum](experiment_3/) | Do bridging candidates draw support from across the spectrum? | `python experiment_3/visualize.py` |
| [4. Synthetic Elections](experiment_4/) | How do group sizes and approval rates affect bridging? | `python experiment_4/analyze.py` |
| [5. Robustness](experiment_5/) | How robust is bridging vs Polis consensus? | `python experiment_5/run_experiment.py` |
| [6. Real Pol.is Rankings](experiment_6/) | Do methods rank real Pol.is comments differently? | `python experiment_6/analyze.py` |

## Data

| Dataset | Source | Files |
|---------|--------|-------|
| French Election (00026) | [PrefLib](https://preflib.simonrey.fr/) | 6 complete matrices |
| Pol.is (00069) | Deliberation data | 20 matrix-completed |

Format: `.npz` files with shape `(n_items, n_voters)`. Values: `1.0` (approve), `0.0` (disapprove), `NaN` (missing).

## Key Concepts

**Bridging Score:**
```
b(c) = (4/n²) × Σ dᵢⱼ   for all approver pairs i,j
```
High score = approved by voters who otherwise disagree.

**IPW Correction:** Weights observations by inverse inclusion probability to correct for biased sampling.

## Requirements

```bash
pip install numpy scipy matplotlib tqdm scikit-learn
```

## References

- [Pol.is source code](https://github.com/compdemocracy/polis)
- [Group-Informed Consensus](https://compdemocracy.org/group-informed-consensus/)
