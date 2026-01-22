# Experiment 1: Approval Rate vs Approver Diversity

Explores the relationship between how popular an item is and how diverse its supporters are.

## Usage

```bash
python experiment_1/analyze.py      # Generate plots
python experiment_1/correlation.py  # Compute correlation statistics
```

## Output

Three types of plots in `plots/`:

| Directory | Description |
|-----------|-------------|
| `plain/` | Basic scatter plot of approval vs diversity |
| `colored/` | Same scatter, colored by PD bridging score (viridis) |
| `comparison/` | Highlights top comment from PD vs Polis methods |

## Metrics

**Approver Diversity:** Average pairwise Hamming distance between complete voting profiles of all approvers.
```
Diversity(c) = Σ_{i<j ∈ N_c} d_ij / [k_c(k_c-1)/2]
```
Uses O(m) closed-form computation instead of O(n²) pairwise.

**PD Bridging Score:** Total pairwise disagreement among approvers, normalized by total voters.
```
b^PD(c) = (4/n²) × Σ_{i<j ∈ N_c} d_ij
```

**Key difference:** Diversity normalizes by approver pairs; bridging normalizes by total voters². This means bridging score scales with both diversity AND approval rate.

## Results

### Approval vs Diversity

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election | 6 | -0.504 ± 0.270 | -0.492 ± 0.321 |
| Pol.is | 20 | 0.395 ± 0.368 | 0.823 ± 0.173 |

The correlation is **negative** for French Election (more approved candidates have less diverse supporters) but **positive** for Pol.is (more approved comments have more diverse supporters).

### PD Bridging vs Approval

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election | 6 | 0.973 ± 0.005 | 0.988 ± 0.007 |
| Pol.is | 20 | 0.896 ± 0.080 | 0.996 ± 0.005 |

PD bridging score is **very strongly correlated** with approval rate in both datasets.

### PD Bridging vs Diversity

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election | 6 | -0.419 ± 0.232 | -0.436 ± 0.317 |
| Pol.is | 20 | 0.611 ± 0.329 | 0.838 ± 0.170 |

PD bridging is negatively correlated with diversity for French Election but positively correlated for Pol.is, mirroring the approval-diversity pattern.

> **Note:** The Pol.is data was artificially completed using a matrix completion algorithm, which may affect these correlations.

## Data

- French Election (00026): `data/processed/preflib/00026-*.npz`
- Pol.is (00069): `data/completed/00069-*.npz`
