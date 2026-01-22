# Experiment 1: Approval Rate vs Approver Diversity

Explores the relationship between how popular an item is and how diverse its supporters are.

## Usage

```bash
python experiment_1/analyze.py
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

## Data

- French Election (00026): `data/processed/preflib/00026-*.npz`
- Pol.is (00069): `data/completed/00069-*.npz`
