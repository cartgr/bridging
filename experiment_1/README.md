# Experiment 1: Approval Rate vs Approver Diversity

Explores the relationship between how popular an item is and how diverse its supporters are.

## Usage

```bash
python experiment_1/analyze.py
```

## Output

Scatter plots in `plots/hamming_distance/`:
- X-axis: Approver diversity (mean pairwise Hamming distance)
- Y-axis: Approval rate

High diversity + high approval suggests bridging items.

## Method

**Approver Diversity:** Average pairwise Hamming distance between complete voting profiles of all approvers.

Uses O(m) closed-form computation instead of O(n²) pairwise:
```
Mean Hamming = Σⱼ[kⱼ(n-kⱼ)] / [n(n-1)/2 × m]
```

## Data

- French Election (00026): `data/processed/preflib/00026-*.npz`
- Pol.is (00069): `data/completed/00069-*.npz`
