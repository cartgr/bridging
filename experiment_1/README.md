# Experiment 1: Approval Rate vs. Approver Diversity

## Purpose

This experiment explores the relationship between how popular an item/comment is and how diverse its supporters are. The hypothesis is that highly approved items might be supported by either:
- A homogeneous group (low diversity) - suggesting echo chamber effects
- A diverse group (high diversity) - suggesting bridging/consensus items

## Metrics

### Approval Rate (x-axis)
The fraction of voters who approved an item:
```
approval_rate[i] = mean(matrix[i, :])
```

### Approver Diversity (y-axis)
The average pairwise Hamming distance between the full voting vectors of all approvers:
1. Find all voters who approved item i
2. For each pair of approvers, compute the normalized Hamming distance between their complete voting profiles
3. Return the mean of all pairwise distances

A high diversity score means approvers tend to vote differently on other items, suggesting they come from different "ideological" positions.

**Optimization:** Uses O(m) closed-form formula instead of O(n²) pairwise computation:
```
Mean Hamming = sum_j[k_j * (n - k_j)] / [n*(n-1)/2 * m]
```
where k_j = count of approvers who voted 1 on item j.

## Data Sources

- **French Election (00026)**: 6 files from PrefLib, complete ground truth data
  - Location: `data/processed/preflib/00026-*.npz`

- **Pol.is Completed (00069)**: 20 files, originally sparse data completed using KNNBaseline
  - Location: `data/completed/00069-*.npz`

## Matrix Format

- Shape: `(n_items, n_voters)` - rows are items/comments, columns are voters
- Values: `1.0` (approved), `0.0` (disapproved)

## How to Run

```bash
python experiment_1/analyze.py
```

## Output

Plots are saved to:
```
experiment_1/plots/hamming_distance/
├── 00026-00000001.png
├── 00026-00000002.png
├── ...
├── 00069-00000001.png
└── ...
```

Each plot shows:
- X-axis: Approver diversity as Hamming distance (0 to 1)
- Y-axis: Approval rate (0 to 1)
- Each point represents one item/comment
- Annotation shows number of valid items (items with <2 approvers are excluded)

## TODO

- [ ] Implement bridging metrics
- [ ] Test other diversity scores
