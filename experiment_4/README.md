# Experiment 4: Synthetic Elections with Group Structure

Creates synthetic elections with known group structures to understand bridging score behavior analytically.

## Usage

```bash
python experiment_4/analyze.py
```

## Output

- `plots/equal_groups_3d.png` — Bridging surface for equal groups
- `plots/unequal_groups_3d.png` — Bridging surface with constrained total support
- Interactive HTML versions for 3D exploration

## Key Result

For two equal-sized groups with between-group disagreement d ≈ 1:

```
E[bridging] ≈ a₁ × a₂
```

Maximum bridging occurs when both groups fully approve.

## Method

Groups are defined by approval of disjoint "base" comments:
- Group 1: approves even-indexed comments
- Group 2: approves odd-indexed comments

This ensures d ≈ 1 between groups and d ≈ 0 within groups.

### Scenarios

| Scenario | Setup | Varies |
|----------|-------|--------|
| Equal groups | w = 0.5 | a₁, a₂ ∈ [0,1] |
| Unequal groups | Fixed total support = 0.5 | w, a₁ (a₂ derived) |
