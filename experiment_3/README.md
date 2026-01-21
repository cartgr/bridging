# Experiment 3: Voter PCA Spectrum Visualization

## Overview

This experiment visualizes which voters (positioned on a left-right political spectrum via PCA) approve each candidate in the French Election data. This helps visualize whether candidates are "bridging" (approved by voters across the spectrum) or "polarizing" (approved by only one side).

## Data

- **Source**: `data/processed/preflib/00026-*.npz` (6 French election polling stations)
- **Shape**: (16 candidates × N voters), N = 365-476 per station
- **Format**: Fully observed binary approval matrix (1.0/0.0, no NaN)

## Method

### Voter PCA Scores

Each voter is assigned a position on a 1D political spectrum by:
1. Treating each voter as a point in 16-dimensional space (their approval pattern)
2. Centering the data
3. Computing PCA and projecting onto PC1
4. PC1 captures the dominant axis of variation, typically left-right ideology

### Visualization

For each polling station, we create a plot with 16 rows (one per candidate):
- **Left column**: Candidate name
- **Middle column**: Horizontal line showing voter spectrum
  - Approvers: Colored dots (blue=left, red=right based on PC1)
  - Non-approvers: Small grey dots
- **Right column**: Bridging score

Candidates are sorted by bridging score (highest at top).

## Usage

```bash
python experiment_3/visualize.py
```

## Output

Plots are saved to `experiment_3/plots/`:
- `00026-00000001.png` through `00026-00000006.png`

## Interpretation

- **Bridging candidates**: Colored dots spread across both sides of spectrum
- **Polarizing candidates**: Colored dots concentrated on one side
- **Vertical dashed line**: Center of spectrum (PC1 = 0)

## Candidate Names

Row order in the matrix corresponds to:
1. Megret
2. Lepage
3. Gluckstein
4. Bayrou
5. Chirac
6. LePen
7. Taubira
8. Saint-Josse
9. Mamere
10. Jospin
11. Boutin
12. Hue
13. Chevenement
14. Madelin
15. Laguiller
16. Besancenot
