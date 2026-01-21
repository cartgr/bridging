# Experiment 3: Voter PCA Spectrum Visualization

Visualizes which voters (positioned on a political spectrum via PCA) approve each candidate.

## Usage

```bash
python experiment_3/visualize.py              # Ridgeline plots
python experiment_3/visualize.py --histogram  # Histogram plots
```

## Output

- Ridgeline plots: `plots/*.png`
- Histogram plots: `plots_histogram/*.png`

## Visualization Styles

| Style | Shows |
|-------|-------|
| **Ridgeline** | Density of approvers with blue→red color gradient |
| **Histogram** | Approval rate (blue) vs disapproval (grey) by spectrum position |

## Interpretation

- **Bridging candidates:** Approval spread across entire spectrum
- **Polarizing candidates:** Approval concentrated on one side

## Method

1. Project voters onto PC1 (dominant axis of variation)
2. For each candidate, visualize approval pattern across the spectrum
3. Sort candidates by bridging score (highest at top)

## Data

French Election (00026): `data/processed/preflib/00026-*.npz`
