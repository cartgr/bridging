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
4. Display bridging score and approval rate for each candidate

## Data

| Dataset | Files | Items |
|---------|-------|-------|
| French Election 2002 (00026) | 6 | 16 candidates |
| San Sebastian Poster (00033) | 2 | 17 posters |
| CTU Tutorial (00063) | 1 | 23 time slots |
| French Election 2007 (00071) | 6 | 12 candidates |
| Pol.is (00069) | 20 | top 30 comments |

## Key Finding: Bridging ≈ Approval

In **32 out of 35 datasets**, the highest-bridging item is also the most approved item. This confirms that PD bridging score is dominated by approval rate (see Experiment 1 correlation analysis).

### Datasets Where Top Bridging ≠ Top Approval

Only 3 datasets show a different top item:

| Plot | Top Approval | Top Bridging |
|------|--------------|--------------|
| `00026-00000003.png` | Jospin (40.1%) | Bayrou (39.5%) |
| `00071-00000003.png` | Laguiller (54.3%) | Bayrou (52.3%) |
| `00071-00000005.png` | Bayrou (46.1%) | Royal (45.3%) |

In all three cases, the approval rates are very close (<2% difference), and Bayrou appears frequently as the bridging candidate—consistent with his centrist political position drawing support from across the spectrum.
