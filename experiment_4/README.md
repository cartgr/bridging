# Experiment 4: Synthetic Elections with Group Structure

This experiment creates synthetic elections with known group structures to understand bridging score behavior analytically.

## Overview

Groups are defined by approval of disjoint sets of "base" comments:
- Group 1 voters approve even-indexed base comments (0, 2, 4, ...)
- Group 2 voters approve odd-indexed base comments (1, 3, 5, ...)

This construction ensures:
- d_ij ≈ 1 between groups (voters from different groups disagree on all base comments)
- d_ij ≈ 0 within groups (voters from the same group agree on all base comments)

Focal comments vary in their approval rates from each group, allowing us to explore the bridging score surface.

## Bridging Score Formula

For a comment c with approvers N_c:

```
b(c) = (4/n²) × Σ_{i<j, i,j∈N_c} d_ij
```

Where d_ij = fraction of comments where voters i and j disagree.

### Analytical Expected Value

For two groups with sizes w and (1-w), approval rates a₁ and a₂:

```
E[b] = 4 × w × (1-w) × a₁ × a₂ × d_between
```

Where d_between = n_base / (n_base + 1) ≈ 1.

For equal groups (w = 0.5), this simplifies to:

```
E[b] = a₁ × a₂ × d_between ≈ a₁ × a₂
```

## Experimental Scenarios

### Scenario 1: Two Equal-Sized Groups

- **Setup**: n/2 voters in each group (w = 0.5)
- **Vary**: a₁ ∈ [0,1], a₂ ∈ [0,1] independently
- **Formula**: E[b] = a₁ × a₂
- **Output**: 3D surface plot of b(c) vs (a₁, a₂)

**Results**:
- Maximum bridging (~1.0) when both groups fully approve (a₁ = a₂ = 1)
- Bridging at (0.5, 0.5) ≈ 0.25
- Low bridging when approval is one-sided (a₁=1, a₂=0 or vice versa)
- Surface is symmetric: b(a₁, a₂) = b(a₂, a₁)

### Scenario 2: Two Unequal-Sized Groups

- **Setup**: Group 1 has fraction w, Group 2 has (1-w)
- **Constraint**: Fix total support w·a₁ + (1-w)·a₂ = 0.5
- **Vary**: w ∈ [0.05, 0.95], a₁ ∈ [0,1] (a₂ determined by constraint)
- **Formula**: E[b] = 4 × w × (1-w) × a₁ × a₂
- **Output**: 3D surface plot of b(c) vs (w, a₁)

**Results**:
- Valid region is where a₂ = (0.5 - w·a₁)/(1-w) falls in [0,1]
- Maximum bridging (~0.25) occurs near w=0.5, a₁≈0.55, a₂≈0.46
- Shows how group size imbalance affects bridging under fixed total support

## Usage

```bash
# Run all scenarios
python experiment_4/analyze.py

# Check output plots
ls experiment_4/plots/
```

## Output Files

Static plots (PNG):
- `plots/equal_groups_3d.png` - Scenario 1 surface plot
- `plots/unequal_groups_3d.png` - Scenario 2 surface plot

Interactive plots (HTML, viewable in browser):
- `plots/equal_groups_interactive.html` - Scenario 1 interactive 3D
- `plots/unequal_groups_interactive.html` - Scenario 2 interactive 3D

## Module Structure

```
experiment_4/
├── __init__.py           # Package exports
├── synthetic.py          # Generate synthetic election matrices
├── analyze.py            # Compute bridging scores, generate plots
├── README.md             # This file
└── plots/                # Output directory
```

## Key Functions

### synthetic.py

- `generate_synthetic_matrix()` - Create synthetic election with specified group structure
- `assign_voters_to_groups()` - Assign voters to groups based on size fractions
- `generate_and_compute_bridging_fast()` - Efficient bridging computation using known group structure

### analyze.py

- `compute_bridging_surface_equal_groups()` - Compute bridging scores for grid of (a₁, a₂)
- `compute_bridging_surface_unequal_groups()` - Compute bridging with constraint w·a₁ + (1-w)·a₂ = 0.5
- `plot_3d_surface()` - Create static 3D surface visualization
- `plot_3d_interactive()` - Create interactive 3D surface with plotly

## Parameters

Default values:
- `n_base_comments`: 1000 (only affects d_between ≈ 0.999)
- `grid_resolution`: 50×50 for surface plots
