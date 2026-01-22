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
| French Election 2002 | 6 | -0.504 ± 0.270 | -0.492 ± 0.321 |
| San Sebastian Poster | 2 | -0.821 ± 0.063 | -0.866 ± 0.029 |
| CTU Tutorial | 1 | -0.327 | -0.288 |
| French Election 2007 | 12 | 0.030 ± 0.684 | 0.064 ± 0.740 |
| Pol.is | 20 | 0.395 ± 0.368 | 0.823 ± 0.173 |

The correlation is **negative** for fully-observed approval voting datasets (more approved items have less diverse supporters) but **positive** for Pol.is (more approved comments have more diverse supporters).

### PD Bridging vs Approval

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election 2002 | 6 | 0.973 ± 0.005 | 0.988 ± 0.007 |
| San Sebastian Poster | 2 | 0.981 ± 0.008 | 0.997 ± 0.002 |
| CTU Tutorial | 1 | 0.968 | 0.996 |
| French Election 2007 | 12 | 0.970 ± 0.009 | 0.993 ± 0.007 |
| Pol.is | 20 | 0.896 ± 0.080 | 0.996 ± 0.005 |

PD bridging score is **very strongly correlated** with approval rate across all datasets (Spearman ρ > 0.99).

### PD Bridging vs Diversity

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election 2002 | 6 | -0.419 ± 0.232 | -0.436 ± 0.317 |
| San Sebastian Poster | 2 | -0.729 ± 0.117 | -0.849 ± 0.038 |
| CTU Tutorial | 1 | -0.206 | -0.234 |
| French Election 2007 | 12 | 0.150 ± 0.689 | 0.089 ± 0.738 |
| Pol.is | 20 | 0.611 ± 0.329 | 0.838 ± 0.170 |

PD bridging is negatively correlated with diversity for fully-observed datasets but positively correlated for Pol.is.

> **Note:** The Pol.is data was artificially completed using a matrix completion algorithm, which may affect these correlations.

### Polis vs Approval

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election 2002 | 6 | 0.721 ± 0.103 | 0.877 ± 0.071 |
| San Sebastian Poster | 2 | 0.916 ± 0.022 | 0.966 ± 0.013 |
| CTU Tutorial | 1 | 0.795 | 0.961 |
| French Election 2007 | 12 | 0.811 ± 0.150 | 0.922 ± 0.061 |
| Pol.is | 20 | 0.760 ± 0.180 | 0.989 ± 0.016 |

Polis consensus score is also **strongly correlated** with approval rate (Spearman ρ > 0.87), though the correlation is weaker than PD bridging.

### Polis vs Diversity

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election 2002 | 6 | -0.205 ± 0.254 | -0.284 ± 0.316 |
| San Sebastian Poster | 2 | -0.613 ± 0.153 | -0.831 ± 0.043 |
| CTU Tutorial | 1 | -0.086 | -0.218 |
| French Election 2007 | 12 | 0.186 ± 0.620 | 0.088 ± 0.686 |
| Pol.is | 20 | 0.645 ± 0.285 | 0.852 ± 0.163 |

Similar pattern to PD bridging: negative correlation with diversity for fully-observed datasets, positive for Pol.is.

### PD Bridging vs Polis

| Dataset | n | Pearson | Spearman |
|---------|---|---------|----------|
| French Election 2002 | 6 | 0.756 ± 0.122 | 0.890 ± 0.066 |
| San Sebastian Poster | 2 | 0.956 ± 0.013 | 0.960 ± 0.015 |
| CTU Tutorial | 1 | 0.901 | 0.963 |
| French Election 2007 | 12 | 0.805 ± 0.185 | 0.919 ± 0.055 |
| Pol.is | 20 | 0.912 ± 0.116 | 0.993 ± 0.011 |

The two methods are **highly correlated** (Spearman ρ > 0.89), indicating they largely agree on which items are "bridging."

## Data

| Dataset | Source | Files | Items |
|---------|--------|-------|-------|
| French Election 2002 (00026) | `data/processed/preflib/` | 6 | 16 |
| San Sebastian Poster (00033) | `data/processed/preflib/` | 2 | 17 |
| CTU Tutorial (00063) | `data/processed/preflib/` | 1 | 23 |
| French Election 2007 (00071) | `data/processed/preflib/` | 12 | 12 |
| Pol.is (00069) | `data/completed/` | 20 | varies |

> **Note:** Files with missing values have NaN replaced with 0 for correlation analysis. The analyze.py script skips files with NaN for plot generation.
