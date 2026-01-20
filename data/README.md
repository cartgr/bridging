# Data Pipeline for Voting/Approval Data

This directory contains tools for downloading and processing voting data from PrefLib into standardized sparse matrices for collaborative filtering experiments.

## Data Source

[PrefLib](https://preflib.org) is a library of preference data. We use categorical datasets (`.cat` files) that can be interpreted as approval voting.

## Datasets

### 00026 - French Election 2002 Approval Voting

Approval voting experiment conducted during the 2002 French presidential election. Voters at 6 polling stations approved/disapproved of 16 presidential candidates.

| File | Location | Candidates | Voters | Sparsity |
|------|----------|------------|--------|----------|
| `00026-00000001` | GylesNonains | 16 | 365 | 0.0% |
| `00026-00000002` | Orsay 1 | 16 | 409 | 0.0% |
| `00026-00000003` | Orsay 5 | 16 | 476 | 0.0% |
| `00026-00000004` | Orsay 6 | 16 | 460 | 0.0% |
| `00026-00000005` | Orsay 7 | 16 | 472 | 0.0% |
| `00026-00000006` | Orsay 12 | 16 | 415 | 0.0% |

### 00069 - Pol.is Conversations

Deliberation data from [Pol.is](https://pol.is), a real-time survey system for gathering open-ended feedback. Each file is a separate conversation where participants approved/disapproved/skipped statements.

| File | Topic | Statements | Voters | Sparsity |
|------|-------|------------|--------|----------|
| `00069-00000001` | Seattle $15/hour minimum wage | 54 | 339 | 87.5% |
| `00069-00000002` | Improving Bowling Green / Warren County | 896 | 2031 | 91.8% |
| `00069-00000003` | Energie (German climate) | 1039 | 1756 | 94.7% |
| `00069-00000004` | Produktion und Konsum (German climate) | 522 | 1116 | 91.9% |
| `00069-00000005` | Mobilität (German climate) | 2138 | 3142 | 96.5% |
| `00069-00000006` | Wohnen (German climate) | 613 | 1503 | 93.8% |
| `00069-00000007` | Ernährung und Landnutzung (German climate) | 1452 | 3616 | 96.2% |
| `00069-00000008` | Together, we'll build the BG of 2050 | 371 | 126 | 75.1% |
| `00069-00000009` | Can there be consensus on Brexit? | 50 | 204 | 54.5% |
| `00069-00000010` | Canadian Electoral Reform | 174 | 448 | 87.1% |
| `00069-00000011` | Concussions in the NFL | 298 | 1487 | 97.5% |
| `00069-00000012` | Youth engagement in police scrutiny (UK) | 39 | 26 | 42.4% |
| `00069-00000013` | Operation Marching Orders | 2162 | 6289 | 96.5% |
| `00069-00000014` | ScoopNZ: Affordable housing (NZ) | 165 | 381 | 80.0% |
| `00069-00000015` | Protecting and Restoring NZ's Biodiversity | 316 | 536 | 85.4% |
| `00069-00000016` | HiveMind: Freshwater Quality (NZ) | 80 | 117 | 61.9% |
| `00069-00000017` | Tax HiveMind Window (NZ) | 148 | 334 | 69.4% |
| `00069-00000018` | Universal Basic Income (NZ) | 71 | 234 | 61.8% |
| `00069-00000019` | Land use in San Juan Islands | 297 | 404 | 77.6% |
| `00069-00000020` | UberX regulation (vTaiwan) | 197 | 1921 | 88.7% |

## Matrix Format

All processed data is stored as `.npz` files with a standardized format:

- **Shape**: `(n_items, n_voters)` - rows are items/statements, columns are voters
- **Values**:
  - `1.0` = approved/agree
  - `0.0` = disapproved/disagree
  - `np.nan` = no response/abstain

### NPZ File Contents
- `matrix`: The (n_items, n_voters) numpy array
- `n_items`: Number of items/statements
- `n_voters`: Number of voters/participants
- `sparsity`: Fraction of missing values
- `source`: Data source identifier

## Folder Structure

```
data/
├── README.md           # This file
├── download.py         # Download raw data
├── process.py          # Convert to matrices
├── complete.py         # Matrix completion using KNNBaseline
├── raw/
│   └── preflib/
│       ├── 00026/      # French Election .cat files
│       └── 00069/      # Pol.is .cat files
├── processed/
│   └── preflib/        # Standardized numpy matrices
│       ├── 00026-*.npz
│       └── 00069-*.npz
└── completed/          # Matrix-completed Pol.is data
    └── 00069-*.npz
```

## Usage

### Download raw data
```bash
python data/download.py
```

### Process into matrices
```bash
python data/process.py
```

### Complete sparse matrices (Pol.is only)
```bash
python data/complete.py
python data/complete.py --k 40  # Number of neighbors (default: 40)
```

Uses [scikit-surprise](https://surpriselib.com/) KNNBaseline algorithm. We evaluated SVD, NMF, and KNN variants on the complete French Election data with random masking at 50-95% sparsity. KNNBaseline was the most stable across sparsity levels, while NMF collapsed at high sparsity and SVD was inconsistent. Predictions are thresholded at 0.5 to produce binary values (0 or 1).

### Load processed data
```python
import numpy as np

# Load a dataset
data = np.load('data/processed/preflib/00069-00000020.npz', allow_pickle=True)
matrix = data['matrix']  # Shape: (n_items, n_voters)
print(f"Shape: {matrix.shape}")
print(f"Sparsity: {float(data['sparsity']):.1%}")

# Count approvals, disapprovals, and missing
approvals = np.nansum(matrix == 1.0)
disapprovals = np.nansum(matrix == 0.0)
missing = np.isnan(matrix).sum()
```

## Value Mappings

### 2-Category (Yes/No) - Dataset 00026
| Original | Processed |
|----------|-----------|
| Yes (Category 1) | 1.0 |
| No (Category 2) | 0.0 |

### 3-Category (Disapproved/Neutral/Approved) - Dataset 00069
| Original | Processed |
|----------|-----------|
| Disapproved (Category 0) | 0.0 |
| Neutral/Skipped (Category 1) | np.nan |
| Approved (Category 2) | 1.0 |
