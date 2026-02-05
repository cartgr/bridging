# Paper Experiments (Section 8)

Reproducibility guide for the three experiments in Section 8 of the paper. Each experiment compares three bridging/consensus metrics on approval voting data.

---

## Table of Contents

1. [Metrics Overview](#metrics-overview)
2. [Directory Structure](#directory-structure)
3. [Quick Start](#quick-start)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Data](#data)
6. [Experiment A: Qualitative Rankings](#experiment-a-qualitative-rankings)
7. [Experiment B: Robustness Under Missing Data](#experiment-b-robustness-under-missing-data)
8. [Experiment C: Real Polis Data](#experiment-c-real-polis-data)
9. [Dependencies](#dependencies)
10. [Troubleshooting](#troubleshooting)

---

## Metrics Overview

All three experiments compare the same three metrics. The data format throughout is a binary vote matrix `V` of shape `(n_items, n_voters)` where `V[c, i] = 1` means voter `i` approves item `c`, `V[c, i] = 0` means disapproval, and `NaN` means unobserved.

### 1. PD (Pairwise Disagreement)

The PD bridging score measures how much a comment is approved by voters who otherwise disagree with each other.

#### Mathematical Definition

For a fully-observed vote matrix `V` with `m` items (comments/candidates) and `n` voters, define pairwise voter disagreement:

```
d(i, j) = (1/m) * SUM_{c'=1}^{m} 1[V[c', i] != V[c', j]]
```

This is the fraction of items on which voters `i` and `j` cast different votes.

The PD bridging score for item `c` is:

```
b^PD(c) = (4/n^2) * SUM_{i < j, i,j in N_c} d(i, j)
```

where `N_c = {i : V[c, i] = 1}` is the set of voters who approve `c`.

**Normalization**: The factor `4/n^2` ensures `b^PD(c) in [0, 1]`. The maximum is achieved when exactly half the voters approve `c` and those approvers come from maximally disagreeing pairs.

**Intuition**: A comment scores high on PD when it is approved by voters who disagree on many other comments. This captures "bridging" -- consensus among otherwise-divided people.

#### Vectorized Ground Truth Implementation

The ground truth computation avoids materializing the full `O(n^2)` disagreement matrix by using the identity:

```
SUM_{i<j, i,j in N_c} d(i,j) = (1/m) * SUM_{c'=1}^{m} |N_c AND N_{c'}| * |N_c AND NOT(N_{c'})|
```

where `|N_c AND N_{c'}|` counts voters who approve both `c` and `c'`, and `|N_c AND NOT(N_{c'})|` counts voters who approve `c` but not `c'`. The product counts the number of disagreeing pairs on `c'` among approvers of `c`.

**Implementation** (`experiment_2/bridging.py`, `compute_bridging_scores_vectorized()`):

```python
approves = (matrix == 1.0)  # (n_items, n_voters) boolean
for c in range(n_items):
    approvers_c = approves[c]  # (n_voters,) boolean
    # For all c': count approvers of c who also approve c'
    approve_both = (approves & approvers_c).sum(axis=1)          # (n_items,)
    # For all c': count approvers of c who disapprove c'
    approve_c_disapprove_cp = (~approves & approvers_c).sum(axis=1)  # (n_items,)
    # Sum of products = total disagreement count
    total_disagreement = (approve_both * approve_c_disapprove_cp).sum()
    bridging_scores[c] = (4.0 / n_voters**2) * total_disagreement / n_items
```

**Complexity**: `O(m^2 * n)` — but in practice the inner operations are vectorized over `n`, so the loop is over `m` items (typically 12-54), making this fast.

#### Naive Estimator (for missing data)

When data is partially observed (entries may be `NaN`), we estimate `d(i,j)` using only co-observed items:

```
d_hat(i, j) = (1/|C_ij|) * SUM_{c' in C_ij} 1[V[c', i] != V[c', j]]
```

where `C_ij = {c' : V[c', i] and V[c', j] are both observed}`.

Approvers `N_c` are restricted to voters who observed item `c` and voted 1.

**Implementation** (`experiment_2/estimation.py`, `estimate_bridging_scores_naive()`):

```python
# Step 1: compute pairwise disagreement for all voter pairs
for i in range(n_voters):
    for j in range(i + 1, n_voters):
        both_observed = observed_mask[:, i] & observed_mask[:, j]
        n_both = both_observed.sum()
        if n_both == 0:
            d_naive[i, j] = np.nan
        else:
            votes_i = observed_matrix[both_observed, i]
            votes_j = observed_matrix[both_observed, j]
            d_naive[i, j] = (votes_i != votes_j).sum() / n_both

# Step 2: compute bridging scores using estimated disagreement
for c in range(n_items):
    approvers = observed_mask[c, :] & (observed_matrix[c, :] == 1.0)
    approver_indices = np.where(approvers)[0]
    total = sum(d_naive[i, j] for i, j in pairs(approver_indices) if not np.isnan(d_naive[i, j]))
    bridging_scores[c] = (4.0 / n_voters**2) * total
```

**Complexity**: `O(n^2 * m)` for pairwise disagreement + `O(m * |N_c|^2)` for bridging scores. The pairwise disagreement is the bottleneck — for `n = 2597` voters this means ~3.37M pairs.

**Bias**: Under MCAR, the naive estimator is **unbiased** for `d(i,j)` because each item is equally likely to be observed. Under informative missingness (e.g., Polis routing), it can be biased because controversial items are more likely to be shown.

---

### 2. Pol.is GIC (Group-Informed Consensus)

The Pol.is Group-Informed Consensus score measures cross-group approval using clustered voter groups. It is based on the [Polis](https://compdemocracy.org/group-informed-consensus/) open-source deliberation platform.

**Reference**: [Polis source code (conversation.clj)](https://github.com/compdemocracy/polis/blob/edge/math/src/polismath/math/conversation.clj)

#### Algorithm (Full Pipeline)

The GIC pipeline has 5 stages, detailed below. Implementation is in `experiment_5/polis.py`, function `polis_consensus_pipeline()`.

##### Stage 0: Participant Filtering

Remove voters with too few observations, matching Polis production behavior.

```python
# Count votes per participant
votes_per_participant = observed_mask.sum(axis=0)  # (n_voters,)
threshold = min(7, n_items)  # 7 minimum votes, or all items if fewer

# Keep participants meeting threshold
meets_threshold = votes_per_participant >= threshold
```

If fewer than 15 participants meet the threshold, greedily add the highest-contributing voters until 15 are reached:

```python
if meets_threshold.sum() < 15:
    sorted_indices = np.argsort(-votes_per_participant)  # descending
    kept_indices = sorted_indices[:min(15, n_voters)]
```

**Hyperparameters**:
- `min_votes = 7` (minimum votes per participant)
- `min_participants = 15` (minimum participants to retain)

##### Stage 1: Mean Imputation

Replace missing votes with the per-comment (row) mean of observed values. This makes the matrix complete for PCA.

```python
# For each comment c:
comment_mean[c] = sum(V[c, i] for i where observed) / count(observed on c)
# Default to 0.5 if no observations

# Replace NaN with comment mean:
imputed[c, i] = V[c, i] if observed else comment_mean[c]
```

**Rationale**: Polis uses column-mean imputation in its source code (where comments are columns in the voter-centric view). Since our data format is `(n_items, n_voters)`, this corresponds to row-mean imputation. The imputed value represents the "average opinion" on that comment across all voters who saw it.

##### Stage 2: PCA via Power Iteration

Compute 2-component PCA projections of voters.

**Data preparation**:
```python
data = imputed_matrix.T      # (n_voters, n_items) — voters as samples
data_centered = data - data.mean(axis=0)  # mean-center each comment
```

**Power iteration** (matching Polis's `pca.clj`):

The algorithm finds eigenvectors of `X^T X` (where `X` is the centered data matrix) using iterative multiplication:

```python
def power_iteration(data, n_iter=100, tol=1e-6):
    """Find leading eigenvector of data^T @ data."""
    v = random_unit_vector(n_features)  # seed=42
    for _ in range(n_iter):
        v_old = v
        Xv = data @ v             # (n_samples,)
        XtXv = data.T @ Xv        # (n_features,)
        v = XtXv / ||XtXv||       # normalize
        if ||v - v_old|| < tol:
            break
    return v
```

For multiple components, **deflation** is used: after finding each component, its contribution is subtracted from the data:

```python
def power_iteration_pca(data, n_components=2):
    residual = data.copy()
    components = []
    for k in range(n_components):
        v = power_iteration(residual)
        components.append(v)
        scores = residual @ v          # (n_voters,)
        residual -= np.outer(scores, v)  # remove component
    return np.array(components)  # (n_components, n_items)
```

**Voter projections**:
```python
projections = data_centered @ components.T  # (n_voters, 2)
```

**Hyperparameters**:
- `n_pca_components = 2`
- Power iteration: `n_iter = 100`, `tol = 1e-6`
- Random seed for initialization: `42`

##### Stage 3: K-Means Clustering with Silhouette Selection

Cluster voters in 2D PCA space using k-means. The number of clusters `k` is selected automatically.

```python
# Determine max k following Polis formula:
polis_max_k = min(5, 2 + n_voters // 12)

# Try each k from 2 to polis_max_k:
for k in range(2, polis_max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(projections)
    score = silhouette_score(projections, labels)
    if score > best_score:
        best_k, best_labels = k, labels
```

**Silhouette score**: Measures how similar each point is to its own cluster vs. other clusters. Range [-1, 1], higher is better. Requires at least 2 clusters with 2+ members each.

**Fallback**: If no valid clustering is found (e.g., too few voters), all voters are placed in a single cluster.

**Hyperparameters**:
- `min_k = 2`, `max_k = 5`
- `n_init = 10` (k-means restarts)
- `seed = 42`
- Cluster count formula: `k_max = min(5, 2 + floor(n_voters / 12))`

##### Stage 4: Group-Informed Consensus Score

For each comment `c` and group `g`, compute a Laplace-smoothed approval rate:

```
P_g(c) = (A_g(c) + 1) / (S_g(c) + 2)
```

where:
- `A_g(c)` = number of voters in group `g` who observed `c` and approved it
- `S_g(c)` = number of voters in group `g` who observed `c`
- The `+1` / `+2` is Laplace smoothing (equivalent to a Beta(1,1) = Uniform prior)

The consensus score is the **product** across all groups:

```
GIC(c) = PRODUCT_{g=1}^{K} P_g(c)
```

Computed in log-space for numerical stability:

```python
for c in range(n_items):
    log_consensus = 0.0
    for g in unique_labels:
        group_voters = (cluster_labels == g)
        group_observed = observed_mask[c, :] & group_voters
        S_g = group_observed.sum()
        A_g = (group_observed & (matrix[c, :] == 1.0)).sum()
        P_g = (A_g + 1) / (S_g + 2)
        log_consensus += log(P_g)
    consensus[c] = exp(log_consensus)
```

**Intuition**: A comment scores high when it has high approval in **every** group. The product ensures that low approval in any single group pulls the overall score down significantly.

**Score range**: `[0, 1]`. The maximum (near 1) requires near-unanimous approval across all groups. With `K = 3` groups each at 90% approval: `0.9^3 = 0.729`.

**Key property**: Pol.is GIC natively handles missing data (via imputation + participant filtering), so the same function is used for both fully-observed and sparse data. No separate "naive" estimator is needed.

---

### 3. p-mean (p = -10)

The p-mean bridging score generalizes the idea of "minimum cross-group approval" using weighted power means.

#### Mathematical Definition

For comment `c`, define "slices" of the population by their opinion on each other comment `c'`:

- `a_1(c, c')` = approval rate of `c` among voters who approve `c'`
  ```
  a_1(c, c') = |{i : V[c,i]=1 AND V[c',i]=1}| / |{i : V[c',i]=1}|
  ```
- `a_2(c, c')` = approval rate of `c` among voters who disapprove `c'`
  ```
  a_2(c, c') = |{i : V[c,i]=1 AND V[c',i]=0}| / |{i : V[c',i]=0}|
  ```
- `w(c')` = overall approval rate of `c'`
  ```
  w(c') = |{i : V[c',i]=1}| / n
  ```

The p-mean bridging score is:

```
b_p(c) = (1/(m-1)) * SUM_{c' != c} M_p(a_1(c,c'), a_2(c,c'); w(c'))
```

where `M_p` is the weighted power mean:

```
M_p(a_1, a_2; w) = (w * a_1^p + (1-w) * a_2^p)^(1/p)
```

#### Special cases of p

| p | M_p | Interpretation |
|---|-----|----------------|
| p -> +inf | max(a_1, a_2) | Maximum approval in either slice |
| p = 1 | w*a_1 + (1-w)*a_2 | Overall approval rate of c (trivial) |
| p = 0 | a_1^w * a_2^(1-w) | Weighted geometric mean |
| p = -1 | Weighted harmonic mean | Sensitive to low values |
| p = -10 | Approximates min(a_1, a_2) | Comment must be approved in both slices |
| p -> -inf | min(a_1, a_2) | Exact minimum |

We use **p = -10** throughout the paper, which closely approximates the minimum while remaining differentiable.

#### Handling zeros for negative p

When `a_1 = 0` or `a_2 = 0`, the term `a^p` for `p < 0` diverges. We define:

```
M_p(a_1, a_2; w) = 0   if a_1 = 0 or a_2 = 0 (for p < 0)
```

This is the correct limit: if a comment has zero approval in any population slice, its bridging contribution from that slice is zero.

#### Ground Truth Implementation

**Implementation** (`experiment_2/bridging.py`, `compute_bridging_pnorm()`):

```python
n_items, n_voters = matrix.shape

# Precompute
w = matrix.sum(axis=1) / n_voters                    # (n_items,) approval rate per item
n_approvers = matrix.sum(axis=1)                      # (n_items,)
n_disapprovers = n_voters - n_approvers               # (n_items,)
both_approve = matrix @ matrix.T                      # (n_items, n_items) overlap matrix

# a_1[c, c'] = both_approve[c, c'] / n_approvers[c']
a_1 = both_approve / n_approvers[np.newaxis, :]

# a_2[c, c'] = (n_approvers[c] - both_approve[c, c']) / n_disapprovers[c']
approve_c_only = n_approvers[:, np.newaxis] - both_approve
a_2 = approve_c_only / n_disapprovers[np.newaxis, :]

# p-mean term for each (c, c') pair
# For p < 0: set to 0 where a_1=0 or a_2=0
terms = np.where(
    (a_1 > 0) & (a_2 > 0),
    (w * a_1**p + (1-w) * a_2**p) ** (1/p),
    0.0
)
np.fill_diagonal(terms, 0.0)

# Average over c' != c
scores = terms.sum(axis=1) / (n_items - 1)
```

**Key insight**: `both_approve = matrix @ matrix.T` computes all pairwise overlap counts in one matrix multiply. Element `[c, c']` counts voters who approve both `c` and `c'`.

**Complexity**: `O(m^2 * n)` for the matrix multiply, then `O(m^2)` for the element-wise operations. Very fast in practice.

#### Naive Estimator (for missing data)

When data is partially observed, we restrict all quantities to co-observed voters. For each pair `(c, c')`, we only consider voters who observed **both** `c` and `c'`.

**Implementation** (`experiment_2/estimation.py`, `estimate_pnorm_naive()`):

```python
obs = observed_mask.astype(np.float64)
approve = np.where(observed_mask, observed_matrix, 0.0)

# Co-observed counts
co_observed = obs @ obs.T                        # [c,c'] = voters who saw both
n_approve_cprime = obs @ approve.T               # [c,c'] = voters who saw c AND approved c'
both_approve = approve @ approve.T               # [c,c'] = voters who approved both

# Conditional approval rates
w = n_approve_cprime / co_observed               # approval rate of c' among co-observers
a_1 = both_approve / n_approve_cprime            # P(approve c | approve c', co-observed)

# Voters who approve c but disapprove c' (among co-observers)
approve_c_and_obs_cprime = approve @ obs.T
approve_c_disapprove_cprime = approve_c_and_obs_cprime - both_approve
n_disapprove_cprime = co_observed - n_approve_cprime
a_2 = approve_c_disapprove_cprime / n_disapprove_cprime

# Compute p-mean terms, average over valid pairs (co_observed > 0)
```

**Key difference from ground truth**: The ground truth uses all `n` voters for each `(c, c')` pair, while the naive estimator uses only voters who observed both `c` and `c'`. The normalization `w(c')` is also local — it's the approval rate of `c'` among co-observers, not all voters.

**Correctness verification**: On fully-observed data (no NaN), `estimate_pnorm_naive` produces scores identical to `compute_bridging_pnorm` (max absolute difference = 0.0).

---

## Directory Structure

```
paper_experiments/
├── README.md                  # This file
├── experiment_a/              # Qualitative rankings (French elections)
│   ├── run.py                 # Compute scores, save to JSON
│   └── plot.py                # Ridgeline plots from JSON
├── experiment_b/              # Robustness under missing data
│   ├── run.py                 # MCAR + simulated routing, save to JSON
│   └── plot.py                # Robustness plots from JSON
├── experiment_c/              # Real Polis data (00069)
│   ├── run.py                 # Compute scores on sparse data, save to JSON
│   └── plot.py                # MDS ridgeline plots from JSON
├── results/                   # JSON data files produced by run.py scripts
└── plots/                     # PNG figures produced by plot.py scripts
```

Each experiment is split into `run.py` (computation, saves JSON) and `plot.py` (visualization, reads JSON). This separation means you can re-plot without re-running expensive computations.

---

## Quick Start

All commands should be run from the **repository root** (the parent of `paper_experiments/`):

```bash
# Experiment A: ~10 seconds
python paper_experiments/experiment_a/run.py
python paper_experiments/experiment_a/plot.py

# Experiment B: ~2-4 hours (dominated by naive PD on 2597 voters)
python paper_experiments/experiment_b/run.py
python paper_experiments/experiment_b/plot.py

# Experiment C: ~1-2 hours (dominated by voter similarity on large datasets)
python paper_experiments/experiment_c/run.py
python paper_experiments/experiment_c/plot.py
```

---

## Data Processing Pipeline

The raw data comes from [PrefLib](https://www.preflib.org/) and goes through several processing stages before being used in experiments.

### Step 1: Download raw data

```bash
python data/download.py
```

Downloads `.cat` (categorical) files from the PrefLib GitHub repository for each dataset:
- **00026** (frenchapproval): 6 files, one per polling station (2002 French presidential election)
- **00069** (polis): 20 files, one per Pol.is conversation
- **00071** (french-approval-2007): 12 files, one per polling station (2007 French presidential election)

Files are saved to `data/raw/preflib/{dataset_id}/`.

### Step 2: Parse into numpy matrices

```bash
python data/process.py
```

Parses each `.cat` file into a numpy matrix of shape `(n_items, n_voters)`.

**PrefLib `.cat` format**: Each line represents a group of voters with identical ballots:
```
count: approved_items, disapproved_items           # 2-category (French elections)
count: disapproved_items, neutral_items, approved_items  # 3-category (Polis)
```

Item sets can be single integers (`6`) or sets (`{1,2,3}`).

**Value encoding**:
| Category | Value | Meaning |
|----------|-------|---------|
| Approved | `1.0` | Voter approves this item/candidate |
| Disapproved | `0.0` | Voter disapproves |
| Neutral/Pass | `0.5` | Voter saw but chose not to vote (Polis only) |
| Unobserved | `NaN` | Voter never saw this item (Polis only) |

French election data (00026, 00071) is 2-category (binary, no NaN). Polis data (00069) is 3-category with extensive NaN (typically 84-96% unobserved).

**Important**: Item IDs in 2-category files are 1-indexed (converted to 0-indexed in the matrix). Item IDs in 3-category files are already 0-indexed.

Output: `data/processed/preflib/{dataset_id}-{file_number}.npz`

### Step 3: Combine polling stations (French elections only)

The individual `.npz` files for French elections represent separate polling stations, each with the same candidates but different voters. The `*-combined.npz` files are created by **horizontal concatenation** (stacking voter columns):

```python
# Conceptually:
matrices = [np.load(f)["matrix"] for f in individual_files]
combined = np.hstack(matrices)  # shape: (n_candidates, sum_of_voters)
```

| Combined file | Source files | Candidates | Total voters |
|---------------|-------------|-----------|--------------|
| `00026-combined.npz` | 00026-1 through 00026-6 | 16 | 365+409+476+460+472+415 = 2597 |
| `00071-combined.npz` | 00071-7 through 00071-12 | 12 | subset of 12 polling stations = 2836 |

Note: `00071-combined.npz` uses only polling stations 7-12 (not all 12 stations).

### Step 4: Matrix completion (not used in paper experiments)

```bash
python data/complete.py  # Optional, uses KNNBaseline collaborative filtering
```

This step imputes missing entries in Polis data using KNNBaseline (k=40) from the `surprise` library. **The paper experiments do NOT use completed data** — they use either the raw sparse matrices (Experiment C) or the fully-observed French election matrices (Experiments A and B).

### Pass vote handling

In the Polis data, pass votes (`0.5`) represent "voter saw the comment but chose not to approve or disapprove." In the paper experiments:
- Pass votes are stored in the matrix as `0.5`
- For PD and p-mean computation, pass votes are treated as observed non-approvals (they contribute to `d_ij` calculations). Specifically, the `== 1.0` check means pass votes are **not** counted as approvals.
- For Pol.is GIC, pass votes are included in the observation mask (`S_g` increments) but not counted as approvals (`A_g` does not increment). So `P_g(c)` treats pass as disapproval.
- `NaN` entries (truly unobserved) are excluded from all computations

---

## Data

### French Election Data (PrefLib)

Binary approval voting matrices from the PrefLib repository.

| File | Election | Candidates | Voters | Source |
|------|----------|-----------|--------|--------|
| `00026-combined.npz` | France 2002 | 16 | 2597 | PrefLib 00026, all polling stations combined |
| `00071-combined.npz` | France 2007 | 12 | 2836 | PrefLib 00071, polling stations 7-12 combined |

These are **fully observed** matrices (no NaN values). Values are `0.0` (disapprove) or `1.0` (approve). Shape: `(n_candidates, n_voters)`.

**Candidate names (2002)**: Megret, Lepage, Gluckstein, Bayrou, Chirac, LePen, Taubira, Saint-Josse, Mamere, Jospin, Boutin, Hue, Chevenement, Madelin, Laguiller, Besancenot

**Candidate names (2007)**: Besancenot, Buffet, Schivardi, Bayrou, Bove, Voynet, Villiers, Royal, Nihous, Le Pen, Sarkozy, Laguiller

### Polis Data (PrefLib 00069)

20 real Pol.is conversations from the PrefLib dataset 00069. These are **sparse** matrices — most entries are NaN (unobserved). Observation rates range from ~4% to ~16%.

| File | Items | Voters | Obs Rate |
|------|-------|--------|----------|
| `00069-00000001.npz` | 54 | 339 | 15.7% |
| `00069-00000002.npz` | 896 | 2031 | 12.4% |
| `00069-00000003.npz` | 1039 | 1756 | 7.2% |
| ... | ... | ... | ... |
| `00069-00000020.npz` | varies | varies | varies |

Comment texts are loaded via `load_all_comments_for_dataset()` from `experiment_6/comments.py`, which fetches `.cat` files from the PrefLib GitHub repository to map matrix row indices to comment IDs, then matches them to locally stored comment CSV files in `data/polis_comments/`.

### Data format

All `.npz` files contain a single array under the key `"matrix"`:

```python
data = np.load("data/processed/preflib/00026-combined.npz")
matrix = data["matrix"]  # shape: (n_items, n_voters), dtype: float64
```

---

## Experiment A: Qualitative Rankings

### Purpose

Produce a qualitative comparison of the three metrics on fully-observed French election data. The ridgeline plots show which voters (positioned on a left-right political spectrum) approve each candidate, alongside the metric rankings.

### Methodology

1. **Load** the fully-observed vote matrix from `.npz`.
2. **PCA positioning**: Compute PC1 scores for each voter:
   ```python
   voter_matrix = matrix.T  # (n_voters, n_candidates)
   centered = voter_matrix - voter_matrix.mean(axis=0)
   U, S, Vt = np.linalg.svd(centered, full_matrices=False)
   pc1_scores = U[:, 0] * S[0]
   ```
   PC1 captures the dominant left-right political axis. This is appropriate because the French election data is fully observed (no NaN values).

3. **Compute metrics**:
   - PD: `compute_bridging_scores_vectorized(matrix)` — ground truth, no estimation needed.
   - Pol.is GIC: `polis_consensus_pipeline(matrix, ones_mask)` — with a fully-observed mask.
   - p-mean: `compute_bridging_pnorm(matrix, p=-10)` — ground truth.

4. **Save** all scores, voter PC1 positions, approval fractions, and candidate names to `results/experiment_a.json`.

5. **Plot**: For each dataset, produce a ridgeline plot showing the **top 5 candidates by PD score** (descending). Each row shows:
   - Left: candidate name
   - Center: KDE density of approving voters along PC1 (bandwidth=0.3), unnormalized (height proportional to count), filled with a blue-to-red gradient (coolwarm colormap mapped to PC1 position)
   - Right columns: Approval %, PD rank+score, Pol.is GIC rank+score, p-mean rank+score

### Outputs

| File | Description |
|------|-------------|
| `results/experiment_a.json` | All computed scores and metadata |
| `plots/experiment_a_00026-combined.png` | 2002 election ridgeline (top 5) |
| `plots/experiment_a_00071-combined.png` | 2007 election ridgeline (top 5) |

### Key code paths

```python
# run.py
from experiment_2.bridging import compute_bridging_scores_vectorized, compute_bridging_pnorm
from experiment_5.polis import polis_consensus_pipeline
from experiment_3.visualize import CANDIDATE_NAMES_2002, CANDIDATE_NAMES_2007
```

---

## Experiment B: Robustness Under Missing Data

### Purpose

Measure how each metric's ranking degrades as data becomes increasingly incomplete. Two types of missingness are tested: random (MCAR) and informative (simulated Polis routing).

### Ground Truth

For each dataset, ground truth is computed on the **full, fully-observed** matrix:

```python
gt_pd = compute_bridging_scores_vectorized(matrix)              # PD on full data
gt_polis, _ = polis_consensus_pipeline(matrix, full_mask)        # Pol.is GIC on full data
gt_pmean = compute_bridging_pnorm(matrix, p=-10)                # p-mean on full data
```

Each metric's ground truth is compared against itself — i.e., PD-estimated is compared to PD-ground-truth via Spearman, not cross-compared.

### Evaluation Measures

For each trial, three measures compare the estimated ranking to the ground-truth ranking:

1. **Spearman rank correlation** (`scipy.stats.spearmanr`): Measures monotonic association between estimated and ground-truth score vectors. Range [-1, 1], where 1 = identical rankings.

2. **Kendall tau** (`scipy.stats.kendalltau`): Counts concordant vs. discordant pairs. More sensitive to local rank swaps than Spearman. Range [-1, 1].

3. **Top-1 frequency**: Binary indicator — does the estimated top-ranked item match the ground-truth top-ranked item?
   ```python
   def top1_match(estimated, ground_truth):
       valid = ~np.isnan(estimated) & ~np.isnan(ground_truth)
       est_valid = np.where(valid, estimated, -np.inf)
       gt_valid = np.where(valid, ground_truth, -np.inf)
       return float(np.argmax(est_valid) == np.argmax(gt_valid))
   ```

All measures exclude NaN scores before computation. If fewer than 3 valid items remain, the trial returns NaN for correlation measures.

### Sub-experiment B1: MCAR (Missing Completely At Random)

**Procedure**:

1. For each mask rate `r` in `{0.1, 0.3, 0.5, 0.6, 0.7}` (corresponding to observation rates `{0.9, 0.7, 0.5, 0.4, 0.3}`):
2. For each of 20 trials (seeds generated from `np.random.default_rng(42).integers(0, 2^31, size=20)`):
3. Apply MCAR masking using `apply_random_mask(matrix, mask_rate, seed)`:
   - Each entry is independently hidden with probability `r`
   - Constraint: each item (row) retains at least 2 observed entries
   - Constraint: each voter (column) retains at least 1 observed entry
   - Unmasked entries that violate constraints are randomly restored
4. Compute three estimates on the masked data:
   ```python
   pd_est = estimate_bridging_scores_naive(masked_matrix, observed_mask)
   polis_est, _ = polis_consensus_pipeline(masked_matrix, observed_mask)
   pmean_est = estimate_pnorm_naive(masked_matrix, observed_mask, p=-10)
   ```
5. Compute Spearman rho, Kendall tau, and top-1 match vs ground truth for each metric.

#### MCAR Masking Implementation Detail

**Implementation** (`experiment_5/masking.py`, `apply_random_mask()`):

```python
rng = np.random.default_rng(seed)
random_values = rng.random(matrix.shape)     # uniform [0, 1) per entry
observed_mask = random_values >= mask_rate    # True = observed
```

Then minimum-observation constraints are enforced:

```python
# For each item with too few observations:
for i in range(n_items):
    if mask[i, :].sum() < min_observed_per_item:  # default: 2
        masked_positions = np.where(~mask[i, :])[0]
        to_unmask = rng.choice(masked_positions, size=deficit, replace=False)
        mask[i, to_unmask] = True

# Same for each voter with too few observations (min 1)
```

This guarantees every item has at least 2 votes and every voter has at least 1 vote, preventing degenerate cases in metric computation.

### Sub-experiment B2: Simulated Polis Routing

**Purpose**: Test robustness under **informative** missingness — where which items a voter sees depends on the evolving state of the conversation.

#### How Polis Routing Works

In production, Polis does not show every comment to every voter. Instead, it uses a priority-based routing algorithm to decide which comment to show next. Comments that are controversial, under-seen, or "extreme" (high PCA loading) get shown more often. This creates non-random missingness — the observation pattern is correlated with comment properties.

#### Priority Formula (from Polis Source Code)

The priority of comment `c` after receiving `S` total votes is:

```
priority(c) = [importance(c) * vote_factor(c)]^2
```

where:

```
importance(c) = (1 - p) * (E + 1) * a
```

and:

```
vote_factor(c) = 1 + 8 * 2^(-S/5) = 1 + 2^(3 - S/5)
```

The quantities are:
- **`a = (A + 1) / (S + 2)`**: Laplace-smoothed approval rate. `A` = number of approvals, `S` = total votes. New comments start at `a = 1/2`.
- **`p = (P + 1) / (S + 2)`**: Laplace-smoothed pass rate. `P` = number of passes. In binary setting (no passes), `P = 0`, so `p = 1/(S+2)`.
- **`E`**: PCA extremeness — the L2 norm of the comment's loading vector in the first 2 principal components of the vote matrix. Measures how "polarizing" a comment is. Computed via power iteration PCA on the mean-imputed, participant-filtered vote matrix (same pipeline as Pol.is GIC Stage 0-2).
- **`vote_factor`**: Exploration bonus for under-seen comments. At `S = 0`: factor = `1 + 8 = 9`. At `S = 15`: factor = `1 + 8 * 2^(-3) = 2`. At `S = 50`: factor ~= 1. This ensures new comments are quickly shown.

**Implementation** (`experiment_2/priority.py`, `compute_priorities()`):

```python
n_votes, n_agrees, n_passes = compute_vote_stats(matrix, observed_mask)
a = (n_agrees + 1) / (n_votes + 2)          # Laplace-smoothed approval
p = (n_passes + 1) / (n_votes + 2)          # Laplace-smoothed pass rate
extremeness = compute_pca_extremeness(matrix, observed_mask)
importance = (1 - p) * (1 + extremeness) * a
vote_factor = 1 + np.power(2.0, 3 - n_votes / 5)
priorities = (importance * vote_factor) ** 2
```

#### PCA Extremeness Computation

PCA extremeness measures how much a comment contributes to voter polarization. The full computation:

1. **Filter participants**: Same filtering as Pol.is GIC (min 7 votes, at least 15 participants)
2. **Impute**: Replace NaN with per-comment mean
3. **PCA**: Power iteration (2 components) on transposed centered data
4. **Extremeness**: For each comment `c`, its loading vector across 2 PCs is `[component_1[c], component_2[c]]`. The extremeness is:
   ```
   E(c) = sqrt(component_1[c]^2 + component_2[c]^2)
   ```

Comments with high extremeness are those that most strongly divide the electorate along the first two principal axes.

#### Sampling Process

Comments are sampled **without replacement** using probability-proportional-to-size (PPS) sampling:

```python
p_c = priority(c) / SUM_{c' eligible} priority(c')
```

After sampling comment `c`, it becomes ineligible for this voter. The probabilities are renormalized over the remaining eligible comments.

**Implementation** (`experiment_2/priority.py`, `compute_sampling_probabilities()`):

```python
masked_priorities = priorities * eligible_mask
total = masked_priorities.sum()
if total == 0:
    probs = eligible_mask / eligible_mask.sum()  # uniform fallback
else:
    probs = masked_priorities / total
```

#### Full Routing Simulation

**Implementation** (`paper_experiments/experiment_b/run.py`, `simulate_polis_routing_naive()`):

```python
def simulate_polis_routing_naive(ground_truth, target_obs_rate, seed):
    rng = np.random.default_rng(seed)
    n_items, n_voters = ground_truth.shape

    # Each voter sees approximately target_obs_rate * n_items comments
    target_k = max(1, int(n_items * target_obs_rate))
    # Add small variation: k sampled uniformly from [target_k-2, target_k+2]
    votes_dist = np.arange(max(1, target_k - 2), min(n_items, target_k + 2) + 1)

    observed_matrix = np.full((n_items, n_voters), np.nan)
    observed_mask = np.zeros((n_items, n_voters), dtype=bool)

    for voter_idx in range(n_voters):
        # Step 1: Compute PCA extremeness on current observed data
        extremeness = compute_pca_extremeness(observed_matrix, observed_mask)

        # Step 2: Compute priority for each comment (uses current vote counts)
        priorities = compute_priorities(observed_matrix, observed_mask, extremeness)

        # Step 3: Determine how many comments this voter sees
        k_votes = min(rng.choice(votes_dist), n_items)

        # Step 4: Sample k comments without replacement
        eligible = np.ones(n_items, dtype=bool)
        for _ in range(k_votes):
            p = compute_sampling_probabilities(priorities, eligible)
            if p.sum() == 0:
                break
            sampled = rng.choice(n_items, p=p)
            eligible[sampled] = False

            # Reveal ground-truth vote
            observed_mask[sampled, voter_idx] = True
            observed_matrix[sampled, voter_idx] = ground_truth[sampled, voter_idx]
```

**Key behavioral properties**:
- **Early exploration**: The `vote_factor` gives high priority to unseen comments (S=0), ensuring all comments get some initial votes.
- **Controversy amplification**: As votes accumulate, extremeness `E` boosts comments that split the PCA space, meaning controversial comments get shown more.
- **Sequential dependence**: Each voter's comment selection depends on all previous voters' data. This creates temporal correlation in the observation pattern.
- **Non-random missingness**: The probability of observing `V[c, i]` depends on comment `c`'s properties (extremeness, approval rate, vote count), violating MCAR.

#### After Routing: Metric Estimation

After all voters have been routed, compute all three metrics on the observed data using naive estimation (identical to B1 step 4). **No IPW correction** is applied — this is intentional. The paper experiments test how well each metric performs under informative missingness without any bias correction.

**Key difference from `experiment_5/`**: The full experiment_5 robustness pipeline also computes IPW-corrected estimates using Monte Carlo estimation of inclusion probabilities (500 samples per voter per item). This is omitted here because (a) it is extremely slow, and (b) the paper only compares the three metrics without correction.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MCAR mask rates | 0.1, 0.3, 0.5, 0.6, 0.7 | Observation rates: 0.9, 0.7, 0.5, 0.4, 0.3 |
| MCAR trials per rate | 20 | |
| Routing target rates | 0.3, 0.5, 0.6, 0.7, 0.9 | |
| Routing trials per rate | 20 | |
| Base seed | 42 | For reproducible seed generation |
| Min observed per item | 2 | MCAR constraint |
| Min observed per voter | 1 | MCAR constraint |
| Routing votes variance | +/- 2 | Each voter sees target_k +/- 2 comments |

### Outputs

| File | Description |
|------|-------------|
| `results/experiment_b.json` | All trial results (correlations, obs rates, metadata) |
| `plots/experiment_b_mcar_spearman_{dataset}.png` | MCAR Spearman rho plot |
| `plots/experiment_b_mcar_kendall_{dataset}.png` | MCAR Kendall tau plot |
| `plots/experiment_b_mcar_top1_{dataset}.png` | MCAR top-1 frequency plot |
| `plots/experiment_b_routing_spearman_{dataset}.png` | Routing Spearman rho plot |
| `plots/experiment_b_routing_kendall_{dataset}.png` | Routing Kendall tau plot |
| `plots/experiment_b_routing_top1_{dataset}.png` | Routing top-1 frequency plot |

Total: 12 plots (3 measures x 2 experiment types x 2 datasets).

### Plot format

Each plot shows the measure (y-axis) vs observation rate (x-axis) with three lines:
- Blue circles: PD
- Orange squares: Pol.is GIC
- Green triangles: p-mean

Shaded bands show +/- 1 standard deviation across trials.

### Key code paths

```python
# run.py
from experiment_2.bridging import compute_bridging_scores_vectorized, compute_bridging_pnorm
from experiment_2.estimation import estimate_bridging_scores_naive, estimate_pnorm_naive
from experiment_5.polis import polis_consensus_pipeline
from experiment_5.masking import apply_random_mask
from experiment_2.priority import compute_priorities, compute_pca_extremeness, compute_sampling_probabilities
```

### Performance notes

The dominant cost is `estimate_bridging_scores_naive()`, which has an `O(n_voters^2)` inner loop for pairwise disagreement. For `00026-combined` (2597 voters), this means ~3.4M pairs per trial, taking ~26 seconds. Total MCAR time for one dataset: ~100 trials * 26s = ~43 minutes.

The routing simulation is slower per trial because `compute_pca_extremeness()` and `compute_priorities()` are called once per voter (2597 times per trial). However, these are cheap individually (~10ms each).

---

## Experiment C: Real Polis Data

### Purpose

Apply all three metrics to real, sparse Pol.is conversation data and visualize the results. Unlike Experiments A and B which use fully-observed data (or artificially masked data), this uses naturally sparse data where voters typically see only 4-16% of comments.

### Methodology

For each of the 20 `00069-*.npz` datasets:

1. **Voter positioning via MDS**:
   - Compute pairwise voter similarity: for each pair `(i, j)`, find comments both rated, compute agreement rate. If no shared comments, use 0.5 (neutral).
     ```python
     shared = observed[:, i] & observed[:, j]
     if shared.sum() == 0:
         similarity[i, j] = 0.5
     else:
         agree = (matrix[shared, i] == matrix[shared, j]).sum()
         similarity[i, j] = agree / shared.sum()
     ```
   - Convert similarity to distance: `distance = 1 - similarity`
   - Apply MDS to get 1D positions:
     ```python
     mds = MDS(n_components=1, dissimilarity='precomputed', random_state=42)
     positions = mds.fit_transform(distance).flatten()
     ```
   - MDS is used instead of PCA because the data is sparse — PCA on imputed data can introduce artifacts, while MDS on pairwise similarity uses only actually co-observed votes.

2. **Compute metrics** (all naive, appropriate for sparse data):
   ```python
   pd_scores = estimate_bridging_scores_naive(matrix, observed)
   polis_scores, meta = polis_consensus_pipeline(matrix, observed)
   pmean_scores = estimate_pnorm_naive(matrix, observed, p=-10)
   ```

3. **Load comment texts** via `load_all_comments_for_dataset(file_id)`:
   - Fetches `.cat` files from PrefLib GitHub to get matrix index -> comment ID mapping
   - Matches comment IDs to locally stored comment CSVs in `data/polis_comments/`
   - Returns `dict[int, str]` mapping matrix row index to comment text

4. **Compute approval fractions**: For each comment, the approval fraction is computed among voters who actually voted on it (not all voters):
   ```python
   voted = observed[c, :]
   approval_frac = (matrix[c, :] == 1)[voted].sum() / voted.sum()
   ```

5. **Save** everything to `results/experiment_c.json`.

6. **Plot**: For each dataset, produce a ridgeline plot showing the **top 5 comments by PD score**. Layout:
   - Left: wrapped comment text (55 chars wide, 6pt font)
   - Center: KDE of approving voters along MDS axis (bandwidth=0.3, unnormalized, coolwarm gradient)
   - Right columns: Approval% (n_voted), PD rank+score, Pol.is GIC rank+score, p-mean rank+score

### Assumptions

- **Voter similarity**: Agreement rate on co-observed items is a reasonable proxy for overall similarity. Pairs with no shared items get similarity 0.5 (the uninformative prior).
- **MDS**: The 1D MDS embedding captures the dominant axis of voter disagreement. This is analogous to PC1 in fully-observed data but works directly from pairwise similarities without imputation.
- **Naive estimation**: Without knowledge of the observation mechanism (Polis's routing algorithm), we cannot compute IPW corrections. The naive estimators treat observed data as representative.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| MDS components | 1 | 1D voter positioning |
| MDS random state | 42 | Reproducibility |
| KDE bandwidth | 0.3 | For ridgeline density estimation |
| Polis max_k | 5 | Maximum clusters for k-means |
| Polis min_votes | 7 | Minimum votes per participant |
| p-mean p | -10 | Approximate minimum |
| Top N displayed | 5 | Top comments by PD score |
| Text wrap width | 55 | Characters per line in comment labels |

### Outputs

| File | Description |
|------|-------------|
| `results/experiment_c.json` | All scores, voter positions, comment texts |
| `plots/experiment_c_00069-00000001.png` | Ridgeline plot for dataset 1 |
| ... | One plot per dataset |
| `plots/experiment_c_00069-00000020.png` | Ridgeline plot for dataset 20 |

### Performance notes

The dominant cost is `compute_voter_similarity_sparse()`, which is `O(n_voters^2 * avg_shared_items)`. For the largest datasets (~6000 voters), this takes ~17 minutes. The full run across all 20 datasets takes 1-2 hours.

### Key code paths

```python
# run.py
from experiment_2.estimation import estimate_bridging_scores_naive, estimate_pnorm_naive
from experiment_5.polis import polis_consensus_pipeline
from experiment_6.comments import load_all_comments_for_dataset
```

---

## Dependencies

### Internal modules

All computation is delegated to existing modules in the repository:

| Module | Functions used | Purpose |
|--------|---------------|---------|
| `experiment_2/bridging.py` | `compute_bridging_scores_vectorized()`, `compute_bridging_pnorm()` | Ground-truth PD and p-mean scores |
| `experiment_2/estimation.py` | `estimate_bridging_scores_naive()`, `estimate_pnorm_naive()` | Naive PD and p-mean on sparse data |
| `experiment_2/priority.py` | `compute_priorities()`, `compute_pca_extremeness()`, `compute_sampling_probabilities()`, `power_iteration_pca()` | Polis routing simulation and PCA |
| `experiment_5/polis.py` | `polis_consensus_pipeline()` | Pol.is GIC (handles both full and sparse) |
| `experiment_5/masking.py` | `apply_random_mask()` | MCAR masking |
| `experiment_3/visualize.py` | `CANDIDATE_NAMES_2002`, `CANDIDATE_NAMES_2007` | Candidate name mappings |
| `experiment_6/comments.py` | `load_all_comments_for_dataset()` | Polis comment text loading |

### External packages

```
numpy
scipy          # spearmanr, kendalltau, gaussian_kde
matplotlib     # plotting
scikit-learn   # MDS, KMeans, silhouette_score
tqdm           # progress bars
```

---

## Troubleshooting

**`estimate_bridging_scores_naive()` is very slow**: This function has `O(n_voters^2)` complexity. For datasets with >2000 voters, each call takes 20-30 seconds. In Experiment B, this is called once per trial (100 trials per dataset = ~43 minutes per dataset).

**Pol.is GIC returns different scores on re-run**: K-means clustering is seeded (`seed=42` by default), but results can vary slightly across numpy/sklearn versions. The number of clusters selected by silhouette can also change.

**Comment texts missing in Experiment C**: Comment loading requires network access (fetches `.cat` files from GitHub) and local CSV files in `data/polis_comments/`. If either fails, comments fall back to `"Item N"` placeholders.

**SpearmanRConstantInputWarning in Experiment B**: At high mask rates, some metrics may produce constant scores (e.g., all zeros), making Spearman undefined. These trials return NaN and are excluded from the mean/std computation in plots.

**p-mean returns all zeros on sparse data**: This was a bug in an earlier version of `estimate_pnorm_naive()` that called `compute_bridging_pnorm()` directly, which doesn't handle NaN. The current version computes co-observed statistics via matrix products of observation masks. If you see all-zero p-mean scores, ensure you have the latest `experiment_2/estimation.py`.

---

## Appendix Experiment: Naive PD vs IPW-Corrected PD

### Purpose

Compare the naive PD estimator against a properly IPW-corrected PD estimator under MCAR, when the observation rate is known.

### Mathematical Background

The full PD formula is:
```
θ_PD(x) = (4/mn²) Σ_y Σ_{i<j} 1[A_{i,x}=1, A_{j,x}=1] · 1[A_{i,y}≠A_{j,y}]
```

Under missing data, each term in the sum requires observing 4 entries:
- V_{i,x} = 1 (voter i observed comment x)
- V_{j,x} = 1 (voter j observed comment x)
- V_{i,y} = 1 (voter i observed comment y)
- V_{j,y} = 1 (voter j observed comment y)

The IPW estimator weights each observed term by the inverse joint probability:
```
weight = 1 / (q_{i,x} · q_{j,x} · q_{i,y} · q_{j,y})
```

Under MCAR with observation rate q:
- q_{i,c} = q for all (i, c)
- weight = 1/q^4 for all observed quadruples

### Methodology

1. Load datasets (00026-combined, 00071-combined)
2. Compute ground truth PD on full matrix
3. For each observation rate (5%, 27.5%, 50%, 72.5%, 95%):
   - For each trial (20 trials):
     - Apply MCAR mask at the target rate
     - Compute naive PD (`estimate_bridging_scores_naive`)
     - Compute IPW PD (`estimate_bridging_scores_ipw_mcar` with known q)
     - Compute Spearman/Kendall correlation vs ground truth for both

### IPW Estimator Implementation

```python
def estimate_bridging_scores_ipw_mcar(observed_matrix, observed_mask, obs_rate):
    """IPW-corrected PD under MCAR assumption."""
    n_items, n_voters = observed_matrix.shape
    weight = 1.0 / (obs_rate ** 4)  # IPW weight for observed 4-tuple
    normalization = 4.0 / (n_items * n_voters ** 2)

    for x in range(n_items):
        # Find observed approvers of x
        approvers_x = where(observed_mask[x] & (matrix[x] == 1.0))

        for i, j in pairs(approvers_x):
            for y != x:
                # Check all 4 entries observed
                if not (observed_mask[y, i] and observed_mask[y, j]):
                    continue

                # Check disagreement on y
                if disagree(matrix[y, i], matrix[y, j]):
                    total += weight  # Apply IPW weight

        bridging_scores[x] = normalization * total
```

### Key Differences from Naive Estimator

| Aspect | Naive | IPW |
|--------|-------|-----|
| Disagreement | Estimated from co-observed items | Weighted by 1/q^4 |
| Bias under MCAR | Unbiased (E[d̂] = d) | Unbiased with correct weights |
| Variance | Lower (no weighting) | Higher (weights amplify noise) |
| Known q required | No | Yes |

### Outputs

| File | Description |
|------|-------------|
| `results/appendix.json` | All trial results |
| `plots/appendix_spearman_{dataset}.png` | Individual Spearman plots |
| `plots/appendix_kendall_{dataset}.png` | Individual Kendall plots |
| `plots/appendix_top1_{dataset}.png` | Individual Top-1 plots |
| `plots/appendix_*_combined.png` | Combined 1x2 figures |

### Running

```bash
python paper_experiments/appendix/run.py
python paper_experiments/appendix/plot.py
```

### Expected Results

At low observation rates (5-30%), the IPW estimator should show better correlation with ground truth than the naive estimator, since it correctly accounts for the reduced sample size. At high observation rates (>70%), both estimators should perform similarly since the weights approach 1.
