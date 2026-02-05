"""
Process raw PrefLib voting data into standardized numpy matrices.

Matrix format:
- Shape: (n_items, n_voters)
- Values: 1.0 (approved), 0.0 (disapproved), 0.5 (pass/neutral), np.nan (unobserved)

Note: Only Polis datasets (00069) have 3 categories with pass votes.
      French election data (00026, 00071) remains binary (approve/disapprove only).

Usage:
    python data/process.py  # Process all downloaded datasets
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

# Base paths
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw" / "preflib"
PROCESSED_DIR = DATA_DIR / "processed" / "preflib"


def _parse_item_set(s: str) -> list[int]:
    """Parse an item specification from PrefLib .cat format.

    Handles formats like:
    - "6" -> [6]
    - "{1,2,3}" -> [1, 2, 3]
    - "{}" -> []
    """
    s = s.strip()
    if not s or s == "{}":
        return []
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [int(x.strip()) for x in inner.split(",")]
    return [int(s)]


def _split_cat_items(items_str: str) -> list[str]:
    """Split category items string on commas, respecting braces.

    Example: "{1,2}, {3,4}, 5" -> ["{1,2}", "{3,4}", "5"]
    """
    result = []
    depth = 0
    current = []
    for ch in items_str:
        if ch == '{':
            depth += 1
            current.append(ch)
        elif ch == '}':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append(''.join(current).strip())
    return result


def process_cat_file(cat_path: Path) -> tuple[np.ndarray, dict]:
    """Process a PrefLib .cat (categorical) file into a matrix.

    PrefLib categorical format:
    - Header lines starting with #
    - 2-category format: "count: yes_items, no_items"
    - 3-category format: "count: disapproved_items, neutral_items, approved_items"

    Conversion (2 categories - Yes/No):
    - Category 1 (Yes) -> 1.0 (approved)
    - Category 2 (No) -> 0.0 (disapproved)

    Conversion (3 categories - Disapproved/Neutral/Approved):
    - Category 0 (Disapproved) -> 0.0
    - Category 1 (Neutral/Pass) -> 0.5 (voter saw but chose not to vote)
    - Category 2 (Approved) -> 1.0

    Args:
        cat_path: Path to .cat file

    Returns:
        Tuple of (matrix, metadata)
        matrix: np.ndarray of shape (n_items, n_voters)
        metadata: dict with n_items, n_voters, sparsity, source
    """
    with open(cat_path, 'r') as f:
        lines = f.readlines()

    # Parse header to find number of alternatives, voters, and categories
    n_items = 0
    n_voters = 0
    n_categories = 2  # default
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("# NUMBER ALTERNATIVES:"):
            n_items = int(line.split(":", 1)[1].strip())
        elif line.startswith("# NUMBER VOTERS:"):
            n_voters = int(line.split(":", 1)[1].strip())
        elif line.startswith("# NUMBER CATEGORIES:"):
            n_categories = int(line.split(":", 1)[1].strip())
        elif not line.startswith("#") and line:
            data_start = i
            break

    if n_items == 0 or n_voters == 0:
        raise ValueError(f"Could not parse header of {cat_path}")

    # Initialize matrix with NaN
    matrix = np.full((n_items, n_voters), np.nan)

    # Parse data lines
    voter_idx = 0
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on first colon to get count and items
        if ":" not in line:
            continue

        count_str, items_str = line.split(":", 1)
        try:
            count = int(count_str.strip())
        except ValueError:
            continue

        # Split items by category
        cat_parts = _split_cat_items(items_str.strip())

        if n_categories == 2 and len(cat_parts) >= 2:
            # 2-category: Yes (1.0), No (0.0)
            yes_items = _parse_item_set(cat_parts[0])
            no_items = _parse_item_set(cat_parts[1])

            for v in range(voter_idx, min(voter_idx + count, n_voters)):
                for item_id in yes_items:
                    if 1 <= item_id <= n_items:
                        matrix[item_id - 1, v] = 1.0
                for item_id in no_items:
                    if 1 <= item_id <= n_items:
                        matrix[item_id - 1, v] = 0.0

        elif n_categories == 3 and len(cat_parts) >= 3:
            # 3-category: Disapproved (0.0), Neutral/Pass (0.5), Approved (1.0)
            disapproved_items = _parse_item_set(cat_parts[0])
            neutral_items = _parse_item_set(cat_parts[1])  # pass votes -> 0.5
            approved_items = _parse_item_set(cat_parts[2])

            for v in range(voter_idx, min(voter_idx + count, n_voters)):
                for item_id in disapproved_items:
                    if 0 <= item_id < n_items:  # 0-indexed for 3-cat
                        matrix[item_id, v] = 0.0
                for item_id in neutral_items:
                    if 0 <= item_id < n_items:  # 0-indexed for 3-cat
                        matrix[item_id, v] = 0.5
                for item_id in approved_items:
                    if 0 <= item_id < n_items:  # 0-indexed for 3-cat
                        matrix[item_id, v] = 1.0

        voter_idx += count

    sparsity = np.isnan(matrix).sum() / matrix.size

    metadata = {
        "n_items": n_items,
        "n_voters": n_voters,
        "sparsity": sparsity,
        "source": "preflib",
    }

    return matrix, metadata


def save_matrix(matrix: np.ndarray, metadata: dict, dest_path: Path) -> None:
    """Save matrix and metadata to .npz file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest_path, matrix=matrix, **metadata)


def process(datasets=None):
    """Process all downloaded PrefLib datasets.

    Args:
        datasets: List of dataset IDs. If None, processes all in raw/preflib/

    Returns:
        Dict mapping dataset IDs to success status
    """
    if datasets is None:
        if not RAW_DIR.exists():
            print("No raw data found. Run download.py first.")
            return {}
        datasets = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]

    results = {}
    print("Processing PrefLib datasets...")

    for dataset_id in tqdm(datasets, desc="PrefLib"):
        dataset_dir = RAW_DIR / dataset_id
        if not dataset_dir.exists():
            print(f"  Skipping {dataset_id}: directory not found")
            results[dataset_id] = False
            continue

        # Find .cat files
        cat_files = list(dataset_dir.glob("*.cat"))
        if not cat_files:
            print(f"  Skipping {dataset_id}: no .cat files found")
            results[dataset_id] = False
            continue

        # Process each .cat file
        any_success = False
        for cat_path in cat_files:
            try:
                matrix, metadata = process_cat_file(cat_path)
                dest_path = PROCESSED_DIR / f"{cat_path.stem}.npz"
                save_matrix(matrix, metadata, dest_path)
                any_success = True
            except Exception as e:
                print(f"  Error processing {cat_path.name}: {e}")

        results[dataset_id] = any_success

    return results


def main():
    results = process()

    # Print summary
    print("\nProcessing Summary:")
    success_count = sum(results.values())
    total_count = len(results)
    print(f"  {success_count}/{total_count} datasets processed")


if __name__ == "__main__":
    main()
