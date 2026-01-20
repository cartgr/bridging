"""
Download raw voting data from PrefLib.

Usage:
    python data/download.py          # Download all default datasets
    python data/download.py --force  # Re-download existing files
"""

import requests
from pathlib import Path
from tqdm import tqdm

# Base paths
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw" / "preflib"

# Default PrefLib datasets: mapping dataset_id -> (folder_name, file_count)
# folder_name is the name used in the PrefLib-Data GitHub repo
DEFAULT_DATASETS = {
    "00026": ("frenchapproval", 6),   # French Election 2002 Approval (6 districts)
    "00069": ("polis", 20),           # Pol.is conversations (20 polls)
}

# PrefLib GitHub raw content base URL
PREFLIB_GITHUB_BASE = "https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets"


def download_file(url: str, dest_path: Path, skip_existing: bool = True) -> bool:
    """Download a file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Local path to save file
        skip_existing: If True, skip download if file exists

    Returns:
        True if file was downloaded or already exists, False on error
    """
    if skip_existing and dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except requests.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download(
    datasets: dict[str, tuple[str, int]] | None = None,
    skip_existing: bool = True,
) -> dict[str, bool]:
    """Download PrefLib categorical datasets from GitHub.

    Args:
        datasets: Dict mapping dataset_id -> (folder_name, file_count).
                  If None, uses DEFAULT_DATASETS.
        skip_existing: If True, skip files that already exist.

    Returns:
        Dict mapping dataset IDs to success status
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS

    results = {}
    print("Downloading PrefLib datasets...")

    for dataset_id, (folder_name, file_count) in tqdm(datasets.items(), desc="PrefLib"):
        dest_dir = RAW_DIR / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # URL pattern: {base}/{dataset_id} - {folder_name}/{dataset_id}-{number}.cat
        folder_path = f"{dataset_id} - {folder_name}"

        any_success = False
        for i in range(1, file_count + 1):
            filename = f"{dataset_id}-{i:08d}.cat"
            url = f"{PREFLIB_GITHUB_BASE}/{folder_path}/{filename}"
            dest_path = dest_dir / filename

            if download_file(url, dest_path, skip_existing):
                any_success = True

        results[dataset_id] = any_success

        if not any_success:
            print(f"  Failed to download any files for {dataset_id}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download voting data from PrefLib")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    results = download(skip_existing=not args.force)

    # Print summary
    print("\nDownload Summary:")
    success_count = sum(results.values())
    total_count = len(results)
    print(f"  {success_count}/{total_count} datasets downloaded")


if __name__ == "__main__":
    main()
