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

# Default PrefLib datasets: mapping dataset_id -> (folder_name, file_range)
# folder_name is the name used in the PrefLib-Data GitHub repo
# file_range is (start, end) inclusive for the file numbers
DEFAULT_DATASETS = {
    "00026": ("frenchapproval", (1, 6)),       # French Election 2002 Approval (6 districts)
    "00033": ("poster", (2, 3)),               # San Sebastian Poster Competition (.cat files 2-3)
    "00063": ("ctu", (1, 1)),                  # CTU AG1 Tutorial Time Selection (1 file)
    "00069": ("polis", (1, 20)),               # Pol.is conversations (20 polls)
    "00071": ("french-approval-2007", (1, 12)),# French Presidential Election 2007 (12 files)
}
# Note: 00027 (frenchrate) and 00029 (frenchratecombi) only have .toc/.dat files, not .cat

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
    datasets=None,
    skip_existing: bool = True,
):
    """Download PrefLib categorical datasets from GitHub.

    Args:
        datasets: Dict mapping dataset_id -> (folder_name, file_range).
                  file_range is (start, end) inclusive.
                  If None, uses DEFAULT_DATASETS.
        skip_existing: If True, skip files that already exist.

    Returns:
        Dict mapping dataset IDs to success status
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS

    results = {}
    print("Downloading PrefLib datasets...")

    for dataset_id, (folder_name, file_range) in tqdm(datasets.items(), desc="PrefLib"):
        dest_dir = RAW_DIR / dataset_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        # URL pattern: {base}/{dataset_id} - {folder_name}/{dataset_id}-{number}.cat
        folder_path = f"{dataset_id} - {folder_name}"

        start, end = file_range
        any_success = False
        for i in range(start, end + 1):
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
