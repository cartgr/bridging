"""
Utilities for loading Pol.is comment text.
"""

import csv
import json
import re
from pathlib import Path


def load_dataset_mapping() -> dict:
    """Load mapping from PrefLib dataset names to openData directories."""
    mapping_path = Path(__file__).parent.parent / "data" / "polis_comments" / "dataset_mapping.json"
    with open(mapping_path) as f:
        return json.load(f)


def load_comment_id_mapping(dataset_name: str) -> dict[int, int]:
    """
    Load mapping from matrix row index to original comment ID.

    Parses the PrefLib .cat file to extract the ALTERNATIVE NAME mappings.

    Returns:
        Dict mapping matrix_index -> comment_id
    """
    import urllib.request

    url = f"https://raw.githubusercontent.com/PrefLib/PrefLib-Data/main/datasets/00069%20-%20polis/{dataset_name}.cat"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not fetch .cat file for {dataset_name}: {e}")
        return {}

    mapping = {}
    for line in content.split('\n'):
        # Match lines like "# ALTERNATIVE NAME 0: Comment #53"
        match = re.match(r'# ALTERNATIVE NAME (\d+): Comment #(\d+)', line)
        if match:
            matrix_idx = int(match.group(1))
            comment_id = int(match.group(2))
            mapping[matrix_idx] = comment_id

    return mapping


def load_comment_texts(dataset_name: str) -> dict[int, str]:
    """
    Load comment texts for a dataset.

    Args:
        dataset_name: PrefLib dataset name (e.g., "00069-00000001")

    Returns:
        Dict mapping comment_id -> comment_text
    """
    mapping = load_dataset_mapping()

    if dataset_name not in mapping:
        print(f"Warning: No openData mapping for {dataset_name}")
        return {}

    opendata_dir = mapping[dataset_name]
    comments_path = (
        Path(__file__).parent.parent / "data" / "polis_comments" /
        f"{opendata_dir}_comments.csv"
    )

    if not comments_path.exists():
        print(f"Warning: Comments file not found: {comments_path}")
        return {}

    texts = {}
    with open(comments_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                comment_id = int(row['comment-id'])
                text = row.get('comment-body', '').strip()
                texts[comment_id] = text
            except (ValueError, KeyError):
                continue

    return texts


def get_comment_text_by_index(
    dataset_name: str,
    matrix_index: int,
    id_mapping: dict[int, int] = None,
    texts: dict[int, str] = None,
) -> str:
    """
    Get comment text for a matrix row index.

    Args:
        dataset_name: PrefLib dataset name
        matrix_index: Row index in the vote matrix
        id_mapping: Optional pre-loaded index -> comment_id mapping
        texts: Optional pre-loaded comment_id -> text mapping

    Returns:
        Comment text or placeholder if not found
    """
    if id_mapping is None:
        id_mapping = load_comment_id_mapping(dataset_name)
    if texts is None:
        texts = load_comment_texts(dataset_name)

    comment_id = id_mapping.get(matrix_index)
    if comment_id is None:
        return f"[Comment #{matrix_index}]"

    text = texts.get(comment_id, f"[Comment #{comment_id}]")
    return text


def load_all_comments_for_dataset(dataset_name: str) -> dict[int, str]:
    """
    Load all comments for a dataset, indexed by matrix row.

    Args:
        dataset_name: PrefLib dataset name

    Returns:
        Dict mapping matrix_index -> comment_text
    """
    id_mapping = load_comment_id_mapping(dataset_name)
    texts = load_comment_texts(dataset_name)

    result = {}
    for matrix_idx, comment_id in id_mapping.items():
        result[matrix_idx] = texts.get(comment_id, f"[Comment #{comment_id}]")

    return result
