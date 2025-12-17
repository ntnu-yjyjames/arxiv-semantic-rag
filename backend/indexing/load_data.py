# arxiv_faiss_zarr/load_data.py
import os
import json
from typing import List, Dict, Tuple

import pandas as pd

from .config import DEFAULT_NUM_DOCS


def load_arxiv_df(csv_path: str,
                  num_docs: int = DEFAULT_NUM_DOCS) -> pd.DataFrame:
    """
    Loads and validates the arXiv dataset from a CSV file.

    Performs basic preprocessing:
    1. Checks file existence and required columns ('title', 'abstract').
    2. Handles missing values (NaN) in text columns by filling with empty strings.
    3. Supports partial loading via `num_docs` for rapid prototyping.

    Args:
        csv_path (str): Path to the source CSV file.
        num_docs (int): Number of rows to read. If <= 0, reads the entire dataset.

    Returns:
        pd.DataFrame: A sanitized DataFrame containing at least 'title' and 'abstract'.

    Raises:
        FileNotFoundError: If `csv_path` does not exist.
        ValueError: If the CSV schema is missing required columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading metadata from {csv_path} (num_docs={num_docs})")
    if num_docs and num_docs > 0:
        df = pd.read_csv(csv_path, nrows=num_docs)
    else:
        df = pd.read_csv(csv_path)

    if "title" not in df.columns or "abstract" not in df.columns:
        raise ValueError("CSV must contain at least 'title' and 'abstract' columns.")

    df["title"] = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    print(f"[INFO] Loaded {len(df)} rows from CSV.")
    return df


def save_metadata_jsonl(metadata: List[Dict],
                        meta_path: str):
    """
    Serializes metadata to a JSON Lines (JSONL) file.
    
    Format: Each line in the file is a valid, independent JSON object.
    Advantage: JSONL allows for efficient stream processing and line-by-line reading
    without parsing the entire file into memory.

    Args:
        metadata (List[Dict]): A list of dictionaries containing chunk metadata.
        meta_path (str): Output file path.
    """
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def load_metadata_jsonl(meta_path: str) -> List[Dict]:
    """
    Parses a JSON Lines file into a list of dictionaries.

    This function reads the metadata file line-by-line and reconstructs the
    list of per-chunk metadata objects used when mapping search results
    (vector row indices) back to human-readable information (paper ID, title, section, etc.).

    Args:
        meta_path (str): Path to the .jsonl file.

    Returns:
        List[Dict]: The restored metadata list.
    """
    metadata: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata
