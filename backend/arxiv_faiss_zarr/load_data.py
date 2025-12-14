# arxiv_faiss_zarr/load_data.py
import os
import json
from typing import List, Dict, Tuple

import pandas as pd

from .config import DEFAULT_NUM_DOCS


def load_arxiv_df(csv_path: str,
                  num_docs: int = DEFAULT_NUM_DOCS) -> pd.DataFrame:
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
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def load_metadata_jsonl(meta_path: str) -> List[Dict]:
    metadata: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata
