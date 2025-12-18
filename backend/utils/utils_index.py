import numpy as np
import json
from typing import List, Dict
def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    """
    Performs L2 normalization on vectors to prepare for Cosine Similarity search.
    
    In FAISS, standard dot-product indices (IP) become equivalent to Cosine Similarity
    only when input vectors are normalized to unit length.

    Args:
        x (np.ndarray): Input embedding batch (shape: [batch_size, dim]).
        axis (int): Axis along which to compute the norm.
        eps (float): Epsilon to prevent division by zero.

    Returns:
        np.ndarray: The normalized unit vectors.
    """
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)

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