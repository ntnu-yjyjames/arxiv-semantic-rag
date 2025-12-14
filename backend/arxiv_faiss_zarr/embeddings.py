# arxiv_faiss_zarr/embeddings.py
import time
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def load_model(device: str = "cpu") -> SentenceTransformer:
    print(f"[INFO] Loading embedding model: {EMBED_MODEL_NAME} on {device}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model = model.to(device)
    return model


def choose_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {name}")
        return "cuda"
    if use_gpu:
        print("[WARN] use_gpu=True but CUDA not available; fallback to CPU.")
    else:
        print("[INFO] Using CPU.")
    return "cpu"


def embed_chunks(model: SentenceTransformer,
                 chunk_texts: List[str],
                 batch_size: int = 256) -> np.ndarray:
    """
    對所有 chunk 做 embedding，回傳 (N, dim) 的 numpy array（已經 L2-normalize）。
    """
    print("[INFO] Start embedding all chunks...")
    t0 = time.perf_counter()
    emb = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype(np.float32)
    emb = l2_normalize(emb, axis=1)
    elapsed = time.perf_counter() - t0
    print(f"[INFO] Finished embedding {len(chunk_texts)} chunks in {elapsed:.2f} s")
    return emb
