# arxiv_faiss_zarr/embeddings.py
import time
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL_NAME


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


def load_model(device: str = "cpu") -> SentenceTransformer:
    """
    Initializes the SentenceTransformer model and moves it to the target device.

    Note:
        This function assumes the `device` argument has already been validated 
        (e.g., CUDA availability checked) by the caller context.

    Args:
        device (str): The target accelerator (e.g., 'cpu', 'cuda'). 
                      Must be a valid and available device string.

    Returns:
        SentenceTransformer: The model instance loaded on the specified device.
    """
    print(f"[INFO] Loading embedding model: {EMBED_MODEL_NAME} on {device}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    model = model.to(device)
    return model


def choose_device(use_gpu: bool) -> str:
    """
    Selects the optimal compute device with a graceful fallback mechanism.

    Verifies CUDA availability before assigning the device. If the user requests GPU 
    (`use_gpu=True`) but no CUDA device is detected, the function automatically 
    degrades to CPU to prevent runtime errors.

    Args:
        use_gpu (bool): Whether the user intends to utilize GPU acceleration.

    Returns:
        str: A valid device string ('cuda' or 'cpu') ready for `model.to()`.
    """
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
    Encodes text chunks into normalized vectors optimized for cosine similarity search.

    Key Steps:
    1. Batch inference using the SentenceTransformer model.
    2. Casts to float32 (required for FAISS standard indices).
    3. Applies L2 normalization so that Inner Product (IP) distance equals Cosine Similarity.

    Args:
        model (SentenceTransformer): The inference engine.
        chunk_texts (List[str]): List of context-aware passages (e.g., title + section label + body text).
        batch_size (int): Number of chunks to process in parallel.

    Returns:
        np.ndarray: Normalized embedding matrix of shape (num_chunks, embedding_dim).
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
