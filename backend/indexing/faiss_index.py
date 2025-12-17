# arxiv_faiss_zarr/faiss_index.py
import os
from typing import Tuple

import faiss
import numpy as np
import psutil
import zarr

from .config import CHUNK_SIZE, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH


def build_flat_index(emb_store: zarr.Array, dim: int) -> faiss.IndexFlatIP:
    """
    Constructs a brute-force FAISS index for exact nearest neighbor search.

    This function iterates through the on-disk Zarr array in chunks to minimize 
    RAM usage during index construction.

    Args:
        emb_store (zarr.Array): The Zarr array containing pre-computed embeddings.
        dim (int): Dimensionality of the vectors (e.g., 384).

    Returns:
        faiss.IndexFlatIP: A populated Flat index using Inner Product metric.
                           (Equivalent to Cosine Similarity if vectors are normalized).
    """
    print("[INFO] Building FAISS IndexFlatIP (flat, CPU)...")
    index = faiss.IndexFlatIP(dim)
    n = emb_store.shape[0]
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = emb_store[start:end][:]
        index.add(chunk)
    print(f"[INFO] Flat index total vectors = {index.ntotal}")
    return index


def build_hnsw_index(emb_store: zarr.Array, dim: int) -> faiss.IndexHNSWFlat:
    """
    Builds a Hierarchical Navigable Small World (HNSW) index for fast approximate search.

    Algorithm Details:
    - Uses HNSW graph structure to achieve logarithmic time complexity O(log N).
    - Configured with `M={HNSW_M}` and `efConstruction={HNSW_EF_CONSTRUCTION}` from config.
    - Loads data incrementally from Zarr to handle large datasets efficiently.

    Args:
        emb_store (zarr.Array): Source embeddings on disk.
        dim (int): Vector dimension.

    Returns:
        faiss.IndexHNSWFlat: The optimized HNSW index ready for high-speed retrieval.
    """
    print("[INFO] Building FAISS HNSW index...")
    index_hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index_hnsw.hnsw.efSearch = HNSW_EF_SEARCH

    n = emb_store.shape[0]
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = emb_store[start:end][:]
        index_hnsw.add(chunk)

    print(f"[INFO] HNSW index total vectors = {index_hnsw.ntotal}")
    return index_hnsw

'''
def search_flat(index: faiss.IndexFlatIP,
                query_vec: np.ndarray,
                top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Executes exact search while capturing latency and memory telemetry.

    This function is instrumented with `psutil` to detect memory spikes (RSS delta)
    during the search operation, providing data for performance benchmarking.

    Args:
        index (faiss.IndexFlatIP): The baseline exact search index.
        query_vec (np.ndarray): Input query vector.
        top_k (int): Top-K results.

    Returns:
        Tuple containing:
        1. Result Scores (cosine similarity if normalized)
        2. Result Indices (row indices into the embedding / metadata arrays)
        3. Latency (seconds)
        4. Memory Footprint (MB)
        5. Memory Spike/Delta (MB)
    """
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    import time
    t0 = time.perf_counter()
    q = query_vec.reshape(1, -1)
    scores, idx = index.search(q, top_k)
    elapsed = time.perf_counter() - t0

    rss_after = proc.memory_info().rss
    rss_mb = rss_after / 1e6
    delta_mb = (rss_after - rss_before) / 1e6

    return scores[0], idx[0].astype(int), elapsed, rss_mb, delta_mb


def search_baseline(emb_store: zarr.Array,
                    query_vec: np.ndarray,
                    top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Executes a naive NumPy-based exact search to serve as a performance baseline.

    ⚠️ Engineering Note:
    This implementation forces a full data load (`emb_store[:]`) into RAM, causing 
    significant memory spikes. It is used solely for benchmarking to demonstrate 
    the necessity of vector databases (like FAISS) for handling large-scale datasets.

    Args:
        emb_store (zarr.Array): Source embeddings (will be fully loaded into RAM).
        query_vec (np.ndarray): Query vector.
        top_k (int): Top-K count.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, float, float]: 
            Returns search results and telemetry data (latency, memory usage).
    """
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    import time
    t0 = time.perf_counter()
    emb = emb_store[:]
    load_time = time.perf_counter() - t0
    print(f"[baseline] loaded embeddings in {load_time:.3f}s")

    t1 = time.perf_counter()
    scores_all = emb @ query_vec
    idx = np.argpartition(-scores_all, top_k)[:top_k]
    idx = idx[np.argsort(-scores_all[idx])]
    scores = scores_all[idx]
    elapsed = time.perf_counter() - t1
    print(f"[baseline] similarity computed in {elapsed:.3f}s")

    rss_after = proc.memory_info().rss
    rss_mb = rss_after / 1e6
    delta_mb = (rss_after - rss_before) / 1e6

    return scores, idx.astype(int), elapsed, rss_mb, delta_mb
'''