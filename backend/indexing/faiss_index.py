# arxiv_faiss_zarr/faiss_index.py
import os
from typing import Tuple

import faiss
import numpy as np
import psutil
import zarr

from .config import CHUNK_SIZE, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH


def build_flat_index(emb_store: zarr.Array, dim: int) -> faiss.IndexFlatIP:
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


def search_flat(index: faiss.IndexFlatIP,
                query_vec: np.ndarray,
                top_k: int = 10) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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
