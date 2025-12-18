# search_backend.py
import json
import os
import time
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Any , Optional, Set

import faiss
import numpy as np
import psutil
import zarr


from sentence_transformers import SentenceTransformer
from indexing.config import EMBED_MODEL_NAME

# Use your own data path(with FAISS/ZARR index)
#INDEX_DIR = "./arxiv_zarr/arxiv_index_full" 
INDEX_DIR = "./arxiv_zarr/arxiv_index_demo_transformer"  

# Aligned with building parameters
HNSW_EF_SEARCH = 64


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    L2-normalize an array along the last dimension.

    Args:
        x (np.ndarray): Input array of shape (..., dim).
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Same shape as `x`, with each vector on the last axis normalized
                    to unit length (||v||_2 ~= 1).
    """
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _load_metadata(meta_path: str) -> List[Dict[str, Any]]:
    """
    Load chunk-level metadata from a JSON Lines (.jsonl) file.

    Each line in the file is expected to be a standalone JSON object describing
    one chunk (doc_idx, title, section, chunk_index, previews, etc.).

    Args:
        meta_path (str): Path to the metadata.jsonl file.

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries in file order
                              (index-aligned with the embedding matrix).
    """
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

import re

def _normalize_title(text: str) -> str:
    """
    Normalize a title string for matching:
    - collapse all whitespace (spaces, tabs, newlines) into single spaces
    - strip leading/trailing whitespace
    - lowercase for case-insensitive matching
    """
    if not isinstance(text, str):
        return ""
    # 將 \n、\t、多個空白 等全部壓成一個空白
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip().lower()



class IndexResources:
    """
    Centralized resource manager acting as the data access layer for the RAG pipeline.

    This class orchestrates the lifecycle of high-memory artifacts to ensure efficient 
    retrieval. It serves as a **Singleton** (managed by the app state) to prevent 
    redundant reloading of heavy indices.

    Core Responsibilities:
    1. **Storage Abstraction**: Manages pointers to on-disk Zarr arrays (Embeddings) and in-memory FAISS indices.
    2. **Metadata Indexing**: Builds O(1) lookup tables (Reverse Indices) to map Documents <-> Chunks.
    3. **Query Engine**: Holds the shared SentenceTransformer model for real-time query encoding.

    Attributes:
        emb_store (zarr.Array): Memory-mapped interface to the Zarr embedding store on disk.
        faiss_index_flat (faiss.IndexFlatIP): The baseline exact search index.
        faiss_index_hnsw (Optional[faiss.IndexHNSWFlat]): The optimized approximate index (if available).
        metadata (List[Dict]): Full list of chunk-level metadata (Titles, Sections).
        doc2rows (Dict[int, List[int]]): Reverse index mapping internal doc_idx -> list of chunk row indices.
        title2docs (Dict[str, List[int]]): Inverted index for fast title-based document lookup.
    """


    def __init__(self, index_dir: str):
        """
        Initializes the resource pool by validating paths and hydrating lookup tables.

        Performs a linear scan of metadata to build the `doc2rows` and `title2docs` 
        maps, enabling fast aggregation for recommendation tasks.
        doc_titles (Dict[int, str]): Canonical title string for each internal doc_idx.

        Args:
            index_dir (str): Directory path containing the build artifacts 
                             (`embeddings.zarr`, `faiss_*.bin`, `metadata.jsonl`).

        Raises:
            FileNotFoundError: If critical artifacts (Zarr, Flat Index, Metadata) are missing.
        """
        zarr_path = os.path.join(index_dir, "embeddings.zarr")
        faiss_flat_path = os.path.join(index_dir, "faiss_index_flat.bin")
        faiss_hnsw_path = os.path.join(index_dir, "faiss_index_hnsw.bin")
        meta_path = os.path.join(index_dir, "metadata.jsonl")

        if not os.path.exists(zarr_path):
            raise FileNotFoundError(zarr_path)
        if not os.path.exists(faiss_flat_path):
            raise FileNotFoundError(faiss_flat_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)

        # Zarr
        root = zarr.open(zarr_path, mode="r")
        self.emb_store = root["embeddings"]  # shape = (N_chunks, dim)

        # FAISS flat
        self.faiss_index_flat = faiss.read_index(faiss_flat_path)

        # FAISS HNSW (optional)
        if os.path.exists(faiss_hnsw_path):
            self.faiss_index_hnsw = faiss.read_index(faiss_hnsw_path)
        else:
            self.faiss_index_hnsw = None

        # metadata（chunk-level）
        self.metadata: List[Dict[str, Any]] = _load_metadata(meta_path)

        # doc_idx -> chunk rows
        self.doc2rows: Dict[int, List[int]] = defaultdict(list)
        # doc_idx -> title
        self.doc_titles: Dict[int, str] = {}
        # title(lower) -> doc_idx list
        self.title2docs: Dict[str, List[int]] = defaultdict(list)
        
        # 用於 baseline mode="hot" 的 in-memory embeddings
        self._emb_hot: np.ndarray | None = None


        for row_idx, m in enumerate(self.metadata):
            doc_idx = int(m["doc_idx"])
            self.doc2rows[doc_idx].append(row_idx)

            title = m.get("title", "")
            if doc_idx not in self.doc_titles:
                # 保留原始 title 給前端顯示用（不要 normalize）
                self.doc_titles[doc_idx] = title

            key = _normalize_title(title)
            if key:
                self.title2docs[key].append(doc_idx)

        print(f"[IndexResources] Loaded {len(self.metadata)} chunks, {len(self.doc2rows)} docs.")
        if self.faiss_index_hnsw is None:
            print("[IndexResources] HNSW index not found; only flat + baseline available.")
        # 新增：載入同一個 embedding model，給「查詢問題」用
        print(f"[IndexResources] Loading embedding model for queries: {EMBED_MODEL_NAME}")
        self.query_model = SentenceTransformer(EMBED_MODEL_NAME)


    # -------- Doc / user 向量 --------

    def compute_doc_embedding(self, doc_idx: int) -> np.ndarray:
        """
        Aggregates chunk-level embeddings into a single document-level vector.

        Strategy: **Mean Pooling**. 
        This function fetches all embedding vectors associated with the given document ID 
        from the Zarr store and computes their centroid (average). The result is 
        L2-normalized to represent the document's overall semantic theme.

        Args:
            doc_idx (int): Internal document index (0-based) used within this index.

        Returns:
            np.ndarray: A normalized float32 vector of shape (dim,).

        Raises:
            ValueError: If the document has no associated chunks (data consistency error).
        """
        rows = self.doc2rows.get(doc_idx)
        if not rows:
            raise ValueError(f"doc_idx {doc_idx} has no chunks")
        embs = self.emb_store[rows, :]  # (n_chunks_d, dim)
        v = embs.mean(axis=0)
        v = l2_normalize(v[None, :])[0].astype("float32")
        return v

    def compute_user_vector_from_titles(self, titles: List[str]) -> np.ndarray:
        """
        Synthesizes a 'User Interest Vector' from a list of preferred paper titles.

        This acts as the query encoder for both search and recommendation: 
        it constructs a profile vector by averaging document embeddings for the selected titles.
        
        Process:
        1. Resolution: Maps text titles to internal Document IDs.
        2. Aggregation: Computes document vectors for each ID.
        3. Profiling: Calculates the global mean (Centroid) of these document vectors.

        Args:
            titles (List[str]): List of paper titles provided by the user.
                                Matching is case-insensitive and tolerant to whitespace differences
                                (line breaks / multiple spaces are collapsed).

        Returns: 
            np.ndarray: The normalized user query vector ready for vector search.

        Raises:
            ValueError: If none of the provided titles can be found in the index.
        """
        doc_indices: List[int] = []
        for t in titles:
            key = _normalize_title(t)
            if not key:
                continue
            doc_indices.extend(self.title2docs.get(key, []))
        doc_indices = sorted(set(doc_indices))
        
        if not doc_indices:
            print(doc_indices)
            raise ValueError("No documents found for given titles.")

        doc_vecs = []
        for d in doc_indices:
            try:
                v = self.compute_doc_embedding(d)
                doc_vecs.append(v)
            except Exception as e:
                print(f"[WARN] compute_doc_embedding failed for doc {d}: {e}")
                continue

        if not doc_vecs:
            raise ValueError("No valid doc vectors computed.")

        u = np.mean(np.stack(doc_vecs, axis=0), axis=0)
        u = l2_normalize(u[None, :])[0].astype("float32")
        return u

    # -------- 搜尋：FAISS flat / HNSW / baseline --------

    def _measure_memory(self):
        """
        Captures the current Resident Set Size (RSS) of the process.

        Used to calculate the memory overhead (delta) of a specific search operation.
        
        Returns:
            int: Memory usage in bytes.
        """
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss

    def search_faiss_flat(self, user_vec: np.ndarray, top_k: int = 10):
        """
        Executes an exact nearest neighbor search using FAISS's optimized C++ backend.

        Unlike the naive NumPy baseline, this uses `IndexFlatIP` (Inner Product), which 
        is faster on CPU due to BLAS/MKL optimizations but still performs an exhaustive 
        scan (O(N)).

        Args:
            user_vec (np.ndarray): The normalized query vector (shape: [dim]).
            top_k (int): Number of results to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, float, float]:
                - scores: Cosine similarity scores (assuming embeddings and query are L2-normalized).
                - indices: Vector row indices (into the embedding/metadata arrays).
                - elapsed: Latency in seconds.
                - rss_mb: Total memory usage after search.
                - delta_mb: Transient memory spike during search.
        """
        rss_before = self._measure_memory()
        start = time.perf_counter()

        q = user_vec.reshape(1, -1)
        scores, idx = self.faiss_index_flat.search(q, top_k)

        elapsed = time.perf_counter() - start
        rss_after = self._measure_memory()
        rss_mb = rss_after / 1e6
        delta_mb = (rss_after - rss_before) / 1e6

        return scores[0], idx[0].astype(int), elapsed, rss_mb, delta_mb

    def search_hnsw(self,
                    query_vec: np.ndarray,
                    top_k: int = 10,
                    ef_search: int | None = None):
        """
        Executes an Approximate Nearest Neighbor (ANN) search using the HNSW graph.

        This method allows dynamic tuning of the `efSearch` parameter at runtime,
        enabling a trade-off between recall and latency without rebuilding the index.

        Args:
            query_vec (np.ndarray): L2-normalized query vector of shape (dim,).
            top_k (int): Number of neighbors to return.
            ef_search (int | None): 
                Search depth. Higher values increase recall at the cost of latency.
                If None, uses the global default (HNSW_EF_SEARCH).

        Returns:
            Tuple[np.ndarray, np.ndarray, float, float, float]:
                - scores: Similarity scores (cosine similarity if embeddings are normalized).
                - indices: Vector row indices (used to look up chunk metadata).
                - elapsed: Search latency in seconds.
                - rss_mb: Process RSS after the search (MB).
                - delta_mb: RSS increase during the search (MB).

        Raises:
            RuntimeError: If the HNSW index was not built/loaded (e.g., in `no-hnsw` mode).
        """
        if self.faiss_index_hnsw is None:
            raise RuntimeError("HNSW index is not available. Did you build it?")

        if ef_search is None:
            ef_search = HNSW_EF_SEARCH
        ef_search = int(ef_search)

        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss

        start = time.perf_counter()
        self.faiss_index_hnsw.hnsw.efSearch = ef_search
        q = query_vec.reshape(1, -1)
        scores, idx = self.faiss_index_hnsw.search(q, top_k)
        elapsed = time.perf_counter() - start

        rss_after = proc.memory_info().rss
        rss_mb = rss_after / 1e6
        delta_mb = (rss_after - rss_before) / 1e6

        return scores[0], idx[0].astype(int), elapsed, rss_mb, delta_mb
    
    def search_baseline(self,
                    query_vec: np.ndarray,
                    top_k: int = 5,
                    mode: str = "cold"):
        """
        NumPy full-scan baseline used for comparison against FAISS Flat / HNSW.

        This method performs an exact search by multiplying the entire embedding
        matrix by the query vector and selecting the top-k scores.

        Two modes are supported:

        - mode = "cold":
            - On every call, load embeddings from the Zarr store (`self.emb_store[:]`),
            then perform a full-scan.
            - Simulates an end-to-end, disk-based pipeline (serverless / memory-constrained).

        - mode = "hot":
            - On first use, load embeddings into `self._emb_hot` once and reuse them
            on subsequent calls.
            - Simulates an in-memory NumPy baseline (all embeddings resident in RAM).

        The returned latency always covers the full operation for the chosen mode
        (i.e., load + compute in "cold", compute-only in "hot").

        Args:
            query_vec (np.ndarray): L2-normalized query vector of shape (dim,).
            top_k (int): Number of nearest neighbors to return.
            mode (str): Either "cold" or "hot", as described above.

        Returns:
            Tuple[np.ndarray, np.ndarray, float, float, float]:
                - scores_top: top-k similarity scores (cosine similarity if embeddings are normalized).
                - idx:        top-k row indices into the embedding/metadata arrays.
                - elapsed:    total latency in seconds for this call (including load in "cold").
                - rss_mb:     process RSS after the search (MB).
                - delta_mb:   RSS increase during the search (MB).
        """
        proc = psutil.Process(os.getpid())
        rss_before = proc.memory_info().rss

        start_total = time.perf_counter()

        # 1) 取出 embeddings
        if mode == "cold":
            emb = self.emb_store[:]       # 每次都從磁碟讀
        elif mode == "hot":
            if self._emb_hot is None:
                # 第一次載入到記憶體，之後重用
                self._emb_hot = self.emb_store[:]
            emb = self._emb_hot
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

        # 2) full-scan 內積 + top-k
        N = emb.shape[0]
        if N == 0:
            raise RuntimeError("No embeddings in baseline search.")

        top_k_eff = min(top_k, N)
        scores_all = emb @ query_vec

        if top_k_eff == N:
            idx = np.argsort(-scores_all)[:top_k_eff]
        else:
            kth = top_k_eff - 1
            idx = np.argpartition(-scores_all, kth)[:top_k_eff]
            idx = idx[np.argsort(-scores_all[idx])]

        scores_top = scores_all[idx]

        elapsed_total = time.perf_counter() - start_total

        rss_after = proc.memory_info().rss
        rss_mb = rss_after / 1e6
        delta_mb = (rss_after - rss_before) / 1e6

        return scores_top, idx.astype(int), elapsed_total, rss_mb, delta_mb


    # -------- 結果格式化與 recall --------

    def format_results(self, scores: np.ndarray,
                       indices: np.ndarray) -> List[Dict[str, Any]]:
        """
        Hydrates raw search results with metadata for frontend/API display.

        This function maps low-level vector row indices (as returned by FAISS or the
        baseline search) back to their corresponding chunk-level metadata. It also injects
        ranking information (`rank`) and normalizes types (e.g., np.float32 -> float)
        for JSON serialization.

        Args:
            scores (np.ndarray): Similarity scores from the search engine.
            indices (np.ndarray): Corresponding chunk row indices into `self.metadata`.

        Returns:
            List[Dict[str, Any]]: A list of result objects with fields such as:
                - rank, score
                - chunk_row, doc_idx
                - title, section, chunk_index, chunk_text
        """
        results = []
        for rank, (s, i) in enumerate(zip(scores, indices), start=1):
            m = self.metadata[int(i)]
            results.append({
                "rank": rank,
                "score": float(s),
                "chunk_row": int(i),
                "doc_idx": int(m.get("doc_idx", -1)),
                "title": m.get("title", ""),
                "section": m.get("section", ""),
                "chunk_index": int(m.get("chunk_index", 0)),
                "chunk_text": m.get("chunk_text", m.get("text_preview", "")),
            })
        return results

    def list_some_titles(self, limit: int = 50) -> List[str]:
        """
        Retrieves a sample of document titles for UI population.

        Useful for populating "Example Queries" or dropdown menus in the frontend 
        to help users overcome the "Cold Start" problem (not knowing what to search).
        
        Args:
            limit (int): Maximum number of titles to return.

        Returns:
            List[str]: A list of unique paper titles.
        """
        titles = []
        for d, t in self.doc_titles.items():
            titles.append(t)
            if len(titles) >= limit:
                break
        return titles
    


    @staticmethod
    def compute_recall_at_k(ground_idx: np.ndarray,
                            approx_idx: np.ndarray) -> float:
        """
        Calculates a simple Recall@K metric given two sets of result indices.

        Definition:
            Recall@K = |Intersection(Ground Truth, Approximation)| / |Ground Truth|

        In the benchmarking code, `ground_idx` is typically taken from an exact
        baseline search (e.g., NumPy full-scan or FAISS Flat), and `approx_idx`
        from an approximate or alternative backend (e.g., HNSW).

        Args:
            ground_idx (np.ndarray): Array of indices from the ground truth backend.
            approx_idx (np.ndarray): Array of indices from the candidate backend.

        Returns:
            float: A score between 0.0 and 1.0 (1.0 = perfect match).
        """
        gset = set(int(i) for i in ground_idx)
        aset = set(int(i) for i in approx_idx)
        if not gset:
            return 0.0
        return len(gset & aset) / float(len(gset))

    def benchmark_backends(self, user_vec: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Runs a comparative benchmark across all available search backends.

        This method acts as an experiment runner, executing the same query against:
        1. **NumPy Baseline** (in-memory full-scan, used as ground truth).
        2. **FAISS Flat** (CPU exact search).
        3. **FAISS HNSW** (approximate ANN search, if available).

        It captures key metrics (latency, RSS, memory delta, Recall@k) to visualize
        the trade-offs between accuracy and performance in the frontend.

        Args:
            user_vec (np.ndarray): The query vector (typically a user interest vector).
            top_k (int): Number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of structured metric reports, one per backend.
        """
        backends: List[Dict[str, Any]] = []

        # 1) baseline 當 ground truth
        base_scores, base_idx, base_elapsed, base_rss, base_delta = self.search_baseline(user_vec, top_k,mode="hot")
        base_results = self.format_results(base_scores, base_idx)
        backends.append({
            "backend": "baseline",
            "elapsed_ms": base_elapsed * 1000,
            "memory_rss_mb": base_rss,
            "memory_delta_mb": base_delta,
            "recall_at_k": 1.0,
            "results": base_results,
        })
        gt_idx = base_idx

        # 2) FAISS flat
        flat_scores, flat_idx, flat_elapsed, flat_rss, flat_delta = self.search_faiss_flat(user_vec, top_k)
        recall_flat = self.compute_recall_at_k(gt_idx, flat_idx)
        flat_results = self.format_results(flat_scores, flat_idx)
        backends.append({
            "backend": "faiss_flat",
            "elapsed_ms": flat_elapsed * 1000,
            "memory_rss_mb": flat_rss,
            "memory_delta_mb": flat_delta,
            "recall_at_k": recall_flat,
            "results": flat_results,
        })

        # 3) HNSW (若有 index)
        if self.faiss_index_hnsw is not None:
            hnsw_scores, hnsw_idx, hnsw_elapsed, hnsw_rss, hnsw_delta = self.search_hnsw(user_vec, top_k)
            recall_hnsw = self.compute_recall_at_k(gt_idx, hnsw_idx)
            hnsw_results = self.format_results(hnsw_scores, hnsw_idx)
            backends.append({
                "backend": "faiss_hnsw",
                "elapsed_ms": hnsw_elapsed * 1000,
                "memory_rss_mb": hnsw_rss,
                "memory_delta_mb": hnsw_delta,
                "recall_at_k": recall_hnsw,
                "results": hnsw_results,
            })

        return backends
    # --------  For RAG  --------
    def encode_query(self, text: str) -> np.ndarray:
        """
        Encodes a natural language query into a normalized dense vector.

        This method ensures the query vector resides in the same latent space as the 
        indexed chunks. It applies **L2 Normalization** so that the subsequent 
        Inner Product search is mathematically equivalent to Cosine Similarity.

        Args:
            text (str): The user's input question.

        Returns:
            np.ndarray: A float32 unit vector of shape (embedding_dim,).
        """
        emb = self.query_model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)
        # index 內向量已經 L2-normalize 過，所以 query 也要 normalize
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norm
        return emb[0]
    
    def search_chunks_for_question(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs semantic search to retrieve the most relevant text chunks for a question.

        Pipeline:
        1. Encode question -> Vector.
        2. Vector Search -> Top-K raw indices (using Flat index for precision).
        3. Hydration -> Metadata enrichment (IDs to Text).

        Args:
            question (str): The user's query.
            top_k (int): Number of context chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Hydrated search results sorted by relevance score.
        """
        q_vec = self.encode_query(question)
        scores, idx, _, _, _ = self.search_faiss_flat(q_vec, top_k=top_k)
        return self.format_results(scores, idx)
    
    def aggregate_doc_scores(self, scores, idx, exclude_docs=None, top_k_docs=10):
        """
        Aggregates chunk-level search results into document-level recommendations.

        Aggregation Strategy: **Max Pooling**.
        The relevance score of a document is defined as the maximum score of its 
        constituent chunks found in the search results. This assumes that if *any* part of a paper is highly relevant, the whole paper is worth recommending.

        Args:
            scores (Sequence[float] | np.ndarray): Similarity scores from vector search.
            idx (Sequence[int] | np.ndarray): Corresponding chunk row indices.
            exclude_docs (set[int] | None): Doc indices to filter out (e.g., seed papers).
            top_k_docs (int): Number of unique documents to return.

        Returns:
            List[Dict[str, Any]]: Ranked list of unique documents with:
                - doc_idx: internal document index
                - score: best (max) chunk score for that doc
                - title: document title
        """
        if exclude_docs is None:
            exclude_docs = set()

        doc_best = {}  # doc_idx -> best_score

        for s, i in zip(scores, idx):
            m = self.metadata[int(i)]
            d = int(m.get("doc_idx", -1))
            if d in exclude_docs:
                continue
            if d not in doc_best or s > doc_best[d]:
                doc_best[d] = float(s)

        # 排序取前 K
        items = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)[:top_k_docs]

        results = []
        for d, sc in items:
            title = self.doc_titles.get(d, "")
            results.append({
               "doc_idx": d,
                "score": sc,
                "title": title,
            })
        return results
    def recommend_docs_from_titles(self,
                                   titles,
                                   backend: str = "faiss_flat",
                                   top_k_docs: int = 10,
                                   candidate_chunks: int = 1000):
        """
        Generates paper recommendations based on a list of seed papers (User History).

        This implements an **Item-to-Item Discovery** pipeline:
        1. **Profile Construction**: Computes a centroid vector from the input titles.
        2. **Candidate Generation**: Retrieves a broad set of relevant chunks (e.g., 1000) 
           to ensure sufficient coverage for document-level scoring.
        3. **Filtering**: Explicitly excludes the input seed papers to promote discovery of *new* content.
        4. **Aggregation**: Scores and ranks the remaining documents.

        Args:
            titles (List[str]): List of papers the user already likes (The "Seed").
            backend (str): Search engine to use ('faiss_hnsw' for speed, 'faiss_flat' for precision).
            top_k_docs (int): Final number of unique papers to return.
            candidate_chunks (int): Number of raw chunks to fetch before aggregation. 
                                    Set higher (e.g., 1000) to improve recall.

        Returns:
        List[Dict[str, Any]]: Ranked list of recommended papers with:
            - doc_idx: internal document index
            - title: paper title
            - score: aggregated relevance score (max over chunk scores)

        Raises:
            ValueError: If the input titles cannot be mapped to any known documents.
        """
        # 1) titles -> doc_idx
        doc_indices = []
        for t in titles:
            key = t.strip().lower()
            if not key:
                continue
            doc_indices.extend(self.title2docs.get(key, []))
        doc_indices = sorted(set(doc_indices))
        exclude_docs = set(doc_indices)

        if not doc_indices:
            raise ValueError("No valid documents found for given titles.")

        # 2) 用這些 doc 建 user vector（你原本的函式）
        user_vec = self.compute_user_vector_from_titles(titles)

        # 3) 用指定 backend 搜尋 candidate_chunks 個 chunk
        scores, idx = self.search_chunks_with_backend(
            query_vec=user_vec,
            backend=backend,
            top_k=candidate_chunks,
        )

        # 4) chunk → doc，並排除 seed docs
        doc_results = self.aggregate_doc_scores(
            scores, idx,
            exclude_docs=exclude_docs,
            top_k_docs=top_k_docs,
        )
        return doc_results
    
    def search_chunks_with_backend(self,
                               query_vec: np.ndarray,
                               backend: str,
                               top_k: int,
                               baseline_mode: str = "cold",
                               ef_search: int | None = None):
        """
        Unified dispatcher for executing vector searches across different backends.

        Acts as a small facade that abstracts away the specific implementation details
        (FAISS Flat vs HNSW vs NumPy baseline) and returns a normalized (scores, indices)
        tuple for downstream use.

        Args:
            query_vec (np.ndarray): L2-normalized query vector of shape (dim,).
            backend (str): Strategy selector ('faiss_flat', 'faiss_hnsw', 'baseline').
            top_k (int): Number of nearest neighbors to retrieve (automatically clipped
                        to the total number of chunks).
            baseline_mode (str): Baseline mode: 'cold' (disk I/O each call) or
                                'hot' (in-memory full-scan), used when backend='baseline'.
            ef_search (int | None): HNSW search depth hyperparameter. If None, uses
                                    the global default (HNSW_EF_SEARCH).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - scores: similarity scores.
                - indices: vector row indices (into the embedding/metadata arrays).

        Raises:
            RuntimeError: If the index contains no chunks.
            ValueError: If an unknown backend string is provided.
        """

        total_chunks = len(self.metadata)
        if total_chunks <= 0:
            raise RuntimeError("No chunks in index.")
        if top_k > total_chunks:
            top_k = total_chunks

        if backend == "faiss_flat":
            out = self.search_faiss_flat(query_vec, top_k=top_k)
        elif backend == "faiss_hnsw":
            out = self.search_hnsw(query_vec, top_k=top_k, ef_search=ef_search)
        elif backend == "baseline":
            out = self.search_baseline(query_vec, top_k=top_k, mode=baseline_mode)
        else:
            raise ValueError(f"Unknown backend for recommend: {backend}")

        scores, idx, *rest = out
        return scores, idx


    def aggregate_doc_scores(self,
                             scores,
                             idx,
                             exclude_docs: Optional[Set[int]] = None,
                             top_k_docs: int = 10) -> List[Dict]:
        """
        Aggregates chunk-level retrieval scores into document-level rankings.

        **Aggregation Strategy: Max Pooling.**
        A document's relevance is defined as the maximum score of its constituent
        chunks found in the search results. This assumes that if any part of a paper
        is highly relevant, the whole paper is worth recommending.

        This approach effectively surfaces papers that contain specific, highly
        relevant details, even if other sections are less related.

        Args:
            scores (np.ndarray): Similarity scores from the vector search engine.
            idx (np.ndarray): Corresponding chunk row indices into `self.metadata`.
            exclude_docs (Optional[Set[int]]): Set of internal `doc_idx` values to
                            filter out (e.g., seed papers in recommendation tasks).
            top_k_docs (int): Final number of unique documents to return.

        Returns:
            List[Dict[str, Any]]: Ranked list of document objects, each containing:
                - `doc_idx`: Internal document index.
                - `score`: Aggregated relevance score (max over chunk scores).
                - `title`: Document title.
        """
        if exclude_docs is None:
            exclude_docs = set()

        doc_best = {}  # doc_idx -> best_score

        for s, i in zip(scores, idx):
            m = self.metadata[int(i)]
            d = int(m.get("doc_idx", -1))
            if d in exclude_docs:
                continue
            if d not in doc_best or s > doc_best[d]:
                doc_best[d] = float(s)

        items = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)[:top_k_docs]

        results = []
        for d, sc in items:
            title = self.doc_titles.get(d, "")
            results.append({
                "doc_idx": d,
                "score": sc,
                "title": title,
            })
        return results



@lru_cache(maxsize=1)
def get_index_resources() -> IndexResources:
    return IndexResources(INDEX_DIR)
