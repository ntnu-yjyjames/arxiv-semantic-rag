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
from arxiv_faiss_zarr.config import EMBED_MODEL_NAME

#INDEX_DIR = "./arxiv_zarr/arxiv_index_demo_transformer"  
INDEX_DIR = "./arxiv_zarr/arxiv_index_full"  

# 與 build 時一致
HNSW_EF_SEARCH = 64


def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def _load_metadata(meta_path: str) -> List[Dict[str, Any]]:
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


class IndexResources:
    """
    負責載入並管理：
      - embeddings.zarr
      - faiss_index_flat.bin
      - faiss_index_hnsw.bin (若存在)
      - metadata.jsonl
      - doc/chunk 對應關係
    """


    def __init__(self, index_dir: str):
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
                self.doc_titles[doc_idx] = title
            key = title.strip().lower()
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
        rows = self.doc2rows.get(doc_idx)
        if not rows:
            raise ValueError(f"doc_idx {doc_idx} has no chunks")
        embs = self.emb_store[rows, :]  # (n_chunks_d, dim)
        v = embs.mean(axis=0)
        v = l2_normalize(v[None, :])[0].astype("float32")
        return v

    def compute_user_vector_from_titles(self, titles: List[str]) -> np.ndarray:
        doc_indices: List[int] = []
        for t in titles:
            key = t.strip().lower()
            if not key:
                continue
            doc_indices.extend(self.title2docs.get(key, []))
        doc_indices = sorted(set(doc_indices))

        if not doc_indices:
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
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss

    def search_faiss_flat(self, user_vec: np.ndarray, top_k: int = 10):
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
        NumPy full-scan baseline.

        mode = "cold":
            每次 query 都從 Zarr 讀 embeddings，再 full-scan。
            模擬 disk-based pipeline 的 end-to-end latency。
        mode = "hot":
            embeddings 只在第一次載入到 self._emb_hot，之後每次 query 都在記憶體中 full-scan。
            模擬 in-memory NumPy baseline。
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
        titles = []
        for d, t in self.doc_titles.items():
            titles.append(t)
            if len(titles) >= limit:
                break
        return titles
    


    @staticmethod
    def compute_recall_at_k(ground_idx: np.ndarray,
                            approx_idx: np.ndarray) -> float:
        gset = set(int(i) for i in ground_idx)
        aset = set(int(i) for i in approx_idx)
        if not gset:
            return 0.0
        return len(gset & aset) / float(len(gset))

    def benchmark_backends(self, user_vec: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        同一個 user_vec 下，對 baseline / faiss_flat / hnsw 做比較：
          - latency
          - memory
          - recall@k (以 baseline 作為 ground truth)
        回傳：每個 backend 一個 dict，含 metrics + results。
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
        將使用者的問題轉成向量（跟建 index 時用的模型一致）。
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
        用問題文字（question）做向量，對 flat index 搜尋 top_k chunks，
        回傳格式為 format_results()。
        """
        q_vec = self.encode_query(question)
        scores, idx, _, _, _ = self.search_faiss_flat(q_vec, top_k=top_k)
        return self.format_results(scores, idx)
    
    def aggregate_doc_scores(self, scores, idx, exclude_docs=None, top_k_docs=10):
        """
        將 chunk-level 結果聚合成 doc-level：
          - 每篇 doc 的分數 = 該 doc 所有 chunk 分數的最大值
          - 可選擇排除 exclude_docs (set of doc_idx)
        回傳: List[{"doc_idx", "score", "title"} ...]，已排序取前 top_k_docs
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
        根據使用者勾選的 titles 做推薦：
          1) 用這些 titles 對應的 doc 建立 user vector
          2) 用 backend (faiss_flat / faiss_hnsw / baseline) 搜尋 chunk
          3) 將 chunk-level 結果聚合為 doc-level，
             並排除這些 titles 所對應的 doc（只推薦新文章）

        回傳：doc-level 推薦結果 list[{'doc_idx', 'score', 'title'}, ...]
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
        將 chunk-level 結果聚合成 doc-level：
          - 每篇 doc 的分數 = 該 doc 所有 chunk 分數的最大值
          - exclude_docs: 需要排除的 doc_idx（例如使用者偏好 seed）
        回傳: [{'doc_idx', 'score', 'title'}, ...]，已排序取前 top_k_docs
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
