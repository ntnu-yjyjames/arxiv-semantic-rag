"""
fuzzymatch_vs_bm25.py

目的：
- 比較兩種 title 對應方法在「seed title → internal doc_idx」上的效果：

  1) normalize + fuzzy (RapidFuzz)
  2) BM25 (rank_bm25) over title corpus

指標：
- 正確率：是否有找到原始 doc_idx
- 平均耗時：每個 query 的耗時 (ms)

使用方式：
    python test_title_match_bm25_vs_fuzzy.py \
        --meta-path ./data/index/arxiv_full/metadata.jsonl \
        --num-titles 50000 \
        --num-tests 200
"""

import argparse
import json
import random
import re
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from rapidfuzz import process, fuzz
from rank_bm25 import BM25Okapi


# ---------- utils ----------

def load_metadata_jsonl(meta_path: str) -> List[Dict[str, Any]]:
    metadata: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def normalize_title(text: str) -> str:
    """
    Normalize a title string:
    - collapse whitespace (spaces, tabs, newlines) into single spaces
    - strip leading/trailing spaces
    - lowercase
    """
    if not isinstance(text, str):
        return ""
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip().lower()


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25: lowercase + 按非字母數字切開。
    正式可替換成更好的 tokenizer。
    """
    text = text.lower()
    return re.findall(r"\w+", text)


# ---------- build corpora ----------

def build_title_corpus(metadata: List[Dict[str, Any]],
                       max_docs: int | None = None
                       ) -> Tuple[List[str], List[int]]:
    """
    從 metadata 裡抽出 (title, doc_idx) corpus。
    """
    titles: List[str] = []
    doc_idxs: List[int] = []

    for m in metadata:
        title = m.get("title", "").strip()
        if not title:
            continue
        d = int(m.get("doc_idx", -1))
        if d < 0:
            continue

        titles.append(title)
        doc_idxs.append(d)

        if max_docs is not None and len(titles) >= max_docs:
            break

    return titles, doc_idxs


# ---------- title resolvers ----------

class TitleResolvers:
    def __init__(self,
                 titles: List[str],
                 doc_idxs: List[int],
                 fuzzy_threshold: int = 90,
                 bm25_min_score: float = 3.0):
        assert len(titles) == len(doc_idxs)
        self.titles = titles
        self.doc_idxs = doc_idxs
        self.fuzzy_threshold = fuzzy_threshold
        self.bm25_min_score = bm25_min_score

        # 1) fuzzy corpus：normalized title -> doc_idx list
        self.norm_key_to_docs: Dict[str, List[int]] = {}
        for d, t in zip(doc_idxs, titles):
            key = normalize_title(t)
            if not key:
                continue
            self.norm_key_to_docs.setdefault(key, []).append(d)
        self.norm_keys = list(self.norm_key_to_docs.keys())

        # 2) BM25 corpus：原始 title, tokenized
        tokenized = [tokenize(t) for t in titles]
        self.bm25 = BM25Okapi(tokenized)

    def resolve_fuzzy(self, query: str) -> List[int]:
        """
        先 normalize，再 exact match，
        找不到時用 fuzzy over normalized keys。
        """
        key = normalize_title(query)
        if not key:
            return []

        # 1) exact
        if key in self.norm_key_to_docs:
            return self.norm_key_to_docs[key]

        # 2) fuzzy
        if not self.norm_keys:
            return []

        match = process.extractOne(
            key,
            self.norm_keys,
            scorer=fuzz.ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        if not match:
            return []

        best_key, score, _ = match
        return self.norm_key_to_docs[best_key]

    def resolve_bm25(self, query: str, top_n: int = 3) -> List[int]:
        """
        在原始 title corpus 上做 BM25 檢索，取 top_n，過濾低分結果。
        """
        if not query.strip():
            return []

        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # 取分數最高的 top_n
        order = np.argsort(-scores)[:top_n]

        doc_ids: List[int] = []
        for i in order:
            if scores[i] < self.bm25_min_score:
                continue
            d = self.doc_idxs[int(i)]
            doc_ids.append(d)

        return sorted(set(doc_ids))


# ---------- experiment ----------

def generate_variants(title: str) -> List[str]:
    """
    生成幾種變形 title 來測 robustness。
    可以視需要自行增加規則。
    """
    variants = []

    # 1) 原始
    variants.append(title)

    # 2) 壓縮空白 + lower
    v2 = " ".join(title.split()).lower()
    variants.append(v2)

    # 3) 去掉最後一個詞
    parts = title.split()
    if len(parts) > 3:
        v3 = " ".join(parts[:-1])
        variants.append(v3)

    # 4) 簡單拼錯：把某個詞的兩個字母對調（只做一次）
    if len(parts) > 0:
        p = parts[0]
        if len(p) > 4:
            # swap middle two characters
            s = list(p)
            s[1], s[2] = s[2], s[1]
            parts2 = parts.copy()
            parts2[0] = "".join(s)
            v4 = " ".join(parts2)
            variants.append(v4)

    # 去重
    uniq = []
    seen = set()
    for v in variants:
        if v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq


def run_experiment(meta_path: str,
                   num_titles: int,
                   num_tests: int):
    print(f"[INFO] Loading metadata from {meta_path}")
    metadata = load_metadata_jsonl(meta_path)
    titles, doc_idxs = build_title_corpus(metadata, max_docs=num_titles)
    print(f"[INFO] Built title corpus: {len(titles)} titles")

    resolvers = TitleResolvers(
        titles=titles,
        doc_idxs=doc_idxs,
        fuzzy_threshold=90,
        bm25_min_score=3.0,
    )

    # 隨機抽 num_tests 個樣本
    indices = list(range(len(titles)))
    random.shuffle(indices)
    test_indices = indices[:min(num_tests, len(indices))]

    fuzzy_hits = 0
    bm25_hits = 0
    fuzzy_queries = 0
    bm25_queries = 0

    fuzzy_time = 0.0
    bm25_time = 0.0

    print(f"[INFO] Running {len(test_indices)} test cases...")

    for ti in test_indices:
        true_doc = doc_idxs[ti]
        base_title = titles[ti]
        variants = generate_variants(base_title)

        for q in variants:
            # Fuzzy
            t0 = time.perf_counter()
            docs_fuzzy = resolvers.resolve_fuzzy(q)
            fuzzy_time += time.perf_counter() - t0
            fuzzy_queries += 1
            if true_doc in docs_fuzzy:
                fuzzy_hits += 1

            # BM25
            t0 = time.perf_counter()
            docs_bm25 = resolvers.resolve_bm25(q, top_n=3)
            bm25_time += time.perf_counter() - t0
            bm25_queries += 1
            if true_doc in docs_bm25:
                bm25_hits += 1

    # 統計
    fuzzy_acc = fuzzy_hits / fuzzy_queries if fuzzy_queries else 0.0
    bm25_acc = bm25_hits / bm25_queries if bm25_queries else 0.0

    fuzzy_ms = (fuzzy_time / fuzzy_queries) * 1000 if fuzzy_queries else 0.0
    bm25_ms = (bm25_time / bm25_queries) * 1000 if bm25_queries else 0.0

    print("\n=== Title Matching Experiment (Seed Mapping) ===")
    print(f"Corpus size (titles used): {len(titles)}")
    print(f"Num test titles          : {len(test_indices)}")
    print(f"Num total queries        : {fuzzy_queries} (variants per title)")

    print("\nFuzzy (normalize + RapidFuzz)")
    print(f"  Hit rate  : {fuzzy_acc*100:.2f}%")
    print(f"  Avg time  : {fuzzy_ms:.3f} ms / query")

    print("\nBM25 (rank_bm25 over titles)")
    print(f"  Hit rate  : {bm25_acc*100:.2f}%")
    print(f"  Avg time  : {bm25_ms:.3f} ms / query")

    print("\n[NOTE] Hit rate = proportion of (query variant, method) pairs that")
    print("       correctly matched the original doc_idx.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-path", type=str, required=True,
                    help="Path to metadata.jsonl")
    ap.add_argument("--num-titles", type=int, default=50000,
                    help="Number of titles from corpus to build test index (for speed).")
    ap.add_argument("--num-tests", type=int, default=200,
                    help="Number of base titles to sample for testing.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        meta_path=args.meta_path,
        num_titles=args.num_titles,
        num_tests=args.num_tests,
    )
