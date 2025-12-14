from pydantic import BaseModel
from typing import List, Literal

class SearchRequest(BaseModel):
    titles: List[str]
    backend: Literal["faiss_flat", "faiss_hnsw", "baseline"] = "faiss_flat"
    top_k: int = 10
    ef_search: int | None = None 

class SearchResponse(BaseModel):
    backend: str
    elapsed_ms: float
    memory_rss_mb: float
    memory_delta_mb: float
    results: List[dict]


class BenchmarkRequest(BaseModel):
    titles: List[str]
    top_k: int = 10


class BackendBenchmark(BaseModel):
    backend: str
    elapsed_ms: float
    memory_rss_mb: float
    memory_delta_mb: float
    recall_at_k: float
    results: List[dict]


class BenchmarkResponse(BaseModel):
    top_k: int
    backends: List[BackendBenchmark]

class RagRequest(BaseModel):
    question: str
    top_k: int = 10   
    max_tokens: int = 512


class RagResponse(BaseModel):
    question: str
    top_k: int
    answer: str
    contexts: List[dict]  # title, section, chunk_text

class RecommendRequest(BaseModel):
    titles: List[str]
    backend: Literal["faiss_flat", "faiss_hnsw", "baseline"] = "faiss_flat"
    top_k_docs: int = 10


class RecommendItem(BaseModel):
    doc_idx: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    backend: str
    results: List[RecommendItem]

