"""
Global configuration settings for the arXiv RAG pipeline.

This module centralizes all tunable parameters, including:
1. Model specifications (embedding models).
2. Data processing constraints (chunk sizes, limits).
3. Text segmentation heuristics (arXiv-specific rules).
4. Vector index parameters (FAISS Flat / HNSW settings).

It is imported by the indexing, search, and benchmarking modules
to keep configuration in a single place.
"""

# =========================================================
# 🤖 Model & Data Types
# =========================================================
# The HuggingFace model used for generating dense vector embeddings.
# "all-MiniLM-L6-v2" is chosen for its speed/performance balance (384 dim).
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Data type for vector storage. "f4" (float32) is standard.
# For memory optimization, consider "f2" (float16) if accuracy loss is acceptable.
DTYPE = "f4"


# =========================================================
# ⚙️ Processing & Storage Limits
# =========================================================
# Number of documents to process. Set to 0 to process the FULL dataset.
# Useful for quick debugging with a small subset.
DEFAULT_NUM_DOCS = 0

# Zarr array chunk size (number of vectors per disk chunk).
# Larger values improve I/O throughput but require more RAM during writes.
CHUNK_SIZE = 2048

# Maximum characters per text chunk for embedding.
# Rough character limit chosen so that most chunks stay under the model's token limit (~512 tokens), reducing truncation risk.
DEFAULT_MAX_CHARS_PER_CHUNK = 2000


# =========================================================
# 📄 ArXiv Segmentation Heuristics (Rule-based)
# =========================================================
# Common section headers found in academic papers.
# Used to split full text into semantically meaningful sections.
SECTION_TITLES = [    
    "abstract", "introduction", "related work", "related works",
    "background", "preliminaries", "methods", "materials and methods",
    "experiments", "results", "discussion", "conclusion", "conclusions",
    "acknowledgements",
]

# Column names in the DataFrame that might contain the full paper text.
FULLTEXT_CANDIDATES = ["full_text", "body_text", "paper_text", "text"]

# Punctuation marks used to identify sentence boundaries for soft-chunking.
SENTENCE_ENDINGS = [".", "。", "!", "?", "！", "？"]


# =========================================================
# ⚡ FAISS HNSW Index Parameters (Approximate Search)
# =========================================================
# Trade-off Analysis:
# - HNSW_M: Higher values = Higher accuracy but higher memory usage per vector.
# - EF_CONSTRUCTION: Higher values = Better index quality but slower build time.
# - EF_SEARCH: Higher values = Higher recall but higher query latency.

# The number of neighbors used in the graph. (Memory intensive)
HNSW_M = 32

# Depth of search during index construction. (Build-time intensive)
HNSW_EF_CONSTRUCTION = 200

# Depth of search during query time. (Latency intensive)
# Note: Can be dynamically adjusted at runtime via index.hnsw.efSearch.
HNSW_EF_SEARCH = 64