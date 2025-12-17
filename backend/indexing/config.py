# arxiv_faiss_zarr/config.py
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_NUM_DOCS = 0          # <=0: 全部
CHUNK_SIZE = 2048
DTYPE = "f4"
DEFAULT_MAX_CHARS_PER_CHUNK = 2000

SECTION_TITLES = [
    "abstract", "introduction", "related work", "related works",
    "background", "preliminaries", "methods", "materials and methods",
    "experiments", "results", "discussion", "conclusion", "conclusions",
    "acknowledgements",
]

FULLTEXT_CANDIDATES = ["full_text", "body_text", "paper_text", "text"]
SENTENCE_ENDINGS = [".", "。", "!", "?", "！", "？"]

# HNSW params
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
