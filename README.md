# arXiv Semantic Retrieval & RAG

An end-to-end **ML systems project** for semantic retrieval, paper recommendation, and
**retrieval-augmented generation (RAG)** on arXiv data.

The system compares **NumPy baseline, FAISS Flat, and FAISS HNSW** under real workloads,
analyzing **latency, memory, recall@k**, and **GPU vs CPU scalability**, and integrates
a local LLM backend (Ollama) for RAG.

**Tech:** FAISS · NumPy · FastAPI · Streamlit · sentence-transformers · Ollama · GPU acceleration

## System Features
- **Semantic Retrieval Systems**
  Dense retrieval over arXiv papers using NumPy (baseline), FAISS Flat (exact),
  and FAISS HNSW (ANN), with consistent evaluation against a ground-truth reference.

- **Recommendation via User Embeddings**
  Paper recommendation by aggregating user-selected seed papers into a
  preference vector and retrieving semantically related, unseen papers.

- **System-Level Benchmarking**
  Interactive and offline analysis of **latency, memory usage, recall@k**,
  HNSW `efSearch` trade-offs, and **CPU vs GPU build-time scalability**.

- **Retrieval-Augmented Generation (RAG)**
  A thin RAG layer that reuses the retrieval pipeline to ground LLM responses
  via a local Ollama backend.
  
## System Overview

![System Structure](/backend/sys_arch.png)


* At a high level, the system consists of:
    * Data & Index Layer
        * arXiv metadata → passages (title + abstract)
        * Sentence-transformer embeddings
        * Zarr-backed storage for large embedding arrays
        * FAISS Flat / HNSW indexes
        * NumPy full-scan as reference baseline
    * Backend (FastAPI)
        * Semantic retrieval & recommendation APIs
        * Benchmark APIs for latency / memory / recall
        * RAG API that retrieves context and calls Ollama
    * Frontend (Streamlit)
        * Semantic search & recommendation UI
        * Live backend switching (Flat / HNSW / Baseline)
        * HNSW efSearch tuning
        * Benchmark dashboards
    * LLM Layer
        * Local Ollama backend (e.g. llama3.1:8b)
        * Thin RAG wrapper for pluggable LLM runtimes


## Benchmarks & Trade-offs

This project benchmarks **exact vs approximate** vector retrieval under realistic settings, measuring **latency, memory**, and **Recall@k** (against a NumPy full-scan ground truth).

### Faiss v.s. Baseline Benchmark(Demo)
A side-by-side comparison in the Streamlit UI showing **NumPy (baseline)**, **FAISS Flat (exact)**, and **FAISS HNSW (ANN)** with consistent query inputs.

> **Takeaway:** HNSW achieves near-Flat quality with significantly lower latency after tuning `efSearch`.

This section summarizes end-to-end retrieval performance under identical query settings, comparing exact and approximate backends in terms of latency, memory usage, and recall.

1) Exact vs Approximate Search (Latency · Memory · Recall)
<img src="/backend/bench_images/data50000_top50.png" width="850" />

We compare three backends using the same query vectors and top-k configuration:
* NumPy baseline (full scan, ground truth)
* FAISS Flat (exact nearest neighbour)
* FAISS HNSW (approximate nearest neighbour)

> **Takeaway**: FAISS HNSW achieves orders-of-magnitude lower latency than exact search while retaining high recall.
FAISS Flat serves as an exact-search reference within the FAISS ecosystem, whereas the NumPy baseline defines the ground-truth recall.

2) HNSW efSearch Trade-off (Recall vs Search Effort)
<img src="/backend/bench_images/efs_recall.png" width="850" />

This ablation study sweeps the HNSW efSearch parameter and reports Recall@k relative to FAISS Flat.

> **Takeaway**: Recall increases monotonically with efSearch, with diminishing returns beyond a moderate range.
In practice, efSearch ≈ 64 provides a strong balance between retrieval quality and latency.

3) Index Build-Time Scaling (CPU vs GPU)
<img src="/backend/bench_images/build_time.png" width="850" />

We benchmark index construction time as the corpus size grows, comparing CPU-only and GPU-accelerated pipelines.

> **Takeaway**: GPU acceleration significantly improves build throughput and scales more favorably with corpus size, achieving tens-of-times speedups over CPU-only builds for larger collections.

Notes on Benchmark Design

Recall@k is computed relative to the NumPy full-scan baseline.

All measurements reflect end-to-end system latency, including backend calls and result aggregation.

The goal is to characterize system-level trade-offs, rather than isolated kernel performance.


## Installation & Running

### 1. Set up the environment (with or without conda)

```bash
# Clone this repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

#### Option A – Using conda (example)
```bash
# Create and activate a Python environment
conda create -n arxiv-rag python=3.10 -y
conda activate arxiv-rag
```

#### Option B – Using plain venv + pip
```bash
# In your project root
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```
#### Other dependencies
```bash
# Install PyTorch (choose a wheel that matches your GPU / CUDA or CPU-only)
# Example for CUDA 12.1 (RTX 3080):
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```
If you want to use local RAG, please install Ollama and pull a model, e.g.:
```bash
ollama pull llama3.1:8b
```
### 2. Build an index (one-time / occasional)
Given an arXiv-style CSV with at least title and abstract, run the index builder.
For development and benchmarking you can start with ~50K documents:
```bash
python cli_arxiv.py build \
  --csv-path ./data/arxiv_cornell_title_abstract.csv \
  --output-dir ./data/index/arxiv-50k \
  --num-docs 50000 \
  --batch-size 512 \
  --max-chars-per-chunk 2000
```
Adjust `--num-docs` and `--batch-size` to match your machine.
Once built, the index can be reused until the data changes.
### 3. Start the backend (FastAPI)
```bash
uvicorn api_server:app --reload --port 8000
```
This exposes a REST API at `http://localhost:8000`
(e.g. `/search`, `/recommend`, `/benchmark`, `/rag-answer`).
### 4. Start the Streamlit UI
In a second terminal:
```bash
streamlit run streamlit_app.py
```

By default the UI is available at `http://localhost:8501`:
* Home
    * Select seed papers (user preferences)
    * Choose backend (FAISS Flat / HNSW / Baseline)
    * Adjust top-K passages / top-K papers and efSearch (for HNSW)
    * Inspect semantic retrieval and paper recommendations
* Benchmark
    * View static plots comparing Flat / HNSW / Baseline
    * Inspect HNSW efSearch latency/recall trade-offs
    * See CPU vs GPU build-time scalability
* RAG (if enabled)
    * Ask natural language questions
    * See answers generated by a local Ollama model based on retrieved arXiv abstracts/passages
## Motivation & Engineering Goals
This repository is designed to explore real-world ML systems questions, such as:
* When is approximate nearest neighbour search “good enough”?
* How do latency, memory, and recall trade off in practice?
* How does retrieval quality affect RAG outputs?
* How do system choices (CPU vs GPU, Flat vs HNSW) impact scalability?
It serves as both:
* a learning playground for semantic retrieval & RAG, and
* a reference ML systems project showcasing end-to-end design, benchmarking, and deployment.
## Disclaimer
* This is a clean-room, original implementation inspired by common industry architectures.
* All code is written from scratch for educational and demonstration purposes.
* Only public arXiv data is used.
* No proprietary code, data, or internal systems are included.