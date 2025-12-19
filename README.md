# arXiv Semantic Retrieval & RAG

An end-to-end **ML systems project** for semantic retrieval, paper recommendation, and
**retrieval-augmented generation (RAG)** on arXiv data.

The system compares **NumPy baseline, FAISS Flat, and FAISS HNSW** under real workloads,
analyzing **latency, memory, recall@k**, and **GPU vs CPU scalability**, and integrates
a local LLM backend (Ollama) for RAG.

**Tech:** FAISS · NumPy · FastAPI · Streamlit · sentence-transformers · Ollama · GPU acceleration

## System Features
- **Semantic Retrieval Systems** :
  Dense retrieval over arXiv papers using NumPy (baseline), FAISS Flat (exact),
  and FAISS HNSW (ANN), with consistent evaluation against a ground-truth reference.

- **Recommendation via User Embeddings** :
  Paper recommendation by aggregating user-selected seed papers into a
  preference vector and retrieving semantically related, unseen papers.

- **System-Level Benchmarking** :
  Interactive and offline analysis of **latency, memory usage, recall@k**,
  HNSW `efSearch` trade-offs, and **CPU vs GPU build-time scalability**.

- **Retrieval-Augmented Generation (RAG)** :
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


## Motivation & Engineering Goals

This repository is built to explore **practical ML systems questions** around semantic retrieval and RAG, including:

- When is approximate nearest neighbour search “good enough”?
- How do latency, memory usage, and recall trade off in practice?
- How does retrieval quality shape downstream RAG answers?
- How do system choices (CPU vs GPU, Flat vs HNSW) affect scalability and cost?

It is intended to be both:

- an **applied reference implementation** for semantic retrieval & RAG, and  
- an **end-to-end ML systems case study**, covering design, indexing, benchmarking, and deployment.

## Dataset

All experiments in this project are conducted on a dataset derived from the
**arXiv metadata snapshot** released by Cornell University
(`arxiv-metadata-oai-snapshot.json`).

Starting from the official metadata dump, I implemented a custom preprocessing
pipeline to extract titles and abstracts, normalize text fields, and export the
resulting corpus into a structured CSV format
(**arxiv_cornell_title_abstract.csv**). The preprocessing logic is fully
reproducible and implemented in:
    - `utils/json2csv.py`

Official dataset source:
https://www.kaggle.com/datasets/Cornell-University/arxiv

While this dataset represents only a subset of the full arXiv content (titles
and abstracts rather than full text), it provides a realistic and widely used
workload for evaluating semantic retrieval and recommendation systems. The
corpus scale is sufficient to expose system-level behaviors such as cold-start
latency, memory allocation patterns, ANN parameter sensitivity, and scalability
trends.

The primary focus of this project is **retrieval-system trade-offs**—including
latency, memory stability, and recall—rather than absolute throughput on
full-text corpora.


## Performance Benchmarks & System Analysis

This project benchmarks **exact vs. approximate** vector retrieval under
production-oriented settings, focusing on **system latency, memory stability,
and Recall@k** to expose trade-offs between brute-force matrix operations (NumPy)
and indexed ANN retrieval (FAISS).

### 1. Cold Start: System Latency & Memory Stability

<img src="/backend/bench_images/cold_start_benchmark.png" width="850" />

> **Observation:** Under cold-start conditions, the NumPy baseline exhibits severe
instability, incurring a **73 ms latency spike** and an **~83 MB memory allocation
surge** during initialization. In contrast, FAISS HNSW remains stable at **0.29 ms**
with negligible memory overhead (<0.1 MB).

* **Baseline (NumPy):** 73.70 ms (I/O & allocation bound)
* **FAISS HNSW:** 0.29 ms (memory-resident)
* **System-level speedup:** ~254×

**Insight:** Eliminating initialization and allocation bottlenecks is critical for
low-latency, stateless ML microservices.

---

### 2. Hot Cache: Algorithmic Latency

<img src="/backend/bench_images/hot_start_benchmark.png" width="850" />

> **Observation:** With all data resident in memory, NumPy’s BLAS-optimized linear
scan achieves competitive performance (**2.66 ms**) on small datasets. However,
FAISS HNSW delivers a **~4× algorithmic speedup (0.66 ms)** by reducing search
complexity from \(O(N)\) to sublinear traversal.

**Takeaway:** Exact scans may appear fast at small scale, but degrade linearly,
while HNSW maintains sub-millisecond latency as data grows.

---

### 3. Recall vs. Latency Trade-off (`efSearch`)

<img src="/backend/bench_images/efs_recall.png" width="850" />

> **Takeaway:** Tuning `efSearch` reveals a clear **Pareto frontier**, achieving
**0.96 Recall@k** while maintaining sub-millisecond latency—enabling controlled
trade-offs between retrieval quality and throughput.

---

### 4. Index Build-Time Scaling (CPU vs GPU)

<img src="/backend/bench_images/build_time.png" width="850" />

> **Takeaway:** GPU acceleration provides tens-of-times speedups in index
construction as corpus size increases, ensuring scalable index refresh in CI/CD
pipelines.



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

## Disclaimer
* This is a clean-room, original implementation inspired by common industry architectures.
* All code is written from scratch for educational and demonstration purposes.
* Only public arXiv data is used.
* No proprietary code, data, or internal systems are included.