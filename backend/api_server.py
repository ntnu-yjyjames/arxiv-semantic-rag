# api_server.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uvicorn
from fastapi import FastAPI, HTTPException
from models import * 
import logging, traceback
from search_backend import get_index_resources
from rag_llama import generate_rag_answer






app = FastAPI(title="ArXiv FAISS vs Baseline vs HNSW Search + RAG")

logger = logging.getLogger("uvicorn.error")

@app.get("/titles", response_model=List[str])
def get_titles(limit: int = 50):
    """
    Retrieves a sample of paper titles from the dataset.
    
    This endpoint serves as a helper for the frontend UI, allowing users 
    to select papers to form their interest vector (query vector) without 
    needing to type specific queries.

    Args:
        limit (int): The maximum number of titles to return. Defaults to 50.

    Returns:
        List[str]: A list of paper titles available in the index.
    """
    res = get_index_resources()
    return res.list_some_titles(limit=limit)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Executes a semantic vector search using the specified backend strategy.

    Logic Flow:
    1. **Vectorization**: Aggregates selected paper titles into a single "user interest" vector.
    2. **Retrieval**: Dispatches the query to one of three engines:
        - `baseline`: NumPy full-scan (in-memory matrix multiply, exact).
        - `faiss_flat`: Brute-force search via FAISS Flat (CPU, exact).
        - `faiss_hnsw`: Approximate nearest neighbor graph search (low latency, scalable).
    3. **Telemetry**: Captures end-to-end execution time and memory deltas for the request.

    Args:
        req (SearchRequest): Contains the user's selected titles and backend configuration.

    Returns:
        SearchResponse: Search results (titles, passages) augmented with performance metrics.
    
    Raises:
        HTTPException(400): If the title list is empty or contains unknown titles.
    """
    if not req.titles:
        raise HTTPException(status_code=400, detail="titles cannot be empty")

    res = get_index_resources()

    try:
        user_vec = res.compute_user_vector_from_titles(req.titles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    import time
    start = time.perf_counter()

    if req.backend == "faiss_flat":
        scores, idx, elapsed, rss_mb, delta_mb = res.search_faiss_flat(user_vec, top_k=req.top_k)
    elif req.backend == "faiss_hnsw":
        scores, idx, elapsed, rss_mb, delta_mb = res.search_hnsw(
            user_vec,
            top_k=req.top_k,
            ef_search=req.ef_search,
        )
    else:  # baseline
        scores, idx, elapsed, rss_mb, delta_mb = res.search_baseline(
            user_vec,
            top_k=req.top_k,
            mode="hot",   
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    results = res.format_results(scores, idx)

    return SearchResponse(
        backend=req.backend,
        elapsed_ms=elapsed_ms,
        memory_rss_mb=rss_mb,
        memory_delta_mb=delta_mb,
        results=results,
    )

@app.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(req: BenchmarkRequest):
    """
    Runs a comparative performance analysis across all available search backends.

    This endpoint executes the same "user interest" vector against:
    1. **Baseline** (NumPy exact full-scan)
    2. **FAISS Flat** (CPU exact)
    3. **FAISS HNSW** (approximate ANN)

    It aggregates key metrics (e.g., latency, memory delta, recall@k) to generate
    performance reports for the frontend dashboard.

    Args:
        req (BenchmarkRequest): User-provided titles and top-k configuration.

    Returns:
        BenchmarkResponse: A structured report comparing metrics across backends.
    
    Raises:
        HTTPException(400): If input titles are invalid.
    """
    if not req.titles:
        raise HTTPException(status_code=400, detail="titles cannot be empty")

    res = get_index_resources()

    try:
        user_vec = res.compute_user_vector_from_titles(req.titles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    backends = res.benchmark_backends(user_vec, top_k=req.top_k)
    return BenchmarkResponse(top_k=req.top_k, backends=backends)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Generates paper-level recommendations based on semantic similarity.

    Unlike the raw `/search` endpoint which returns individual passages, this endpoint:
    1. Retrieves a large pool of candidate passages (e.g., top 1000 chunks).
    2. Groups passages by their parent document (doc_idx).
    3. Scores each document based on the best (maximum) passage similarity.
    4. Returns the top unique papers ranked by this document score.

    Args:
        req (RecommendRequest): Includes `backend` choice and `top_k_docs` limit.

    Returns:
        RecommendResponse: A list of recommended papers with their doc indices, titles, and scores.

    Raises:
        HTTPException(400): If input titles are invalid.
        HTTPException(500): For unexpected internal aggregation errors.
    """

    if not req.titles:
        raise HTTPException(status_code=400, detail="titles cannot be empty")

    res = get_index_resources()  # 如果你有 corpus 切換就加 index_name 參數

    try:
        doc_results = res.recommend_docs_from_titles(
            titles=req.titles,
            backend=req.backend,
            top_k_docs=req.top_k_docs,
            candidate_chunks=1000,   # 可以視資料量調整
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Recommend error: %r", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"recommend failed: {e}")

    return RecommendResponse(
        backend=req.backend,
        results=doc_results,
    )



@app.post("/rag-answer", response_model=RagResponse)
def rag_answer(req: RagRequest):
    """
    Orchestrates the complete Retrieval-Augmented Generation (RAG) pipeline.

    This endpoint transforms a user question into a grounded answer using a two-stage process:
    1. **Retrieval Phase**: Embeds the question and performs a semantic search (HNSW/Flat) 
       to fetch the top-k most relevant text chunks from the arXiv corpus.
    2. **Generation Phase**: Constructs a prompt using the retrieved chunks as "context" 
       and instructs the LLM to answer the question based *solely* on this context.

    Args:
        req (RagRequest): Contains the user question, top_k limits, and max_token constraints.

    Returns:
        RagResponse: 
            - `answer`: The synthesized response from the LLM.
            - `contexts`: The raw text chunks used as evidence (crucial for citation/verification).

    Raises:
        HTTPException(400): If the question is empty.
        HTTPException(500): For retrieval or generation failures (logged separately).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        res = get_index_resources()  # startup 已經載過了，這裡只是取 cache
        chunks = res.search_chunks_for_question(req.question, top_k=req.top_k)
    except Exception as e:
        logger.error("RAG retrieval error: %r", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"retrieval failed: {e}")

    try:
        
        answer = generate_rag_answer(
            question=req.question,
            chunks=chunks,
            max_tokens=req.max_tokens,
        )
        
    except Exception as e:
        logger.error("RAG generation error: %r", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

    return RagResponse(
        question=req.question,
        top_k=req.top_k,
        answer=answer,
        contexts=chunks,
    )



if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
