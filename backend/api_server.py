# api_server.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uvicorn
from fastapi import FastAPI, HTTPException
from models import * 
import logging, traceback
from search_backend import get_index_resources
from rag_llama import generate_rag_answer





app = FastAPI(title="ArXiv FAISS vs Baseline vs HNSW Search")
app = FastAPI(title="ArXiv FAISS vs Baseline vs HNSW Search + RAG")

logger = logging.getLogger("uvicorn.error")


'''@app.on_event("startup")
def preload_resources():
    """
    伺服器啟動時就先載入：
      - 索引 (FAISS / Zarr / metadata)
      - LLM 模型 (HF / llama-cpp)
    這樣第一個 HTTP 請求不會卡很久。
    """
    try:
        # 如果你有多個 corpus，可以指定一個預設，如 "transformer"
        res = get_index_resources()  # 或 get_index_resources("transformer")
        logger.info("[STARTUP] Index loaded. chunks=%d docs=%d",
                    len(res.metadata), len(res.doc2rows))

        # 載入 HF 模型
        tokenizer, model = get_hf_model()
        #tokenizer, model = get_llm()
        device = next(model.parameters()).device
        logger.info("[STARTUP] HF model loaded on device: %s", device)

    except Exception as e:
        logger.error("[STARTUP] Failed to preload resources: %r", e)
        traceback.print_exc()
        # 這裡你可以選擇直接 raise 讓 uvicorn 啟動失敗
        raise
@app.on_event("startup")
def preload_resources():
    try:
        res = get_index_resources()
        logger.info("[STARTUP] Index loaded. chunks=%d docs=%d",
                    len(res.metadata), len(res.doc2rows))

        llm = get_llm()   # 這裡原本如果是 tokenizer, model = get_llm() 就要改掉
        logger.info("[STARTUP] GGUF model loaded.")

    except Exception as e:
        logger.error("[STARTUP] Failed to preload resources: %r", e)
        traceback.print_exc()
        raise'''


@app.get("/titles", response_model=List[str])
def get_titles(limit: int = 50):
    res = get_index_resources()
    return res.list_some_titles(limit=limit)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
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
        # 這裡不必再擔心首次載入時間，get_hf_model 已經在 startup 跑過
        #_tok, _model = get_hf_model()  # 這行可以保留或乾脆不用回傳值，反正是 singleton
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
