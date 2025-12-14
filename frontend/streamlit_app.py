# streamlit_app.py
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ArXiv Semantic Search",
    layout="wide",
)


def fetch_titles(limit: int = 100):
    try:
        resp = requests.get(f"{BACKEND_URL}/titles", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to fetch titles from backend: {e}")
        return []


def call_search(titles, backend="faiss_flat", top_k=10):
    payload = {
        "titles": titles,
        "backend": backend,
        "top_k": top_k,
    }
    resp = requests.post(f"{BACKEND_URL}/search", json=payload)
    resp.raise_for_status()
    return resp.json()


def call_recommend(titles, backend="faiss_flat", top_k_docs=10):
    payload = {
        "titles": titles,
        "backend": backend,
        "top_k_docs": top_k_docs,
    }
    resp = requests.post(f"{BACKEND_URL}/recommend", json=payload)
    resp.raise_for_status()
    return resp.json()


def init_session_state():
    if "search_data" not in st.session_state:
        st.session_state["search_data"] = None
    if "rec_data" not in st.session_state:
        st.session_state["rec_data"] = None


def main():
    init_session_state()

    st.title("ArXiv Semantic Search (Flat / HNSW / Baseline)")

    st.markdown(
        """
        ### System Workflow & Capabilities
        
        1.  **Define Query Profile:** Select **Seed Papers** to establish the user interest vector.
        2.  **Configure Index Backend:** Choose the vector search algorithm (**Baseline** / **FAISS Flat** / **HNSW**).
        3.  **Execute Dual-Granularity Pipeline:**
            - **Passage-Level Retrieval:** Perform semantic search on precise text passages (abstracts) for context injection.
            - **Paper Discovery:** Suggest novel papers based on overall similarity (with **Seed Deduplication**).

        ---
         **Performance Analysis:** Navigate to the **Benchmark Page** (sidebar) to evaluate the trade-offs between **Latency**, **Memory**, and **Recall** across different backends.
        """
    )
    # ---- Sidebar ----
    st.sidebar.header("Search / Recommend Settings")
    backend_label = st.sidebar.radio(
        "Vector Retrieval backend",
        ["FAISS Flat", "FAISS HNSW", "Baseline"],
        index=0
    )
    backend_map = {
        "FAISS Flat": "faiss_flat",
        "FAISS HNSW": "faiss_hnsw",
        "Baseline": "baseline",
    }
    backend = backend_map[backend_label]

    top_k_chunks = st.sidebar.slider("Top-K passages (Search)", min_value=1, max_value=30, value=10, step=1)
    top_k_docs = st.sidebar.slider("Top-K papers (Recommendation)", min_value=3, max_value=30, value=10, step=1)
    
    ef_search = st.sidebar.slider("HNSW efSearch", min_value=16, max_value=512, value=64, step=16)
    if backend != "faiss_hnsw":
        st.sidebar.caption("efSearch is only used in  FAISS HNSW.")

    # ---- Step 1----
    st.subheader("Step 1: Define Query Profile (Seed Papers)")

    sample_titles = fetch_titles(limit=100)

   
    selected_titles = st.multiselect(
        "Select Reference Papers from Dataset:",
        options=sample_titles,
        help="Select papers to form the basis of the user interest vector."  
    )

    
    manual_titles_text = st.text_area(
        "Or Input Custom Titles (Line-separated):",
        value="",
        height=120,
        help="Manually inject external paper titles into the query profile."
    )
    manual_titles = [t.strip() for t in manual_titles_text.splitlines() if t.strip()]

    all_titles = list({t for t in selected_titles + manual_titles if t})

    st.markdown("### Active Query Profile (Seed Papers)")

    if not all_titles:
        st.info("Query profile is empty. Please select or input reference papers above to **initialize the user interest vector**.")
    else:
        st.write(f"**{len(all_titles)}** Seed Papers Selected for Profiling:")
        for i, t in enumerate(all_titles, start=1):
            st.markdown(f"- **{i}.** {t}")
    st.markdown("---")

    # ---- Step 2：同時執行「搜尋 + 推薦」----
    
    st.subheader("Step 2: Execute Integrated Retrieval & Recommendation Pipeline")

    
    run_both = st.button("Run Search & Recommendation Pipeline")

    if run_both:
        if not all_titles:
            st.warning(" Action Required: Please define the query profile (select papers) before execution.")
        else:
            with st.spinner("Dispatching query to backend: Vectorizing input & traversing HNSW index..."):
                try:
                    search_payload = {
                        "titles": all_titles,
                        "backend": backend,
                        "top_k": top_k_chunks,
                        "ef_search": ef_search if backend == "faiss_hnsw" else None,
                    }
                    resp_search = requests.post(f"{BACKEND_URL}/search", json=search_payload)
                    resp_search.raise_for_status()
                    search_data = resp_search.json()

                    rec_payload = {
                        "titles": all_titles,
                        "backend": backend,
                        "top_k_docs": top_k_docs,
                        "ef_search": ef_search if backend == "faiss_hnsw" else None,
                    }
                    resp_rec = requests.post(f"{BACKEND_URL}/recommend", json=rec_payload)
                    resp_rec.raise_for_status()
                    rec_data = resp_rec.json()

                    st.session_state["search_data"] = search_data
                    st.session_state["rec_data"] = rec_data

                except Exception as e:
                    st.error(f"Search/Recommend failed: {e}")

    
    col_search, col_rec = st.columns(2)

    with col_search:
        st.markdown("###  Semantic Retrieval ")

        data = st.session_state.get("search_data")
        if not data:
            st.info("Awaiting pipeline execution. Please run the **Search & Recommendation** pipeline above.")
        else:

            st.markdown(
                f"""
                **Backend:** `{data['backend']}` &nbsp;|&nbsp; 
                **Latency:** `{data['elapsed_ms']:.2f} ms` &nbsp;|&nbsp; 
                **RSS:** `{data['memory_rss_mb']:.1f} MB` &nbsp;|&nbsp; 
                **ΔMem:** `{data['memory_delta_mb']:.2f} MB`
                """, 
                unsafe_allow_html=True
            )

            results = data.get("results", [])
            if not results:
                st.warning("No relevant passages retrieved.")
            else:
                
                st.markdown("####  Top-k Retrieved Contexts")
                
                for r in results:
                    # Expander 標題：
                    # 格式：Rank X | Score: 0.xxxx | Title...
                    # Sim. Score (Similarity Score) 比單純 Score 更精確
                    header = f"Rank {r['rank']} | Sim. Score: {r['score']:.4f} | {r['title']}"
                    
                    with st.expander(header):
                        # Metadata 區塊
                        st.markdown(
                            f"""
                            - **Section Metadata:** `{r['section']}`
                            - **Global Vector ID:**  `{r['chunk_row']}`
                            """
                        )
                        st.divider() 
                        
                        st.caption("Passage Content:")
                        st.write(r.get("chunk_text", ""))

    
    with col_rec:
        st.markdown("###  Paper Recommendation ")
    
        st.caption("Candidates are **deduplicated** against the Query Profile (Seed Papers) to ensure **novel** content discovery.")

        rec_data = st.session_state.get("rec_data")
        if not rec_data:
            
            st.info("Awaiting pipeline execution. Please run the **Search & Recommendation** pipeline above.")
        else:
            rec_results = rec_data.get("results", [])
            if not rec_results:
                st.warning("No candidates generated. This may be due to **data sparsity** or the query profile being too specific.")
            else:
                st.markdown("####  Top-N Recommended Papers")
                
                for rank, item in enumerate(rec_results, start=1):
                    # 顯示項目：
                    # - Sim. Score: 相似度分數
                    # - Doc ID: 系統內部 ID
                    st.markdown(
                        f"""
                        **Rank {rank}** — Sim. Score: `{item['score']:.4f}`<br>
                        **{item['title']}** <br>
                        <small style='color:gray'>(System Doc ID: `{item['doc_idx']}`)</small>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.divider() 

if __name__ == "__main__":
    main()
