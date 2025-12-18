# pages/01_Faiss_vs_Baseline_Benchmark.py
import json
import requests
import streamlit as st
import pandas as pd
import altair as alt

from utils.fetch_titles import fetch_titles
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FAISS Flat vs HNSW vs Baseline Benchmark",
    layout="wide",
)





def run_benchmark(titles, top_k: int):
    """
    Trigger the comparative benchmark analysis on the backend.

    This function sends the seed titles to the `/benchmark` endpoint, which
    runs the same query against multiple backends (Baseline, FAISS Flat,
    FAISS HNSW) and returns their aggregated metrics.

    Args:
        titles (List[str]): Selected paper titles used to construct the user vector.
        top_k (int): The 'K' in Recall@K and top-k retrieval.

    Returns:
        dict: Parsed JSON response containing a `backends` list, where each item has:
            - `backend`: backend name
            - `elapsed_ms`, `memory_rss_mb`, `memory_delta_mb`, `recall_at_k`
            - `results`: hydrated top-k passages for that backend
    """
    payload = {
        "titles": titles,
        "top_k": top_k,
    }
    resp = requests.post(f"{BACKEND_URL}/benchmark", json=payload)
    resp.raise_for_status()
    return resp.json()


def main():
    """
    Entry point for the "System Performance Benchmark" dashboard.

    This page serves as a comprehensive evaluation suite for the vector search
    subsystem, combining live interactive profiling with pre-computed scalability
    experiments. The goal is to justify architectural choices through empirical data.

    **Dashboard Architecture:**

    1. **Live Search Benchmark (Interactive):**
       - **Objective:** Evaluate real-time trade-offs between accuracy and resource
         consumption on specific user queries.
       - **Comparison:** `Baseline` (exact ground truth, NumPy full-scan) vs.
         `FAISS Flat` (optimized exact, cosine / inner product) vs.
         `FAISS HNSW` (graph-based ANN).
       - **Key Metrics:**
         - `Recall@K`: approximation error relative to the exact baseline.
         - `Latency`: end-to-end retrieval time (ms).
         - `Extra Memory`: per-query RSS delta.

    2. **HNSW Hyperparameter Ablation (Static Analysis):**
       - **Objective:** Determine a reasonable `efSearch` configuration for production.
       - **Methodology:** Sweep `efSearch` from 16 to 256 and plot the latency–recall
         trade-off (Pareto frontier).
       - **Insight:** Highlights a "knee point" (e.g. around `efSearch=64`) where
         marginal recall gains no longer justify the additional latency.

    3. **Build-time Scalability Analysis (Hardware Benchmarking):**
       - **Objective:** Assess the impact of GPU-accelerated embeddings on index
         construction time as the corpus grows.
       - **Methodology:** Measure end-to-end build times (embedding + FAISS index
         build on CPU) for different corpus sizes (e.g. N=1k to N=50k).
       - **Insight:** Shows the asymptotic speedup from CUDA acceleration on the
         embedding stage and motivates the use of GPU for large-scale RAG.
    """
    st.title("FAISS Flat vs HNSW vs Baseline：Latency / Memory / Recall Benchmark")

    st.markdown(
        """
        This benchmark profiles query performance using a **unified query vector** across three distinct backends:

        - **Baseline (NumPy):** Exhaustive linear scan (serves as the **Ground Truth**).
        - **FAISS Flat:** Optimized exact nearest neighbor search (L2).
        - **FAISS HNSW:** Graph-based **Approximate Nearest Neighbor (ANN)** search.

        ** Experimental Setup:**
        - **Dataset Scale:** $\\approx$ 50,000 documents (Consistent with the live demo corpus).
        - **Encoder Model:** `all-MiniLM-L6-v2` (384-dimensional dense vectors).
        - **HNSW Config:** `M=32`, `efConstruction=200`, `efSearch=64` (Fixed for this comparison).
        - **Query Scope:** `top_k=10`.

        **Performance Metrics:**
        - **Query Latency:** End-to-end execution time (`elapsed_ms`).
        - **Memory Footprint:** Resident Set Size (`RSS`) and memory delta.
        - **Recall@k:** Accuracy measured **relative to the Baseline (Exact Search)**.
        """
    )

    st.subheader("Step 1: Select Reference Paper to Generate Query Vector")

    sample_titles = fetch_titles(limit=100)

    selected_titles = st.multiselect(
        "Select Reference Papers from Dataset:", 
        options=sample_titles,
    )

    manual_titles_text = st.text_area(
        "Or Enter Custom Titles (One per line):", 
        value="",
        height=120,
        help="Manually input arXiv paper titles to generate query vectors." 
    )
    manual_titles = [t.strip() for t in manual_titles_text.splitlines() if t.strip()]

    all_titles = list({t for t in selected_titles + manual_titles if t})

    
    st.write(f"Selected Query Papers: **{len(all_titles)}**")
    if all_titles:
        st.write(all_titles)

    top_k = st.slider("Top-K passages", min_value=5, max_value=50, value=10, step=1)

    if st.button("Run Search Benchmark"):
        if not all_titles:
            st.warning("Action Required: Please select or input at least one paper title to generate query embeddings.")
            return

        with st.spinner("Calling /benchmark..."):
            try:
                data = run_benchmark(all_titles, top_k=top_k)
            except Exception as e:
                st.error(f"Benchmark failed: {e}")
                return

        st.success("Benchmark Execution Completed Successfully!")

        backends = data.get("backends", [])
        if not backends:
            st.info("No reults from backend!")
            return

        # ---- Comparison：latency / memory / recall ----
        rows = []
        for b in backends:
            rows.append({
                "backend": b["backend"],
                "latency_ms": b["elapsed_ms"],
                "extra_mem_mb_per_query": b["memory_delta_mb"],
                "recall_at_k": b["recall_at_k"],
            })

        bench_df = pd.DataFrame(rows)

        st.subheader("Search: Latency / Extra Memory / Recall Summary")

        st.markdown(
            """
            - **latency_ms**: average per-query latency (ms) for this backend.  
            - **extra_mem_mb_per_query**: approximate per-query RSS increase (MB).

            > This value is only meaningful for the NumPy baseline (especially in the cold scenario, where
            > embeddings are reloaded from disk on each query). FAISS Flat / HNSW keep their indices resident
            > in memory, so per-query extra memory is essentially zero.
            """
        )


        st.subheader("Search: Latency / Memory / Recall Summary")
        st.dataframe(bench_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Latency (ms)")
            lat_df = bench_df.set_index("backend")[["latency_ms"]]
            st.bar_chart(lat_df)

        with col2:
            st.markdown("#### Extra memory per query (MB)")
            mem_df = bench_df.set_index("backend")[["extra_mem_mb_per_query"]]
            st.bar_chart(mem_df)

        with col3:
            st.markdown("#### Recall@k")
            rec_df = bench_df.set_index("backend")[["recall_at_k"]]
            st.bar_chart(rec_df)

        st.markdown("---")

        # Show backend search results 
        for b in backends:
            st.subheader(f"{b['backend']} results（top {top_k}）")
            for r in b.get("results", [])[:top_k]:
                header = f"[{b['backend']}] Rank {r['rank']}: {r['title']} (score={r['score']:.4f})"
                with st.expander(header):
                    st.write(f"- **Section**: `{r['section']}`")
                    st.write(f"- **Vector index**: `{r['chunk_index']}` (row={r['chunk_row']})")
                    st.write("---")
                    st.write(r.get("chunk_text", ""))
    # ======================
    # HNSW efSearch Ablation
    # ======================


    st.markdown("---")
    st.header("HNSW Hyperparameter Analysis: `efSearch`")

    # 1. 專業的實驗說明 (Experimental Setup)
    st.markdown(
        """
        This section profiles the **latency-recall trade-off** of the HNSW backend by varying the `efSearch` parameter.
        
        **Experimental Configuration:**
        - **Corpus Size:** `num_docs=10k`
        - **Query Scope:** `top_k=10`
        - **Ground Truth:** Brute-force Flat Index (L2)
        """
    )

    
    hnsw_stats = {
        16:  {"latency_ms": 0.20, "recall": 0.9830},
        32:  {"latency_ms": 0.21, "recall": 0.9925},
        64:  {"latency_ms": 0.26, "recall": 0.9965},
        128: {"latency_ms": 0.36, "recall": 0.9995},
        256: {"latency_ms": 0.58, "recall": 1.0000},
    }#'''
    
    rows = []
    for ef, v in hnsw_stats.items():
        rows.append({"efSearch": ef, "latency_ms": v["latency_ms"], "recall": v["recall"]})
    df = pd.DataFrame(rows).sort_values("efSearch")

    # 推薦的 efSearch
    recommended_ef = 64
    best = hnsw_stats[recommended_ef]

    # --- Chart 1: Latency vs efSearch ---
    st.subheader("1. Query Latency vs. `efSearch`")

    base_lat = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=80, fill="white")) # 加白點讓數據更清晰
        .encode(
            x=alt.X("efSearch:Q", title="HNSW efSearch Parameter", scale=alt.Scale(type="linear")), # 這裡也可以試試 log
            y=alt.Y(
                "latency_ms:Q",
                title="Latency (ms)",
                scale=alt.Scale(domain=[0.15, 0.65]),
                #scale=alt.Scale(domain=[2.00, 22.0]), 
            ),
            tooltip=[
                alt.Tooltip("efSearch", title="efSearch"),
                alt.Tooltip("latency_ms", title="Latency (ms)", format=".3f"),
            ],
        )
        .properties(height=300)
    )

    # 標註推薦點 (Annotation)
    rule_lat = (
        alt.Chart(pd.DataFrame({"efSearch": [recommended_ef]}))
        .mark_rule(strokeDash=[4, 4], color="#FF4B4B") # 使用 Streamlit 紅色強調
        .encode(x="efSearch:Q")
    )

    text_lat = (
        alt.Chart(pd.DataFrame({"efSearch": [recommended_ef], "latency_ms": [best["latency_ms"]]}))
        .mark_text(align="left", dx=10, dy=-10, color="#FF4B4B", fontWeight="bold")
        .encode(
            x="efSearch:Q",
            y="latency_ms:Q",
            text=alt.value(f"Selected: {recommended_ef}")
        )
    )

    st.altair_chart(base_lat + rule_lat + text_lat, use_container_width=True)


    # --- Chart 2: Recall vs efSearch ---
    st.subheader("2. Recall Retention vs. `efSearch`")

    base_rec = (
        alt.Chart(df)
        .mark_line(color="green", point=alt.OverlayMarkDef(size=80, fill="white", color="green"))
        .encode(
            x=alt.X("efSearch:Q", title="HNSW efSearch Parameter"),
            y=alt.Y(
                "recall:Q",
                title="Recall@10 (Relative to Flat)",
                
                scale=alt.Scale(domain=[0.98, 1.0005]),
                #scale=alt.Scale(domain=[0.85, 1.0005]), 
            ),
            tooltip=[
                alt.Tooltip("efSearch", title="efSearch"),
                alt.Tooltip("recall", title="Recall@10", format=".4%"), 
            ],
        )
        .properties(height=300)
    )

    rule_rec = (
        alt.Chart(pd.DataFrame({"efSearch": [recommended_ef]}))
        .mark_rule(strokeDash=[4, 4], color="#FF4B4B")
        .encode(x="efSearch:Q")
    )

    st.altair_chart(base_rec + rule_rec, use_container_width=True)


    st.markdown(f"""
    <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;">
        <h4 style="color: white; margin-top: 0;">💡 Trade-off Analysis & Engineering Decision</h4>
        <p style="color: #dcdcdc; font-size: 16px;">
            The experiment reveals a clear <strong>"Knee Point"</strong> in the performance curve:
        </p>
        <ul style="color: #dcdcdc; font-size: 16px;">
            <li>
                <strong>Latency Cost:</strong> Latency grows linearly from <strong>0.20ms</strong> to <strong>0.58ms</strong> as <code>efSearch</code> increases.
            </li>
            <li>
                <strong>Recall Gain:</strong> Recall steadily climbs from <strong>98.3%</strong>, reaching saturation (100%) at <code>efSearch=256</code>.
            </li>
        </ul>
        <p style="color: #dcdcdc; font-size: 16px;">
            <strong>Decision:</strong> We selected <code>efSearch={recommended_ef}</code> as the production default. 
            At this point, we achieve <strong style="color: #FF4B4B;">{best['recall']:.2%} recall</strong> with a negligible latency of 
            <strong style="color: #FF4B4B;">{best['latency_ms']:.2f} ms</strong>. 
            Pushing beyond this point yields <strong style="color: #FF4B4B;">diminishing returns</strong>, 
            where a < 0.5% gain in accuracy costs a > 60% increase in latency.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")
    st.header("Build-time Benchmark (CPU vs GPU)")

    st.markdown(
        """
        This section visualizes the **scaling behavior** of the indexing pipeline as the corpus size (**num_docs**) increases:

        - ** CPU (Baseline):** Full-stack CPU execution (Embedding Generation + Vector Indexing).
        - ** GPU (Accelerated):** End-to-end GPU acceleration using CUDA for both embeddings and index construction.

        **Experimental Configuration:** `batch_size=512`, `max_chars_per_chunk=2000`.
        """
    )

    # Actual results got by running cli_arxiv.py bench-build mode
    build_stats = {
        1000:  {"cpu": 80.91, "gpu": 4.79},
        5000:  {"cpu": 310.91, "gpu": 9.76},
        10000: {"cpu": 364.95, "gpu": 11.32},
        50000: {"cpu": 1790.22, "gpu": 43.79}
    }


    rows = []
    for nd, v in build_stats.items():
        rows.append({"num_docs": nd, "backend": "CPU", "time_sec": v["cpu"]})
        rows.append({"num_docs": nd, "backend": "GPU", "time_sec": v["gpu"]})
    df = pd.DataFrame(rows)

    # -------- 1) Build time vs num_docs (log scale，Altair) --------
    st.subheader("Build time vs num_docs（log scale）")

    chart_time = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_docs:Q", title="num_docs"),
            y=alt.Y(
                "time_sec:Q",
                title="Build time (sec)",
                scale=alt.Scale(type="log")  
            ),
            color=alt.Color("backend:N", title="Backend"),
            tooltip=["num_docs", "backend", alt.Tooltip("time_sec:Q", format=".2f")],
        )
        .properties(height=320)
    )

    st.altair_chart(chart_time, use_container_width=True)

    # -------- 3) Speedup curve (CPU time / GPU time) --------
    st.subheader("Speedup (CPU time / GPU time)")

    df_speedup = pd.DataFrame(
        {
            "num_docs": list(build_stats.keys()),
            "speedup": [v["cpu"] / v["gpu"] for v in build_stats.values()],
        }
    )

    chart_speedup = (
        alt.Chart(df_speedup)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_docs:Q", title="num_docs"),
            y=alt.Y("speedup:Q", title="Speedup (CPU / GPU)"),
            tooltip=["num_docs", alt.Tooltip("speedup:Q", format=".2f")],
        )
        .properties(height=260)
    )

    st.altair_chart(chart_speedup, use_container_width=True)

    
    last_nd = max(build_stats.keys())
    cpu_last = build_stats[last_nd]["cpu"]
    gpu_last = build_stats[last_nd]["gpu"]
    speedup_last = cpu_last / gpu_last

    st.subheader(f"🚀 Scalability Analysis (N={last_nd:,} Vectors)")

    # 1. Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="🐢 CPU Latency", value=f"{cpu_last:.2f} s")
    with col2:
        st.metric(
            label="⚡ GPU Latency", 
            value=f"{gpu_last:.2f} s", 
            delta=f"-{cpu_last - gpu_last:.2f} s (Saved)"
        )
    with col3:
        st.metric(
            label="🔥 Speedup Factor", 
            value=f"{speedup_last:.1f}x", 
            delta="High Efficiency"
        )

    # 2. Engineering Insight
    st.markdown("""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B;">
            <h4 style="color: white; margin-top: 0;"> System Insight</h4>
            <p style="color: #dcdcdc; font-size: 16px;">
                The trend indicates that while CPU latency grows linearly with data size, the GPU acceleration effectively 
                <strong style="color: #FF4B4B;">amortizes</strong> the indexing cost.
            </p>
            <p style="color: #dcdcdc; font-size: 16px;">
                The speedup factor stabilizes at scale, demonstrating that the <strong style="color: #FF4B4B;">massive parallelism</strong> 
                of CUDA cores is fully utilized to overcome the computational bottleneck. This suggests that for production RAG systems, 
                we can significantly expand the <strong>Knowledge Base</strong> size without incurring the prohibitive latency penalties seen in CPU-based solutions.
            </p>
        </div>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
