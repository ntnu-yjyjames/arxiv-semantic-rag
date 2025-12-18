# pages/02_RAG_chat.py
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

DEMO_QUESTIONS = [
    "What are the common NLP tasks where Transformer models are applied in these papers?",
    "According to these papers, what are the advantages of Transformers over RNNs in language modeling?",
    "What methods do these papers propose to improve long-sequence processing?",
    "Do any papers apply Transformers to tasks other than machine translation? Please provide examples and brief explanations.",
    "What are the common directions for model improvement discussed in these Transformer-related papers?",
]

st.set_page_config(
    page_title="Chat with arXiv (RAG)",
    layout="wide",
)


def main():
    """
    RAG Inference Workbench: a configurable environment for retrieval-augmented QA.

    This page serves as the engineering interface for the Retrieval-Augmented Generation
    (RAG) pipeline. It follows a "glass box" approach, allowing researchers to tune
    retrieval / generation hyperparameters and inspect the evidence used for answers.

    **Key Features:**
    1. **Query Input:** Supports both manual entry and pre-defined demo questions to
       quickly test domain-specific knowledge.
    2. **Parameter Tuning:** Exposes
       - `top_k` (retrieval depth; number of context passages to fetch), and
       - `max_tokens` (maximum generation length)
       for real-time prompt / RAG behavior experiments.
    3. **Evidence Visualization:** Renders the retrieved contexts (source chunks) below
       the answer, which is critical for checking hallucination vs. grounding.

    **Backend Interaction:**
        - POST `{BACKEND_URL}/rag-answer`
          The backend:
            1) encodes the question and performs vector search over arXiv passages, and
            2) calls a local LLM (e.g., Llama 3.1 8B via Ollama) with the retrieved
               passages injected as context to generate the final answer.
    """
    st.title("Chat with arXiv (RAG)")

    st.markdown(
        """
        **RAG Inference Pipeline:**
        
        1.  **Vector Retrieval:** Semantic search against the Transformer arXiv corpus to retrieve relevant text passages.
        2.  **Context Injection:** Augments the user prompt with retrieved passages as ground truth.
        3.  **Local Inference:** Generates response using **Llama-3-8B-Instruct** (Running locally via Ollama).
        """
    )


    demo_q = st.selectbox(
        "Select Sample Query (Auto-fill):",
        options=["(Manual Input)"] + DEMO_QUESTIONS,
        index=0,
        help="Select a pre-defined query to test the system's domain knowledge."
    )

    
    default_text = "" if demo_q == "(Manual Input)" else demo_q
    question = st.text_area(
        "Input Query:",
        value=default_text,
        height=120,
        placeholder="E.g., How does the attention mechanism differ from RNNs?"
    )

    # Hyperparameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "Retrieval Depth (Top-K papers)", 
            min_value=3, max_value=20, value=8, step=1,
            help="Number of context passages retrieved from the vector database to inject into the LLM prompt."
        )
    with col2:
        max_tokens = st.slider(
            "Max Generation Tokens", 
            min_value=128, max_value=1024, value=512, step=64,
            help="Maximum length of the generated response to control latency."
        )

    
    if st.button("Execute RAG Pipeline"):
        if not question.strip():
            st.warning("Action Required: Query input cannot be empty.")
            return
        

        with st.spinner("Retrieving & generating..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k,
                    "max_tokens": max_tokens,
                }
                resp = requests.post(f"{BACKEND_URL}/rag-answer", json=payload, timeout=600)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                # 錯誤處理：使用 "Execution Failed" 強調是執行層面的錯誤
                st.error(f"RAG Pipeline Execution Failed: {e}")
                return

            st.success("Inference Complete!")

            st.subheader("Generated Response")
            st.write(data.get("answer", ""))

        
            st.subheader("Retrieved Contexts (Grounding Source)")

        contexts = data.get("contexts", [])
        for c in contexts:
            header = f"[DOC {c['doc_idx']}] {c['title']}"
            with st.expander(header):
                st.write(c.get("chunk_text", ""))


if __name__ == "__main__":
    main()
