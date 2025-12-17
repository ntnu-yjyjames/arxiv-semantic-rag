# rag_ollama.py
from typing import List, Dict
import ollama

OLLAMA_MODEL = "llama3.1:8b"  


def build_context_from_chunks(chunks: List[Dict], max_chars: int = 3000) -> str:
    """
    Constructs a structured context string from retrieved chunks for the LLM prompt.

    This function formats raw retrieval results into a readable "Context Block" with 
    explicit citation markers (Title, Section). It implements a hard character limit 
    to ensure the resulting prompt fits within the LLM's context window.

    Format per chunk:
    [DOC {id}] [SECTION: {name}]
    TITLE: {title}
    CONTENT: {text}

    Args:
        chunks (List[Dict]): The top-k retrieval results.
        max_chars (int): It applies a hard **character** limit to keep the assembled context within
                         a safe range for the LLM's context window (character-based proxy for token length).

    Returns:
        str: A single aggregated string ready for injection into the system prompt.
    """
    parts = []
    total_len = 0
    for c in chunks:
        title = c.get("title", "")
        section = c.get("section", "")
        text = c.get("chunk_text", "") or c.get("text_preview", "")
        block = f"[DOC {c['doc_idx']}] [SECTION: {section}]\nTITLE: {title}\nCONTENT:\n{text}\n"
        if total_len + len(block) > max_chars:
            break
        parts.append(block)
        total_len += len(block)
    return "\n\n".join(parts)


def generate_rag_answer(
    question: str,
    chunks: List[Dict],
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> str:
    """
    Synthesizes a grounded answer using a local LLM (Llama 3.1 via Ollama).

    This function constructs a system prompt that instructs the LLM to
    answer based only on the provided context (grounding) and to explicitly
    admit uncertainty if the context is insufficient. This helps to minimize
    unsupported hallucinations.

    Args:
        question (str): The user's natural language query.
        chunks (List[Dict]): Retrieved context chunks acting as the knowledge source.
        max_tokens (int): Approximate upper bound on the generated response length (in tokens).
        temperature (float): Controls randomness (lower values produce more focused, less random outputs).

    Returns:
        str: The generated answer text.
    """
    context = build_context_from_chunks(chunks)

    system_prompt = (
        "You are a helpful research assistant. "
        "Use only the given arXiv paper excerpts to answer the user's question. "
        "If the context is insufficient, say you are not sure. "
        "When helpful, mention the paper titles or sections explicitly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Here is some context from arXiv papers:\n\n{context}\n\nUser question: {question}",
        },
    ]

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    answer = resp["message"]["content"]
    return answer.strip()
