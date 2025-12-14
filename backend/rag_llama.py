# rag_ollama.py
from typing import List, Dict
import ollama

OLLAMA_MODEL = "llama3.1:8b"  # 或 "llama3.1-q5km-local" 如果你建立了自訂模型


def build_context_from_chunks(chunks: List[Dict], max_chars: int = 3000) -> str:
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
    用 Ollama 上的 Llama3.1 8B 模型 + RAG context 回答問題。
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
