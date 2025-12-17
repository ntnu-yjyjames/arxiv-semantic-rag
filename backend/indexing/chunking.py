# arxiv_faiss_zarr/chunking.py
import re
from typing import List, Tuple, Dict, Optional

import pandas as pd

from .config import SECTION_TITLES, SENTENCE_ENDINGS, FULLTEXT_CANDIDATES, DEFAULT_MAX_CHARS_PER_CHUNK


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Parse the raw text and segment it into logical sections based on simple
    arXiv-style headings (e.g., 'Abstract', 'Introduction', 'Methods', ...).

    Args:
        text (str): Raw string content of the paper (e.g., extracted from PDF or LaTeX).

    Returns:
        List[Tuple[str, str]]: A list of (section_label, section_text) pairs where:
            - section_label: a normalized, lowercased section name
              (e.g. "abstract", "introduction", "methods", "body", "preamble").
            - section_text: the substring of the original text corresponding to that
              section, starting from the detected heading line (i.e., it usually
              **includes** the section header line).
    """
    if not text or not isinstance(text, str):
        return []

    pattern = r"\n\s*(\d{0,2}\.?\s*)?(?P<title>" + "|".join(
        [re.escape(t) for t in SECTION_TITLES]
    ) + r")\s*\n"

    regex = re.compile(pattern, flags=re.IGNORECASE)
    matches = list(regex.finditer(text))

    if not matches:
        return [("body", text)]

    sections: List[Tuple[str, str]] = []

    first_start = matches[0].start()
    if first_start > 0:
        pre = text[:first_start].strip()
        if pre:
            sections.append(("preamble", pre))

    for i, m in enumerate(matches):
        title = m.group("title").lower()
        sec_start = m.start()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[sec_start:sec_end].strip()
        sections.append((title, sec_text))

    return sections


def find_last_sentence_boundary(chunk: str) -> int:
    """
    Finds the safe cutting point to prevent splitting a sentence in the middle.

    Used during chunking to identify the last complete sentence boundary within 
    the character limit. This ensures semantic integrity by keeping sentences intact.

    Args:
        chunk (str): The candidate text chunk.

    Returns:
        int: The index of the last punctuation (., ?, !).
             Returns -1 if the chunk contains no sentence endings (implies a hard cut is needed).
    """
    last_pos = -1
    for end in SENTENCE_ENDINGS:
        pos = chunk.rfind(end)
        if pos > last_pos:
            last_pos = pos
    return last_pos


def chunk_text_by_length(text: str,
                         max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK
                         ) -> List[str]:
    """
    Chunks text by length with a "soft truncation" strategy to preserve sentence integrity.

    The function attempts to break at the last sentence-ending punctuation within the 
    window. If no suitable boundary is found, or if the resulting chunk is too short 
    (< 30% of max_chars), it falls back to a hard cut at `max_chars`.

    Args:
        text (str): The input text string.
        max_chars (int): Maximum characters per chunk.

    Returns:
        List[str]: A list of processed chunks suitable for embedding models.
    """
    # ...
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        window = text[start:end]
        split_rel = find_last_sentence_boundary(window)

        if split_rel == -1 or (start + split_rel + 1 - start) < max_chars * 0.3:
            cut = end
        else:
            cut = start + split_rel + 1

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        if cut >= n:
            break
        start = cut

    return chunks


def build_chunks_from_df(df: pd.DataFrame,
                         max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK
                         ) -> Tuple[List[str], List[Dict]]:
    """
    Transforms raw document data into semantically enriched chunks for the RAG pipeline.

    Key Features:
    - **Hybrid Chunking:** Combines semantic section splitting with hard length constraints.
    - **Context Injection:** Prepends "{Title}\n[SECTION: {Name}]" to every chunk to preserve global context.
    - **Graceful Fallback:** Automatically detects if full-text is missing and defaults to processing the abstract.

    Args:
        df (pd.DataFrame): Source data containing paper metadata and content.
        max_chars_per_chunk (int): Maximum number of **characters** allowed per chunk (a rough safety limit for downstream token length).

    Returns:
        Tuple[List[str], List[Dict]]: 
            `chunk_texts`: each string has the form "{title}\\n[SECTION: {section_name}]\\n{chunk_text}".
            `chunk_metadata`: per-chunk metadata including doc_idx, id, title, categories,
                                section, chunk_index, chunk_text, and a short preview.
    """
    fulltext_col: Optional[str] = None
    for c in FULLTEXT_CANDIDATES:
        if c in df.columns:
            fulltext_col = c
            break

    if fulltext_col:
        df[fulltext_col] = df[fulltext_col].fillna("")

    chunk_texts: List[str] = []
    chunk_metadata: List[Dict] = []

    print("[INFO] Building chunks (section + length-based) ...")

    for doc_idx, row in df.iterrows():
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        arxiv_id = row.get("id", str(doc_idx))
        categories = row.get("categories", "")

        base_meta = {
            "doc_idx": int(doc_idx),
            "id": arxiv_id,
            "title": title,
            "categories": categories,
        }

        if fulltext_col and isinstance(row[fulltext_col], str) and row[fulltext_col].strip():
            full_text = row[fulltext_col]
            sections = split_into_sections(full_text)
        else:
            sections = [("abstract", abstract)]

        for sec_name, sec_text in sections:
            if not sec_text:
                continue

            sec_chunks = chunk_text_by_length(sec_text, max_chars=max_chars_per_chunk)

            for chunk_index, chunk_text in enumerate(sec_chunks):
                text_for_emb = f"{title}\n[SECTION: {sec_name}]\n{chunk_text}"

                preview = chunk_text.replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "..."

                meta = {
                    **base_meta,
                    "section": sec_name,
                    "chunk_index": int(chunk_index),
                    "chunk_text": chunk_text,
                    "text_preview": preview,
                }

                chunk_texts.append(text_for_emb)
                chunk_metadata.append(meta)

        if (doc_idx + 1) % 1000 == 0:
            print(f"[INFO] Processed {doc_idx + 1} docs, total chunks so far = {len(chunk_texts)}")

    print(f"[INFO] Finished chunking: {len(df)} docs -> {len(chunk_texts)} chunks.")
    return chunk_texts, chunk_metadata
