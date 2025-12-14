# arxiv_faiss_zarr/build.py
import os
import time

import numpy as np
import zarr
import faiss
from .config import CHUNK_SIZE, DTYPE, DEFAULT_MAX_CHARS_PER_CHUNK
from .load_data import load_arxiv_df, save_metadata_jsonl
from .chunking import build_chunks_from_df
from .embeddings import choose_device, load_model, embed_chunks
from .faiss_index import build_flat_index, build_hnsw_index


def build_index(csv_path: str,
                output_dir: str,
                num_docs: int,
                batch_size: int,
                max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
                use_gpu: bool = False,
                build_hnsw: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    zarr_path = os.path.join(output_dir, "embeddings.zarr")
    flat_path = os.path.join(output_dir, "faiss_index_flat.bin")
    hnsw_path = os.path.join(output_dir, "faiss_index_hnsw.bin")
    meta_path = os.path.join(output_dir, "metadata.jsonl")

    # 1) load csv
    df = load_arxiv_df(csv_path, num_docs=num_docs)

    # 2) chunk
    chunk_texts, chunk_metadata = build_chunks_from_df(df, max_chars_per_chunk=max_chars_per_chunk)
    if not chunk_texts:
        raise RuntimeError("No chunks produced.")

    # 3) device + model
    device = choose_device(use_gpu)
    model = load_model(device=device)

    # 4) embed all chunks
    emb = embed_chunks(model, chunk_texts, batch_size=batch_size)
    n, dim = emb.shape

    # 5) write Zarr
    print(f"[INFO] Writing embeddings to Zarr: {zarr_path}")
    root = zarr.open(zarr_path, mode="w")
    emb_store = root.create_dataset(
        "embeddings",
        shape=(n, dim),
        chunks=(CHUNK_SIZE, dim),
        dtype=DTYPE
    )
    emb_store[:] = emb

    # 6) build flat index
    t0 = time.perf_counter()
    index_flat = build_flat_index(emb_store, dim)
    elapsed_flat = time.perf_counter() - t0
    print(f"[INFO] Flat index build time = {elapsed_flat:.2f} s")
    print(f"[INFO] Saving flat index to {flat_path}")
    faiss.write_index(index_flat, flat_path)

    # 7) build HNSW (optional)
    if build_hnsw:
        t1 = time.perf_counter()
        index_hnsw = build_hnsw_index(emb_store, dim)
        elapsed_hnsw = time.perf_counter() - t1
        print(f"[INFO] HNSW index build time = {elapsed_hnsw:.2f} s")
        print(f"[INFO] Saving HNSW index to {hnsw_path}")
        faiss.write_index(index_hnsw, hnsw_path)

    # 8) metadata
    print(f"[INFO] Saving metadata to {meta_path}")
    save_metadata_jsonl(chunk_metadata, meta_path)

    print("[INFO] Build finished.")
