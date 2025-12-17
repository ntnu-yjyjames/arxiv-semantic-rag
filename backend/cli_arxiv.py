# cli_arxiv.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

from indexing.build_index import build_index
from indexing.bench_build import benchmark_build


import argparse
import sys
import traceback
# 假設你的核心函數 import 都在這裡
# from your_module import build_index, benchmark_build 

def main():
    """
    Main entry point for the ArXiv RAG Index Builder CLI.

    Supports two primary modes:
    1. `build`: Production-ready pipeline to process CSV -> Embeddings -> FAISS Index.
    2. `bench-build`: Performance testing mode to compare CPU vs. GPU build times.
    """
    parser = argparse.ArgumentParser(
        description="ArXiv RAG Engine: High-Performance Vector Index Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # ==========================================
    # Command: build
    # ==========================================
    build_p = subparsers.add_parser(
        "build", 
        help="Run the full ETL pipeline: CSV -> Zarr -> FAISS Index.",
        description="Extracts text, computes embeddings (GPU/CPU), and builds Flat+HNSW indices."
    )
    build_p.add_argument(
        "--csv-path", type=str, required=True, 
        help="Path to the source arXiv metadata CSV file."
    )
    build_p.add_argument(
        "--output-dir", type=str, required=True, 
        help="Directory to save artifacts (embeddings.zarr, faiss_index_flat.bin, faiss_index_hnsw.bin, metadata.jsonl)."

    )
    build_p.add_argument(
        "--num-docs", type=int, default=0, 
        help="Number of documents to process. Set to 0 for FULL dataset."
    )
    build_p.add_argument(
        "--batch-size", type=int, default=256, 
        help="Inference batch size. Increase for higher throughput if VRAM allows."
    )
    build_p.add_argument(
        "--max-chars-per-chunk", type=int, default=2000, 
        help="Soft character limit for text chunking before embedding."
    )
    build_p.add_argument(
        "--use-gpu", action="store_true", 
        help="Enable GPU acceleration for the embedding model (SentenceTransformer)."
    )
    build_p.add_argument(
        "--no-hnsw", action="store_true", 
        help="Skip HNSW index construction (Flat index only). Useful for quick debugging."
    )

    # ==========================================
    #  Command: bench-build
    # ==========================================
    bench_p = subparsers.add_parser(
        "bench-build", 
        help="Benchmark mode: Compare CPU vs. GPU build-time for the embedding + index construction pipeline.",
        description="Runs the build pipeline twice (CPU run vs GPU run) and exports metrics."
    )
    bench_p.add_argument(
        "--csv-path", type=str, required=True, 
        help="Path to input CSV."
    )
    bench_p.add_argument(
        "--output-dir-base", type=str, required=True, 
        help="Base path for temporary benchmark outputs."
    )
    bench_p.add_argument(
        "--num-docs", type=int, default=1000, 
        help="Number of documents for the dry-run (keep small for quick benchmarking)."
    )
    bench_p.add_argument(
        "--batch-size", type=int, default=256, 
        help="Batch size for throughput comparison."
    )
    bench_p.add_argument(
        "--max-chars-per-chunk", type=int, default=2000, 
        help="Chunking character limit."
    )
    bench_p.add_argument(
        "--json-out", type=str, default="build_benchmark.json", 
        help="Filename to save the benchmark results (JSON)."
    )

    args = parser.parse_args()

    # Dispatch Logic
    try:
        if args.command == "build":
            print(f"[{args.command.upper()}] Starting pipeline...")
            build_index(
                csv_path=args.csv_path,
                output_dir=args.output_dir,
                num_docs=args.num_docs,
                batch_size=args.batch_size,
                max_chars_per_chunk=args.max_chars_per_chunk,
                use_gpu=args.use_gpu,
                build_hnsw=not args.no_hnsw,
            )
        elif args.command == "bench-build":
            print(f"[{args.command.upper()}] Starting benchmark comparison...")
            benchmark_build(
                csv_path=args.csv_path,
                output_dir_base=args.output_dir_base,
                num_docs=args.num_docs,
                batch_size=args.batch_size,
                max_chars_per_chunk=args.max_chars_per_chunk,
                json_out=args.json_out,
            )
            
    except Exception as e:
        print(f"\n❌ Critical Error in '{args.command}' command:")
        print(f"   {e}")
        # traceback.print_exc() # Uncomment for debugging
        sys.exit(1) # Return non-zero exit code to signal failure


if __name__ == "__main__":
    main()
