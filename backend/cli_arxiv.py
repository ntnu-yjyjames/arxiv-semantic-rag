# cli_arxiv.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

from arxiv_faiss_zarr.build_index import build_index
from arxiv_faiss_zarr.bench_build import benchmark_build



def main():
    parser = argparse.ArgumentParser(
        description="ArXiv vector index builder / benchmark CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = subparsers.add_parser("build", help="Build embeddings + FAISS index")
    build_p.add_argument("--csv-path", type=str, required=True)
    build_p.add_argument("--output-dir", type=str, required=True)
    build_p.add_argument("--num-docs", type=int, default=0)
    build_p.add_argument("--batch-size", type=int, default=256)
    build_p.add_argument("--max-chars-per-chunk", type=int, default=2000)
    build_p.add_argument("--use-gpu", action="store_true")
    build_p.add_argument("--no-hnsw", action="store_true")

    # bench-build
    bench_p = subparsers.add_parser("bench-build", help="Benchmark CPU vs GPU build")
    bench_p.add_argument("--csv-path", type=str, required=True)
    bench_p.add_argument("--output-dir-base", type=str, required=True)
    bench_p.add_argument("--num-docs", type=int, default=1000)
    bench_p.add_argument("--batch-size", type=int, default=256)
    bench_p.add_argument("--max-chars-per-chunk", type=int, default=2000)
    bench_p.add_argument("--json-out", type=str, default="build_benchmark.json")

    args = parser.parse_args()

    if args.command == "build":
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
        benchmark_build(
            csv_path=args.csv_path,
            output_dir_base=args.output_dir_base,
            num_docs=args.num_docs,
            batch_size=args.batch_size,
            max_chars_per_chunk=args.max_chars_per_chunk,
            json_out=args.json_out,
        )


if __name__ == "__main__":
    main()
