# arxiv_faiss_zarr/bench_build.py
import datetime
import time
import json
import torch

from .build_index import build_index


def benchmark_build(csv_path: str,
                    output_dir_base: str,
                    num_docs: int,
                    batch_size: int,
                    max_chars_per_chunk: int,
                    json_out: str):
    '''
    The functions is used for generating benchmark results for building 
    the dataset with CPU or GPU. 
    Args:
            csv_path (str): Path to the raw data CSV (e.g., 'arxiv_cornell_title_abstract.csv').
            output_dir_base (str): The generated FAISS indices and Zarr artifacts will be saved 
                                   in output_dir_base+'_cpu' and output_dir_base+'_gpu'
            num_docs (int): Number of docs that will be built in the index, 0 indicates the full dataset.
            batch_size (int): batch_size (int): The number of texts to process in parallel. 
                            Higher values improve inference speed (throughput) but require more 
                            GPU/CPU memory. If OOM errors occur, reduce this value.
                            Defaults to 256.
            max_chars_per_chunk (int): The maximum number of characters for each text chunk.
                                    This should be strictly less than the model's token limit.
            json_out (str): The output path of result json.

    Returns:
            None: The function writes a JSON file to `json_path` containing:
                - Metadata (csv_path, num_docs, batch_size, timestamp)
                - 'cpu': Execution stats for CPU-only run (time, output_dir).
                - 'gpu': Execution stats for GPU run (time, gpu_name).
    '''
    metrics = {
        "csv_path": csv_path,
        "num_docs": num_docs,
        "batch_size": batch_size,
        "max_chars_per_chunk": max_chars_per_chunk,
        "timestamp": datetime.datetime.now().isoformat(),
        "cpu": {},
        "gpu": {},
    }

    # CPU
    out_cpu = output_dir_base + "_cpu"
    print(f"[BENCH] CPU build -> {out_cpu}")
    t0 = time.perf_counter()
    build_index(
        csv_path=csv_path,
        output_dir=out_cpu,
        num_docs=num_docs,
        batch_size=batch_size,
        max_chars_per_chunk=max_chars_per_chunk,
        use_gpu=False,
        build_hnsw=False,
    )
    t1 = time.perf_counter()
    metrics["cpu"] = {
        "use_gpu": False,
        "total_time_sec": t1 - t0,
        "output_dir": out_cpu,
    }

    # GPU
    out_gpu = output_dir_base + "_gpu"
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"[BENCH] GPU build -> {out_gpu} (cuda_available={cuda_available})")
    t2 = time.perf_counter()
    build_index(
        csv_path=csv_path,
        output_dir=out_gpu,
        num_docs=num_docs,
        batch_size=batch_size,
        max_chars_per_chunk=max_chars_per_chunk,
        use_gpu=True,
        build_hnsw=False,
    )
    t3 = time.perf_counter()
    metrics["gpu"] = {
        "use_gpu": True,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "total_time_sec": t3 - t2,
        "output_dir": out_gpu,
    }

    print(f"[BENCH] Writing result to {json_out}")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
