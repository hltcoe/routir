#!/usr/bin/env python3
"""
FAISS Index Builder

This script builds a FAISS index from embedding vectors stored in .npy files.
It supports GPU-accelerated training, product quantization, and various index types.

The script expects input files in .npy format where each file contains a dictionary with:
- 'features': 2D numpy array of embeddings (shape: [n_docs, embedding_dim])
- 'ids': List of document IDs corresponding to each embedding

Usage:
    python faiss_indexing.py <input_dir> <output_dir> [options]

Example:
    python faiss_indexing.py ./embeddings ./index --index_string "IVF4096,PQ64" --use_gpu
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
from tqdm.auto import tqdm


faiss.omp_set_num_threads(32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from embedding vectors for efficient similarity search.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage with default PQ index:
    python faiss_indexing.py ./embeddings ./index

  Using GPU for faster training:
    python faiss_indexing.py ./embeddings ./index --use_gpu

  Custom IVF+PQ index with higher sampling:
    python faiss_indexing.py ./embeddings ./index --index_string "IVF4096,PQ128" --sampling_rate 0.1

Index String Examples:
  - "Flat": Exact search (no compression)
  - "PQ64": Product Quantization with 64 codes
  - "IVF4096,PQ64": Inverted file with 4096 centroids + PQ compression
  - "IVF4096,Flat": Inverted file with exact vectors (faster than Flat for large datasets)

For more index types, see: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
        """
    )

    # Positional arguments
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing *.npy files with embeddings. Each .npy file must be a dictionary "
             "with 'features' (2D numpy array of embeddings) and 'ids' (list of document IDs). "
             "The first dimension of 'features' must match the length of 'ids'."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory where the FAISS index and document IDs will be saved. "
             "Creates 'index.faiss' (the FAISS index) and 'index.ids' (document ID mapping)."
    )

    # Index configuration
    parser.add_argument(
        "--index_string",
        type=str,
        default="PQ2048x4fs",
        help="FAISS index factory string specifying the index type and parameters. "
             "Default: 'PQ2048x4fs' (Product Quantization with 2048 centroids, 4-bit codes). "
             "Common options: 'Flat' (exact), 'PQ64', 'IVF4096,PQ64', 'IVF4096,Flat'. "
             "See FAISS documentation for more index types."
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=0.07,
        help="Fraction of data to use for training the index (0.0-1.0). "
             "Default: 0.07 (7%%). Higher values improve index quality but increase training time. "
             "Recommended: 0.05-0.1 for large datasets, 0.1-0.5 for smaller datasets."
    )

    # GPU options
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU for index training. Significantly speeds up training for large datasets. "
             "Requires FAISS to be compiled with GPU support and CUDA to be available."
    )
    parser.add_argument(
        "--two_step_training",
        action="store_true",
        default=False,
        help="Use two-step training for IVF indexes on GPU (trains clustering on GPU, quantization on CPU). "
             "Only effective when --use_gpu is enabled and using IVF-based indexes. "
             "Can reduce memory usage for very large datasets."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fns = list(Path(args.input_dir).glob("*.npy"))

    sampled_fns = all_fns[:: int(1 / args.sampling_rate)]
    sampled_vectors = np.concatenate([np.load(fn, allow_pickle=True).item()["features"] for fn in tqdm(sampled_fns)], axis=0)

    # drop example with na
    sampled_vectors = sampled_vectors[~np.isnan(sampled_vectors).any(axis=1)]

    index = faiss.index_factory(sampled_vectors.shape[1], args.index_string, faiss.METRIC_INNER_PRODUCT)

    if args.use_gpu:
        if args.two_step_training:
            index_ivf = faiss.extract_index_ivf(index)
            clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(index_ivf.d))
            index_ivf.clustering_index = clustering_index
        else:
            co = faiss.GpuMultipleClonerOptions()
            co.allowCpuCoarseQuantizer = True
            index = faiss.index_cpu_to_all_gpus(index)

    print("training...")
    index.train(sampled_vectors)

    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)

    docids = []
    for fn in tqdm(all_fns, desc="adding", dynamic_ncols=True):
        shard = np.load(fn, allow_pickle=True).item()
        # dropna features
        mask = ~np.isnan(shard["features"]).any(axis=1)
        features = shard["features"][mask]
        ids = np.array(shard["ids"])[mask].tolist()

        index.add(features)
        docids += ids

    print("saving faiss index")
    faiss.write_index(index, str(output_dir / "index.faiss"))

    print("saving doc ids")
    with (output_dir / "index.ids").open("w") as fw:
        for docid in docids:
            fw.write(f"{docid}\n")
