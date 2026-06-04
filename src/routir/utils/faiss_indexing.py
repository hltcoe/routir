#!/usr/bin/env python3
"""
FAISS Index Builder

This script builds a FAISS index from embedding vectors stored on disk.
It supports GPU-accelerated training, product quantization, and various index types.

Two input layouts are supported and auto-detected from the directory contents:

1. ``*.npy`` files, where each file contains a dictionary with:
   - 'features': 2D numpy array of embeddings (shape: [n_docs, embedding_dim])
   - 'ids': List of document IDs corresponding to each embedding

2. ``*.tar`` WebDataset shards (e.g. the qwen3_vl_2b embeddings), where each tar
   contains many ``*.npz`` files (one per video). Each ``.npz`` holds:
   - 'embeddings': 2D numpy array of embeddings (shape: [n_keyframes, embedding_dim])
   - 'keyframe_ids': 1D array of keyframe IDs (one per embedding row)
   The per-keyframe embeddings are mean-pooled into a single embedding per video,
   and the document ID written is the ``.npz`` stem (the video ID).

Usage:
    python faiss_indexing.py <input_dir> <output_dir> [options]

Example:
    python faiss_indexing.py ./embeddings ./index --index_string "IVF4096,PQ64" --use_gpu
"""

import argparse
import io
import re
import tarfile
from pathlib import Path

import faiss
import numpy as np
from tqdm.auto import tqdm


faiss.omp_set_num_threads(32)


def load_npy_shard(path):
    """Load a single ``.npy`` dict shard -> (features [N, dim], ids list)."""
    shard = np.load(path, allow_pickle=True).item()
    return shard["features"], list(shard["ids"])


def load_tar_shard(path, npz_id_strip_regex=None):
    """Load all ``.npz`` members of a WebDataset tar -> (features [N, dim], ids list).

    Each ``.npz`` member holds the per-keyframe embeddings of a single video. The
    keyframe embeddings are mean-pooled (and re-normalized, since the keyframe
    vectors are unit-norm) into a single embedding per video, with the document ID
    set to the ``.npz`` member stem, optionally normalized by ``npz_id_strip_regex``.
    """
    strip_re = None
    if npz_id_strip_regex:
        strip_re = re.compile(npz_id_strip_regex) if isinstance(npz_id_strip_regex, str) else npz_id_strip_regex

    features = []
    ids = []
    with tarfile.open(path, "r") as tar:
        for member in sorted(tar.getmembers(), key=lambda member: member.name):
            if not (member.isfile() and member.name.endswith(".npz")):
                continue
            with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=True) as npz:
                embeddings = npz["embeddings"]
            # mean-pool across keyframes -> one embedding per video, then re-normalize
            video_embedding = embeddings.mean(axis=0)
            norm = np.linalg.norm(video_embedding)
            if norm > 0:
                video_embedding = video_embedding / norm
            features.append(video_embedding.astype(np.float32))
            doc_id = Path(member.name).stem
            if strip_re:
                doc_id = strip_re.sub("", doc_id)
            ids.append(doc_id)
    if not features:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(features, axis=0), ids


def wrap_loader(load_shard, npz_id_strip_regex=None):
    """Apply tar-specific loader options while keeping ``*.npy`` behavior unchanged."""
    if not npz_id_strip_regex or load_shard is not load_tar_shard:
        return load_shard

    strip_re = re.compile(npz_id_strip_regex)

    def load_tar_shard_with_options(path):
        return load_tar_shard(path, npz_id_strip_regex=strip_re)

    return load_tar_shard_with_options


def discover_shards(input_dir):
    """Return (sorted shard paths, loader fn) auto-detecting the input layout."""
    input_dir = Path(input_dir)

    npy_fns = sorted(input_dir.glob("*.npy"))
    if npy_fns:
        return npy_fns, load_npy_shard

    tar_fns = sorted(input_dir.glob("*.tar"))
    if tar_fns:
        return tar_fns, load_tar_shard

    raise FileNotFoundError(f"No *.npy or *.tar embedding shards found in {input_dir}")


def filter_valid_features(features):
    """Return finite feature rows as float32, dropping rows with NaN values."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    if features.shape[0] == 0:
        return features.astype(np.float32, copy=False)
    mask = ~np.isnan(features).any(axis=1)
    return features[mask].astype(np.float32, copy=False)


def filter_valid_features_and_ids(features, ids):
    """Drop NaN feature rows and keep ids aligned."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    if features.shape[0] != len(ids):
        raise ValueError(f"Feature/id count mismatch: {features.shape[0]} features, {len(ids)} ids")
    if features.shape[0] == 0:
        return features.astype(np.float32, copy=False), []
    mask = ~np.isnan(features).any(axis=1)
    return features[mask].astype(np.float32, copy=False), np.asarray(ids, dtype=object)[mask].tolist()


def sample_training_vectors(all_fns, load_shard, sampling_rate, seed):
    """Sample training rows across all shards, not whole shard files."""
    if not 0.0 < sampling_rate <= 1.0:
        raise ValueError(f"--sampling_rate must be > 0 and <= 1, got {sampling_rate}")

    shard_counts = []
    total_rows = 0
    feature_dim = None
    for fn in tqdm(all_fns, desc="counting training rows", dynamic_ncols=True):
        features, _ = load_shard(fn)
        features = filter_valid_features(features)
        if features.shape[0] == 0:
            continue
        if feature_dim is None:
            feature_dim = features.shape[1]
        elif features.shape[1] != feature_dim:
            raise ValueError(f"Inconsistent feature dimension in {fn}: {features.shape[1]} != {feature_dim}")
        shard_counts.append((fn, features.shape[0]))
        total_rows += features.shape[0]

    if total_rows == 0:
        raise ValueError("No valid embedding rows found for training")

    sample_size = total_rows
    if sampling_rate < 1.0:
        sample_size = max(1, int(np.ceil(total_rows * sampling_rate)))

    rng = np.random.default_rng(seed)
    selected = np.arange(total_rows)
    if sample_size < total_rows:
        selected = np.sort(rng.choice(total_rows, size=sample_size, replace=False))

    sampled_batches = []
    offset = 0
    for fn, rows in tqdm(shard_counts, desc="loading training rows", dynamic_ncols=True):
        end = offset + rows
        left = np.searchsorted(selected, offset, side="left")
        right = np.searchsorted(selected, end, side="left")
        if right > left:
            features, _ = load_shard(fn)
            features = filter_valid_features(features)
            sampled_batches.append(features[selected[left:right] - offset])
        offset = end

    return np.ascontiguousarray(np.concatenate(sampled_batches, axis=0))


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
        """,
    )

    # Positional arguments
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing embedding shards. Auto-detected as either *.npy dict files "
        "(with 'features' and 'ids') or *.tar WebDataset shards of per-video *.npz files "
        "(with 'embeddings' and 'keyframe_ids', mean-pooled to one embedding per video).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory where the FAISS index and document IDs will be saved. "
        "Creates 'index.faiss' (the FAISS index) and 'index.ids' (document ID mapping).",
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
        default=0.25,
        help="Fraction of embedding rows to use for training the index (0.0-1.0). "
             "Default: 0.25 (25%%). Higher values improve index quality but increase training time. "
             "Recommended: 0.05-0.1 for large datasets, 0.1-0.5 for smaller datasets."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for row-level training sampling. Default: 0."
    )
    parser.add_argument(
        "--npz-id-strip-regex",
        type=str,
        default=None,
        help="Optional regex removed from each WebDataset .npz member stem before writing index.ids. "
        "Only applies to *.tar inputs. Example for multivent-raw: '\\.kf.*$'.",
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

    all_fns, load_shard = discover_shards(args.input_dir)
    load_shard = wrap_loader(load_shard, args.npz_id_strip_regex)

    sampled_vectors = sample_training_vectors(all_fns, load_shard, args.sampling_rate, args.seed)

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
        features, ids = load_shard(fn)
        features, ids = filter_valid_features_and_ids(features, ids)
        if not ids:
            continue

        index.add(np.ascontiguousarray(features))
        docids += ids

    print("saving faiss index")
    faiss.write_index(index, str(output_dir / "index.faiss"))

    print("saving doc ids")
    with (output_dir / "index.ids").open("w") as fw:
        for docid in docids:
            fw.write(f"{docid}\n")
