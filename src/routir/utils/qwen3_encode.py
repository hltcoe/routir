#!/usr/bin/env python3
"""
Qwen3 Encoder

This script illustrates how to encode a jsonl document collection with Qwen3. This script's output can be used by faiss_indexing.py to create an index.

Usage:
    python qwen3_encode.py <jsonl file> <output directory> [options]

Example:
    python qwen3_encode.py docs.jsonl embeddings/ --id-field id --text-fields title text --model-name Qwen/Qwen3-Embedding-8B --max-length 8192 --batch-size 8 --docs-per-file 50000
"""

import argparse
import gzip
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import trange

from ..models.qwen3 import Qwen3EmbeddingModel


try:
    import ir_datasets as irds
except ImportError:
    irds = None  # ty:ignore[invalid-assignment]


class JSONLDataset:
    def __init__(self, fn, id_field, text_fields):
        self.fn = os.path.expanduser(fn)
        self.id_field = id_field
        self.text_fields = text_fields

        if isinstance(self.text_fields, str):
            self.text_fields = [self.text_fields]

        self.get_text = lambda d: " ".join(d.get(field, "").strip() for field in self.text_fields).strip()

        self._len = None

    def __iter__(self):
        if self.fn.endswith(".gz"):
            f = gzip.open(self.fn, "rt", encoding="utf-8")
        else:
            f = open(self.fn, "rt", encoding="utf-8")

        for idx, line in enumerate(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                print("JSONDecodeError:", line)
                continue
            text = self.get_text(d)
            if text:
                yield (d[self.id_field], text)

    def __len__(self):
        if not self._len:
            self._len = sum(1 for _ in self.__iter__())
        return self._len


class IRDSDataset:
    def __init__(self, collection_name, id_field: str = "doc_id", text_fields = None):
        if irds is None:
            raise ImportError("ir_datasets is not installed. Please install it to use IRDSDataset.")

        self.dataset = irds.load(collection_name).docs
        self.id_field = id_field
        self.text_fields = text_fields

        if isinstance(self.text_fields, str):
            self.text_fields = [self.text_fields]

        if isinstance(self.text_fields, list):
            self.get_text = lambda d: " ".join(getattr(d, field, "").strip() for field in self.text_fields).strip()  # ty:ignore[not-iterable]
        else:
            self.get_text = lambda d: d.default_text().strip()

    def __iter__(self):
        for d in self.dataset:
            text = self.get_text(d)
            if text:
                yield (getattr(d, self.id_field), text)

    def __len__(self):
        return len(self.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", type=str, help="jsonl file or ir_datasets collection ID")
    parser.add_argument("output_directory", type=Path, help="output directory")
    parser.add_argument("--id-field", type=str, default="id", help="Collection ID field (default: %(default)s)")
    parser.add_argument("--fields", type=str, nargs="*", default=["title", "text"], help="Collection text fields (default: %(default)s)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-8B", help="Qwen3 Embedding model (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum sequence length (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--docs-per-file", type=int, default=50000, help="Maximum documents per numpy file (default: %(default)s)")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed processing (default: %(default)s)")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)), help="World size for distributed processing (default: %(default)s)")
    args = parser.parse_args()

    if 'irds:' in args.collection:
        collection = IRDSDataset(args.collection.replace('irds:', ''), args.id_field, args.fields)
    else:
        collection = JSONLDataset(args.collection, args.id_field, args.fields)
    model = Qwen3EmbeddingModel(
        args.model_name, max_length=args.max_length, batch_size=args.batch_size,
        device=torch.device(f"cuda:{args.local_rank}")
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)

    reader = iter(collection)
    for idx in trange(0, len(collection), args.docs_per_file, desc="Shards", dynamic_ncols=True, disable=args.local_rank != 0):
        fn = args.output_directory / f"shard{idx}.npy"
        chunk = list(itertools.islice(reader, args.docs_per_file))
        if len(chunk) == 0:
            break

        if (idx // args.docs_per_file) % args.world_size != args.local_rank:
            continue

        docids, embeddings = model.encode(chunk, disable_tqdm=args.local_rank != 0)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        np.save(fn, {"features": embeddings, "ids": docids})


if __name__ == "__main__":
    main()
