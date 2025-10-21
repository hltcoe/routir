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

from ..models.qwen3 import Qwen3EmbeddingModel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection", type=str, help="jsonl file")
    parser.add_argument("output_directory", type=Path, help="output directory")
    parser.add_argument("--id-field", type=str, default="id", help="Collection ID field (default: %(default)s)")
    parser.add_argument("--fields", type=str, nargs="*", default=["title", "text"], help="Collection text fields (default: %(default)s)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Embedding-8B", help="Qwen3 Embedding model (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=8192, help="Maximum sequence length (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--docs-per-file", type=int, default=50000, help="Maximum documents per numpy file (default: %(default)s)")
    args = parser.parse_args()

    collection = JSONLDataset(args.collection, args.id_field, args.fields)
    model = Qwen3EmbeddingModel(args.model_name, max_length=args.max_length, batch_size=args.batch_size)

    args.output_directory.mkdir(parents=True, exist_ok=True)

    for idx in range(0, len(collection), args.docs_per_file):
        fn = args.output_directory / f"shard{idx}.npy"
        chunk = list(itertools.islice(collection, args.docs_per_file))
       # import IPython;IPython.embed()
        docids, embeddings = model.encode(chunk)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        np.save(fn, {"features": embeddings, "ids": docids})


if __name__ == "__main__":
    main()
