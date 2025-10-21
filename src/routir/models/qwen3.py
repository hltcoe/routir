import itertools
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import openai
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from trecrun import TRECRun

from ..utils import dict_topk, load_singleton, logger
from .abstract import Engine


try:
    import faiss
except ImportError:
    logger.warning("Failed to import Faiss for Qwen3")


class Qwen3(Engine):
    """
    Qwen3 embedding-based dense retrieval engine.

    Uses Qwen3 embeddings with FAISS for efficient similarity search.
    Supports both local and API-based embedding generation.

    Attributes:
        embedding_model_name: Name of the embedding model
        local_embedding_model: Local embedding model instance (if not using API)
        client: OpenAI-compatible API client (if using API)
        index: FAISS index for search
        doc_ids: List of document IDs corresponding to index vectors
        subset_mapper: Optional mapping of document IDs to subsets
    """

    def __init__(self, name: str = "Qwen3", config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        """
        Initialize Qwen3 engine.

        Args:
            name: Engine name
            config: Configuration with index_path, embedding_model_name, etc.
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        self.embedding_model_name = self.config.get("embedding_model_name", "Qwen/Qwen3-Embedding-8B")

        if self.config.get("embedding_base_url"):
            self.local_embedding_model = None
            self.client = openai.AsyncOpenAI(
                api_key=self.config.get("api_key", os.getenv("OPENAI_API_KEY", "noneset")),
                base_url=self.config["embedding_base_url"],
            )
        else:
            self.local_embedding_model = Qwen3EmbeddingModel(
                model_name=self.embedding_model_name,
                max_length=self.config.get("max_length", 8192),
                batch_size=self.config.get("batch_size", 8),
                instruction="Given a web search query, retrieve relevant passages that answer the query",
            )

        index_dir = Path(self.config["index_path"])
        index_path = index_dir / "index.faiss"
        ids_path = index_dir / "index.ids"

        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"Loading document IDs from: {ids_path}")
        with ids_path.open("r") as f:
            self.doc_ids = [line.strip() for line in f]

        logger.info(f"Index contains {self.index.ntotal} vectors")

        self.subset_mapper: Dict[str, str] = None
        if "id_to_subset_mapping" in self.config:
            if self.config["id_to_subset_mapping"].endswith(".pkl"):
                self.subset_mapper = load_singleton(self.config["id_to_subset_mapping"])
            else:
                logger.warning(f"Unable to load subset mapping file {self.config['id_to_subset_mapping']}")

    def filter_subset(self, scores: Dict[str, float], only_subset: str = None):
        if only_subset is None or self.subset_mapper is None:
            return scores
        return {doc_id: score for doc_id, score in scores.items() if self.subset_mapper[doc_id] == only_subset}

    async def search_batch(
        self, queries: List[str], limit: Union[int, List[int]] = 20, subsets: List[str] = None, maxp: bool = True
    ) -> List[Dict[str, float]]:
        if isinstance(limit, int):
            limit = [int(limit)] * len(queries)

        if subsets is None:
            subsets = [None] * len(queries)

        queries = self.add_query_instructions(queries)
        if self.local_embedding_model:
            _, query_embeddings = self.local_embedding_model.encode(list(enumerate(queries)))
        else:
            query_embeddings = await self.client.embeddings.create(model=self.embedding_model_name, input=queries)
            query_embeddings = np.array([x.embedding for x in query_embeddings.data])

        scores, ids = self.index.search(x=query_embeddings, k=int(max(limit) * self.config.get("k_scale", 20)))

        qmap = dict(enumerate(queries))
        run = TRECRun({qid: dict(zip([self.doc_ids[x] for x in ids[qid]], scores[qid])) for qid in qmap})
        results = [run[str(qid)] for qid, _ in enumerate(queries)]

        return [dict_topk(self.filter_subset(scores, subset), l) for subset, l, scores in zip(subsets, limit, results)]

    def add_query_instructions(self, queries) -> List[str]:
        task_description = "Given a web search query, retrieve relevant passages that answer the query"
        return [f"Instruct: {task_description}\nQuery:{query}" for query in queries]


class Qwen3EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", max_length=8192, batch_size=8, instruction=None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, padding_side="left")
        self.tokenize = partial(
            self.tokenizer,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        try:
            self.model = AutoModel.from_pretrained(
                self.model_name, attn_implementation="flash_attention_2", torch_dtype=torch.float16
            ).cuda()
        except ImportError:
            logger.warning("Failed to import flash_attn for Qwen3... loading model without flash_attention_2")
            self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16).cuda()

        self.device = self.model.device
        self.model.eval()

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def encode_batch(self, tokens):
        outputs = self.model(**tokens)
        embeddings = self.last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def encode(self, data):
        ids = []
        encoded = []

        data_iter = iter(data)
        with torch.no_grad():
            for _ in tqdm(
                range(0, len(data), self.batch_size),
                desc="qwen3.encode",
                leave=True,
            ):
                id_batch, text_batch = zip(*list(itertools.islice(data_iter, self.batch_size)))
                assert text_batch
                ids.extend(id_batch)

                token_batch = {k: v.to(self.device) for k, v in self.tokenize(text_batch).items()}
                embeddings = self.encode_batch(token_batch)
                encoded.append(embeddings.cpu())

        stacked_embeddings = torch.vstack(encoded)
        return ids, stacked_embeddings
