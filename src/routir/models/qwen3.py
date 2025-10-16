import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import openai
from trecrun import TRECRun

from ..utils import dict_topk, load_singleton, logger
from .abstract import Engine


try:
    import faiss
except ImportError:
    logger.warning("Failed to import Faiss for Qwen3")


class Qwen3(Engine):
    def __init__(self, name: str = "Qwen3", config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        super().__init__(name, config, **kwargs)

        self.client = openai.AsyncOpenAI(
            api_key=self.config.get("api_key", os.getenv("OPENAI_API_KEY", "noneset")), base_url=self.config["embedding_base_url"]
        )
        self.embedding_model_name = self.config.get("embedding_model_name", "Qwen/Qwen3-Embedding-8B")

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

    def add_query_instructions(self, queries) -> List[str]:
        # TODO: move this to the config
        task_description = "Given a web search query, retrieve relevant passages that answer the query"
        return [f"Instruct: {task_description}\nQuery:{query}" for query in queries]

    async def search_batch(
        self, queries: List[str], limit: Union[int, List[int]] = 20, subsets: List[str] = None, maxp: bool = True
    ) -> List[Dict[str, float]]:
        if isinstance(limit, int):
            limit = [int(limit)] * len(queries)

        if subsets is None:
            subsets = [None] * len(queries)

        instruct_queries = self.add_query_instructions(queries)
        query_embeddings = await self.client.embeddings.create(model=self.embedding_model_name, input=instruct_queries)
        query_embeddings = np.array([x.embedding for x in query_embeddings.data])
        logger.debug(query_embeddings.shape)
        scores, ids = self.index.search(x=query_embeddings, k=max(limit) * self.config.get("k_scale", 20))

        qmap = dict(enumerate(queries))
        run = TRECRun({qid: dict(zip([self.doc_ids[x] for x in ids[qid]], scores[qid])) for qid in qmap})
        results = [run[str(qid)] for qid, _ in enumerate(queries)]

        return [dict_topk(self.filter_subset(scores, subset), l) for subset, l, scores in zip(subsets, limit, results)]
