import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import aiohttp

from ..utils import FactoryEnabled, dict_topk, session_request


class Engine(FactoryEnabled):
    def __init__(self, name: str = None, config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            config = json.loads(Path(config).read_text())

        self.config: Dict[str, Any] = {**config, **kwargs}

        self.name: str = name or self.config.get("name", None)
        if "index_path" in self.config:
            self.index_path: Path = Path(self.config["index_path"])
        else:
            self.index_path: Path = None

    async def search_batch(self, queries: List[str], limit: Union[int, List[int]] = 20, **kwargs) -> List[Dict[str, float]]:
        raise NotImplementedError

    async def search(self, query: str, limit: int = 20, **kwargs) -> Dict[str, float]:
        return (await self.search_batch([query], limit, **kwargs))[0]

    async def score_batch(
        self, queries: List[str], passages: List[str], candidate_length: List[int] = None, **kwargs
    ) -> List[List[float]]:
        raise NotImplementedError

    async def score(self, query: str, passages: List[str], **kwargs) -> List[float]:
        raise (await self.score_batch([query], passages, [len(passages)]))[0]

    async def decompose_query_batch(self, queries: List[str], limit: List[int] = None, **kwargs) -> List[List[str]]:
        raise NotImplementedError

    async def decompose_query(self, query: str, **kwargs) -> List[str]:
        return (await self.decompose_query_batch([query], **kwargs))[0]

    async def fuse_batch(
        self, queries: List[str], batch_scores: List[List[Dict[str, float]]], **kwargs
    ) -> List[Dict[str, float]]:
        raise NotImplementedError

    async def fuse(self, query: str, scores: List[Dict[str, float]], **kwargs) -> Dict[str, float]:
        return (await self.fuse_batch([query], [scores], **kwargs))[0]

    @property
    def can_search(self) -> bool:
        return self.__class__.search_batch != Engine.search_batch

    @property
    def can_score(self) -> bool:
        return self.__class__.score_batch != Engine.score_batch

    @property
    def can_decompose_query(self) -> bool:
        return self.__class__.decompose_query_batch != Engine.decompose_query_batch

    @property
    def can_fuse(self) -> bool:
        return self.__class__.fuse_batch != Engine.fuse_batch


class Reranker(Engine):
    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        self.upstream = None
        if "upstream_service" in self.config:
            self.upstream = Engine.load(
                self.config["upstream_service"]["engine"], config=self.config["upstream_service"]["config"], **kwargs
            )

        self.text_service = None
        if "text_service" in self.config:
            # Optional, or else you would need to get the document text in other ways
            # e.g. ir_datasets
            assert "endpoint" in self.config["text_service"]
            assert "collection" in self.config["text_service"]
            self.text_service = self.config["text_service"]

        self.rerank_topk_max = int(self.config.get("rerank_topk_max", 100))
        self.rerank_multiplier = float(self.config.get("rerank_multiplier", 5))

    async def get_text(self, docids: Union[str, List[str]]):
        if self.text_service is None:
            raise RuntimeError("No text service provided. Either missing in config or needs to be implemented in subclass.")

        docids = [docids] if isinstance(docids, str) else docids
        async with aiohttp.ClientSession() as session:
            resps = await asyncio.gather(
                *[
                    session_request(
                        session,
                        self.text_service["endpoint"] + "/content",
                        {"collection": self.text_service["collection"], "id": docid},
                    )
                    for docid in set(docids)
                ]
            )
            return {resp["id"]: resp["text"] for resp in resps}

    async def search_batch(self, queries, limit=20, **kwargs) -> List[Dict[str, float]]:
        if self.upstream is None:
            raise RuntimeError(f"Upstream retrieval is not defined, {self.name} only support scoring.")

        if not isinstance(limit, list):
            limit = [limit] * len(queries)
        assert len(limit) == len(queries)

        multiplier = kwargs.get("rerank_multiplier", self.rerank_multiplier)

        upstream_limit = [min(k * multiplier, self.rerank_topk_max) for k in limit]

        upstream_results = await self.upstream.search_batch(queries, limit=upstream_limit, **kwargs)
        candidate_docids = [list(upr.keys()) for upr in upstream_results]

        all_text = await self.get_text(sum(candidate_docids, []))

        candidate_text = [all_text[c] for candidates in candidate_docids for c in candidates]
        candidate_lengths = [len(c) for c in candidate_docids]

        raw_scores = await self.score_batch(queries, candidate_text, candidate_lengths, **kwargs)

        return [
            dict_topk(dict(zip(candidates, qscores)), k) for qscores, candidates, k in zip(raw_scores, candidate_docids, limit)
        ]


class Aggregation:
    def __init__(self, passage_mapping: Dict[str, str]):
        self.mapping = passage_mapping

    def __contains__(self, pid: str) -> bool:
        return pid in self.mapping

    def __getitem__(self, pid: str) -> str:
        return self.mapping[pid]

    def maxp(self, passage_scores: Dict[str, float]) -> Dict[str, float]:
        ret = {}
        for pid, score in passage_scores.items():
            doc_id = self.mapping[pid]
            if doc_id not in ret or ret[doc_id] < score:
                ret[doc_id] = score
        return ret

    @property
    def n_docs(self):
        return len(set(self.mapping.values()))

    def __len__(self):
        return len(self.mapping)


class OptimizedAggregation(Aggregation):
    """Optimized version of Aggregation with pre-computed structures."""

    def __init__(self, passage_mapping: List[str]):
        # Convert list to dict with integer indices for faster lookup
        self.mapping = {str(i): doc_id for i, doc_id in enumerate(passage_mapping)}

        # Pre-compute unique documents for faster statistics
        self._unique_docs = set(self.mapping.values())

        # Create reverse mapping for potential optimizations
        self.reverse_mapping = {}
        for pid, doc_id in self.mapping.items():
            if doc_id not in self.reverse_mapping:
                self.reverse_mapping[doc_id] = []
            self.reverse_mapping[doc_id].append(pid)

    def maxp(self, passage_scores: Dict[str, float]) -> Dict[str, float]:
        """Optimized MaxP using dictionary comprehension."""
        ret = {}
        for pid, score in passage_scores.items():
            if pid in self.mapping:
                doc_id = self.mapping[pid]
                if doc_id not in ret:
                    ret[doc_id] = score
                elif score > ret[doc_id]:
                    ret[doc_id] = score
        return ret

    @property
    def n_docs(self):
        return len(self._unique_docs)
