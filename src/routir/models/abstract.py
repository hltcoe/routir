import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import aiohttp

from ..utils import FactoryEnabled, dict_topk, session_request


class Engine(FactoryEnabled):
    """
    Abstract class for all search and retrieval engines.

    Provides a common interface for search, scoring, query decomposition,
    and result fusion operations. Subclasses should implement the specific
    batch operations they support.

    Attributes:
        name: Engine identifier
        config: Engine configuration dictionary
        index_path: Path to the search index (if applicable)
    """

    def __init__(self, name: str = None, config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        """
        Initialize the engine.

        Args:
            name: Optional name for the engine
            config: Configuration as dict, JSON file path, or JSON string
            **kwargs: Additional configuration parameters
        """
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
        """
        Perform batch search for multiple queries.

        Args:
            queries: List of search queries
            limit: Maximum results per query (int for all, list for per-query limits)
            **kwargs: Additional search parameters

        Returns:
            List of dicts mapping document IDs to scores for each query
        """
        raise NotImplementedError

    async def search(self, query: str, limit: int = 20, **kwargs) -> Dict[str, float]:
        """Perform single query search."""
        return (await self.search_batch([query], limit, **kwargs))[0]

    async def score_batch(
        self, queries: List[str], passages: List[str], candidate_length: List[int] = None, **kwargs
    ) -> List[List[float]]:
        """
        Score query-passage pairs in batch.

        Args:
            queries: List of queries
            passages: Flattened list of passages for all queries
            candidate_length: Number of passages per query
            **kwargs: Additional scoring parameters

        Returns:
            List of score lists for each query
        """
        raise NotImplementedError

    async def score(self, query: str, passages: List[str], **kwargs) -> List[float]:
        """Score a single query against multiple passages."""
        raise (await self.score_batch([query], passages, [len(passages)]))[0]

    async def decompose_query_batch(self, queries: List[str], limit: List[int] = None, **kwargs) -> List[List[str]]:
        """
        Decompose queries into sub-queries.

        Args:
            queries: List of queries to decompose
            limit: Optional limit on sub-queries per query
            **kwargs: Additional parameters

        Returns:
            List of sub-query lists for each query
        """
        raise NotImplementedError

    async def decompose_query(self, query: str, **kwargs) -> List[str]:
        """Decompose a single query into sub-queries."""
        return (await self.decompose_query_batch([query], **kwargs))[0]

    async def fuse_batch(
        self, queries: List[str], batch_scores: List[List[Dict[str, float]]], **kwargs
    ) -> List[Dict[str, float]]:
        """
        Fuse multiple result sets per query.

        Args:
            queries: List of queries
            batch_scores: List of result lists to fuse for each query
            **kwargs: Fusion parameters

        Returns:
            List of fused results for each query
        """
        raise NotImplementedError

    async def fuse(self, query: str, scores: List[Dict[str, float]], **kwargs) -> Dict[str, float]:
        """Fuse multiple result sets for a single query."""
        return (await self.fuse_batch([query], [scores], **kwargs))[0]

    @property
    def can_search(self) -> bool:
        """Check if this engine implements search functionality."""
        return self.__class__.search_batch != Engine.search_batch

    @property
    def can_score(self) -> bool:
        """Check if this engine implements scoring functionality."""
        return self.__class__.score_batch != Engine.score_batch

    @property
    def can_decompose_query(self) -> bool:
        """Check if this engine implements query decomposition."""
        return self.__class__.decompose_query_batch != Engine.decompose_query_batch

    @property
    def can_fuse(self) -> bool:
        """Check if this engine implements result fusion."""
        return self.__class__.fuse_batch != Engine.fuse_batch


class Reranker(Engine):
    """
    Abstract class for reranking engines.

    Rerankers retrieve candidates from an upstream engine and rescore them.
    They require document text, either from a text service or other source.

    This is a helper class for rerankers that provides supports to gather document
    text and perform the reranking process.
    If you do not need these functionalities, you can directly inherit from Engine.

    Attributes:
        upstream: Upstream retrieval engine for candidate generation
        text_service: Configuration for retrieving document text
        rerank_topk_max: Maximum candidates to rerank
        rerank_multiplier: Factor to multiply requested limit for upstream retrieval
    """

    def __init__(self, name=None, config=None, **kwargs):
        """
        Initialize the reranker.

        Args:
            name: Reranker name
            config: Configuration dict or path
            **kwargs: Additional configuration
        """
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
        """
        Retrieve document text for the given document IDs.

        Args:
            docids: Single document ID or list of document IDs

        Returns:
            Dict mapping document IDs to their text content

        Raises:
            RuntimeError: If no text service is configured
        """
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
        """
        Helper method to perform reranking based on initial retrieval using upstream engine.
        """
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
    """
    Maps passages to documents and aggregates passage scores to document scores.

    Used for passage-level retrieval where results need to be aggregated to the
    document level using MaxP (maximum passage score).

    Attributes:
        mapping: Dict mapping passage IDs to document IDs
    """

    def __init__(self, passage_mapping: Dict[str, str]):
        """
        Initialize with passage-to-document mapping.

        Args:
            passage_mapping: Dict mapping passage IDs to document IDs
        """
        self.mapping = passage_mapping

    def __contains__(self, pid: str) -> bool:
        """Check if passage ID exists in mapping."""
        return pid in self.mapping

    def __getitem__(self, pid: str) -> str:
        """Get document ID for a passage ID."""
        return self.mapping[pid]

    def maxp(self, passage_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate passage scores to document scores using MaxP.

        Takes the maximum score among all passages belonging to each document.

        Args:
            passage_scores: Dict mapping passage IDs to scores

        Returns:
            Dict mapping document IDs to their maximum passage scores
        """
        ret = {}
        for pid, score in passage_scores.items():
            doc_id = self.mapping[pid]
            if doc_id not in ret or ret[doc_id] < score:
                ret[doc_id] = score
        return ret

    @property
    def n_docs(self):
        """Get number of unique documents."""
        return len(set(self.mapping.values()))

    def __len__(self):
        """Get number of passages."""
        return len(self.mapping)
