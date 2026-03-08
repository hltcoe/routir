import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import aiohttp

from ..utils import FactoryEnabled, dict_topk, session_request


class Engine(FactoryEnabled):
    """
    Abstract base class for all retrieval and reranking engines.

    Subclass this to integrate any retrieval model — dense retrieval, sparse
    search, cross-encoder reranker, or fusion — with RoutIR's asynchronous
    batching and caching infrastructure.

    Only implement the methods that apply to your engine.  Capabilities are
    auto-detected by the ``can_search``, ``can_score``, ``can_decompose_query``,
    and ``can_fuse`` properties; unimplemented methods raise ``NotImplementedError``.

    **Two-layer extension pattern**

    All RoutIR extensions follow this structure:

    1. A *model wrapper* class that handles tokenisation and synchronous inference.
    2. An *Engine subclass* that adapts the wrapper to RoutIR's async batch interface.

    Example:
        .. code-block:: python

            from routir.models.abstract import Engine

            class MyRerankerEngine(Engine):
                def __init__(self, name=None, config=None, **kwargs):
                    super().__init__(name, config, **kwargs)
                    self.model = MyModelWrapper(
                        model_path=config.get("model", "org/model"),
                    )

                async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
                    # See score_batch docstring for the required pattern.
                    ...

    Engines are loaded by class name via the :class:`~routir.utils.FactoryEnabled`
    registry — the class name in the config ``"engine"`` field must match exactly.

    Attributes:
        name (str): Engine identifier used in API responses and logs.
        config (dict): Merged configuration (from the ``config`` arg plus ``kwargs``).
        index_path (Path or None): ``Path`` parsed from ``config["index_path"]``,
            or ``None`` if not present.
    """

    def __init__(self, name: str = None, config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        """
        Initialize the engine.

        Args:
            name (str, optional): Human-readable engine name.  Falls back to
                ``config["name"]`` if omitted.
            config (dict or str or Path, optional): Engine configuration.
                Accepted forms:

                * ``dict`` — used directly.
                * ``str`` or ``Path`` pointing to a JSON file — loaded and parsed.
                * ``None`` — treated as an empty dict.

                Any extra ``**kwargs`` are merged in and take precedence over the
                config dict values.

                Keys consumed by this base class:

                * ``"name"`` — fallback engine name when ``name`` arg is ``None``.
                * ``"index_path"`` — path to a search index; exposed as
                  ``self.index_path`` (a ``Path`` object).

                All other keys (e.g. ``"model"``, ``"ngpus"``) are preserved in
                ``self.config`` for subclass use.
            **kwargs: Merged into ``config``; kwargs override config-file values.
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
        Retrieve ranked documents for a batch of queries.

        Override this method to implement a search engine (dense, sparse, or
        hybrid).  Return a ``dict`` per query mapping document IDs (strings) to
        relevance scores; higher scores mean higher relevance.

        Example:
            .. code-block:: python

                async def search_batch(self, queries, limit=20, **kwargs):
                    if isinstance(limit, int):
                        limit = [limit] * len(queries)
                    return [
                        {hit.docid: hit.score
                         for hit in self.searcher.search(q, k=k)}
                        for q, k in zip(queries, limit)
                    ]

        Args:
            queries (List[str]): Batch of query strings,
                e.g. ``["What is AI?", "BERT paper"]``.
            limit (int or List[int]): Maximum results per query.  A single
                ``int`` applies the same limit to every query; a ``List[int]``
                sets per-query limits.
            **kwargs: Forwarded from the caller (e.g. pipeline runtime kwargs).

        Returns:
            List[Dict[str, float]]: One result dict per query, mapping document
            IDs to scores::

                [
                    {"doc1": 12.3, "doc2": 9.8},  # results for queries[0]
                    {"doc3": 7.1},                 # results for queries[1]
                ]
        """
        raise NotImplementedError

    async def search(self, query: str, limit: int = 20, **kwargs) -> Dict[str, float]:
        """Perform single query search."""
        return (await self.search_batch([query], limit, **kwargs))[0]

    async def score_batch(
        self, queries: List[str], passages: List[str], candidate_length: List[int] = None, **kwargs
    ) -> List[List[float]]:
        """
        Score query–passage pairs for a batch of queries.

        Override this method to implement a reranker or bi-encoder scorer.

        **Critical**: passages are passed as a *flattened* list across all
        queries.  ``candidate_length[i]`` tells you how many consecutive entries
        in ``passages`` belong to ``queries[i]``.

        Example:
            .. code-block:: python

                # Input layout
                queries          = ["query1",            "query2"          ]
                candidate_length = [2,                   3                 ]
                passages         = ["p1", "p2",          "p3", "p4", "p5"  ]
                #                   ^---- query1 ----^   ^------ query2 ---^

                # Expected return
                # [[score_p1, score_p2], [score_p3, score_p4, score_p5]]

            Standard implementation skeleton:

            .. code-block:: python

                async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
                    if candidate_length is None:
                        candidate_length = [len(passages)]

                    # Expand each query to align with its passages
                    expanded_queries = [
                        queries[i]
                        for i, l in enumerate(candidate_length)
                        for _ in range(l)
                    ]
                    pairs = list(zip(expanded_queries, passages))

                    # Score all pairs (sync model call is fine inside async)
                    all_scores = self.model.score(pairs)

                    # Re-group by query
                    start = 0
                    result = []
                    for l in candidate_length:
                        result.append(all_scores[start:start + l])
                        start += l
                    return result

        Args:
            queries (List[str]): ``N`` query strings.
            passages (List[str]): Flattened list of all passages for all queries.
                Total length must equal ``sum(candidate_length)``.
            candidate_length (List[int], optional): Number of passages for each
                query.  ``candidate_length[i]`` is the passage count for
                ``queries[i]``.  Defaults to ``[len(passages)]`` (all passages
                belong to the single query).
            **kwargs: Extra scoring parameters forwarded from the caller.

        Returns:
            List[List[float]]: One score list per query, preserving passage order::

                [[score_p1, score_p2], [score_p3, score_p4, score_p5]]
        """
        raise NotImplementedError

    async def score(self, query: str, passages: List[str], **kwargs) -> List[float]:
        """Score a single query against multiple passages."""
        return (await self.score_batch([query], passages, [len(passages)]))[0]

    async def decompose_query_batch(self, queries: List[str], limit: List[int] = None, **kwargs) -> List[List[str]]:
        """
        Decompose each query into sub-queries for query expansion.

        Override this method to implement a query expansion engine.  Used in
        pipeline expressions like ``expander{retriever}Fusion``, where the
        expander generates sub-queries that are each sent to the retriever and
        then fused.

        Example:
            .. code-block:: python

                async def decompose_query_batch(self, queries, limit=None, **kwargs):
                    results = []
                    for i, query in enumerate(queries):
                        n = limit[i] if limit else 5
                        sub_queries = self.llm.expand(query, n=n)
                        results.append(sub_queries)
                    return results

        Args:
            queries (List[str]): Original query strings to expand.
            limit (List[int], optional): Maximum number of sub-queries per query.
                ``limit[i]`` applies to ``queries[i]``.  ``None`` means no limit
                (engine decides).
            **kwargs: Additional parameters forwarded from the caller.

        Returns:
            List[List[str]]: One list of sub-queries per input query::

                [["sub-query 1a", "sub-query 1b"], ["sub-query 2a"]]
        """
        raise NotImplementedError

    async def decompose_query(self, query: str, **kwargs) -> List[str]:
        """Decompose a single query into sub-queries."""
        return (await self.decompose_query_batch([query], **kwargs))[0]

    async def fuse_batch(
        self, queries: List[str], batch_scores: List[List[Dict[str, float]]], **kwargs
    ) -> List[Dict[str, float]]:
        """
        Fuse multiple ranked lists into a single ranking per query.

        Override this method to implement a fusion engine.  Used in pipeline
        expressions like ``{dense, sparse}RRF%100``, where each system in the
        braces contributes a ranked list and this method merges them.

        Example:
            .. code-block:: python

                async def fuse_batch(self, queries, batch_scores, **kwargs):
                    results = []
                    for scores_list in batch_scores:
                        fused = {}
                        for rank_list in scores_list:
                            for rank, (doc_id, _) in enumerate(
                                sorted(rank_list.items(), key=lambda x: -x[1])
                            ):
                                fused[doc_id] = fused.get(doc_id, 0) + 1 / (60 + rank + 1)
                        results.append(fused)
                    return results

        Args:
            queries (List[str]): Query strings, one per result set to fuse.
            batch_scores (List[List[Dict[str, float]]]): Outer list has one entry
                per query.  Each entry is a list of ranked-result dicts to fuse::

                    batch_scores[i] = [
                        {"doc1": 0.9, "doc2": 0.7},  # results from system A
                        {"doc2": 0.8, "doc3": 0.6},  # results from system B
                    ]

            **kwargs: Fusion parameters forwarded from the caller.

        Returns:
            List[Dict[str, float]]: One fused result dict per query::

                [{"doc1": 0.032, "doc2": 0.030, "doc3": 0.016}]
        """
        raise NotImplementedError

    async def fuse(self, query: str, scores: List[Dict[str, float]], **kwargs) -> Dict[str, float]:
        """Fuse multiple result sets for a single query."""
        # Rename "queries" from pipeline payload to "expanded_queries" to avoid
        # shadowing fuse_batch's positional `queries` arg, while still making
        # the expander's sub-queries available to custom fusion engines.
        if "queries" in kwargs:
            kwargs["expanded_queries"] = kwargs.pop("queries")
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
    Convenience base class for rerankers that retrieve candidates from an upstream engine.

    Use this instead of :class:`Engine` directly when your reranker should:

    * Fetch candidate documents from an upstream retrieval service before rescoring.
    * Look up document text from a RoutIR ``/content`` endpoint.

    If you handle candidate retrieval yourself (e.g. the pipeline supplies
    candidates), inherit from :class:`Engine` and implement ``score_batch`` only.

    **Configuration keys** (inside the ``"config"`` block of your service definition):

    * ``upstream_service`` *(dict, optional)* — Upstream engine to load for
      first-stage retrieval.  Required fields:

      .. code-block:: json

          {
              "engine": "PyseriniBM25",
              "config": {"index_path": "/path/to/bm25-index"}
          }

    * ``text_service`` *(dict, optional)* — RoutIR endpoint for document-text
      lookup, used by ``search_batch`` to fetch passage text before scoring.
      Both sub-keys are required if this block is present:

      .. code-block:: json

          {
              "endpoint": "http://localhost:5000",
              "collection": "my-collection"
          }

    * ``rerank_topk_max`` *(int, default 100)* — Hard cap on how many candidates
      the upstream engine fetches, regardless of ``rerank_multiplier``.

    * ``rerank_multiplier`` *(float, default 5)* — The upstream engine retrieves
      ``limit × rerank_multiplier`` candidates; the reranker then selects the
      best ``limit``.

    **Full config example**:

    .. code-block:: json

        {
            "name": "my-reranker",
            "engine": "MyRerankerEngine",
            "config": {
                "model": "org/reranker-model",
                "upstream_service": {
                    "engine": "PyseriniBM25",
                    "config": {"index_path": "/data/bm25-index"}
                },
                "text_service": {
                    "endpoint": "http://localhost:5001",
                    "collection": "my-docs"
                },
                "rerank_topk_max": 200,
                "rerank_multiplier": 10
            }
        }

    When ``upstream_service`` is configured, calling ``search_batch`` on this
    reranker automatically:

    1. Retrieves ``limit × rerank_multiplier`` candidates from the upstream engine.
    2. Fetches document text for each candidate via ``text_service``.
    3. Calls your ``score_batch`` implementation with those candidates.
    4. Returns the top-``limit`` results.

    Subclasses only need to implement ``score_batch``; ``search_batch`` is
    provided by this class.

    Attributes:
        upstream (Engine or None): Loaded upstream engine instance, or ``None``
            if ``upstream_service`` is not configured.
        text_service (dict or None): Parsed text-service config dict, or ``None``.
        rerank_topk_max (int): Maximum candidates passed to the reranker.
        rerank_multiplier (float): Upstream over-retrieval factor.
    """

    def __init__(self, name=None, config=None, **kwargs):
        """
        Initialize the reranker.

        Reads ``upstream_service``, ``text_service``, ``rerank_topk_max``, and
        ``rerank_multiplier`` from ``config``.  See the class docstring for the
        expected structure of each key.

        Args:
            name (str, optional): Reranker name.
            config (dict or str or Path, optional): Configuration dict or path.
                See :class:`Engine` for accepted forms.
            **kwargs: Merged into ``config``; kwargs override config values.
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
            if "endpoint" not in self.config["text_service"]:
                raise RuntimeError("text_service config is missing required 'endpoint' field")
            if "collection" not in self.config["text_service"]:
                raise RuntimeError("text_service config is missing required 'collection' field")
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
            failed = sum(1 for r in resps if r is None)
            if failed:
                raise RuntimeError(
                    f"Failed to retrieve text for {failed}/{len(resps)} documents from "
                    f"{self.text_service['endpoint']}/content"
                )
            return {resp["id"]: resp["text"] for resp in resps}

    async def search_batch(self, queries, limit=20, **kwargs) -> List[Dict[str, float]]:
        """
        Retrieve and rerank candidates using the configured upstream engine.

        This implementation is provided by :class:`Reranker` and calls
        ``score_batch`` internally.  Subclasses do **not** need to override this
        method; implement ``score_batch`` instead.

        Raises:
            RuntimeError: If ``upstream_service`` was not configured.
        """
        if self.upstream is None:
            raise RuntimeError(f"Upstream retrieval is not defined, {self.name} only support scoring.")

        if not isinstance(limit, list):
            limit = [limit] * len(queries)
        if len(limit) != len(queries):
            raise RuntimeError(f"len(limit)={len(limit)} does not match len(queries)={len(queries)}")

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
