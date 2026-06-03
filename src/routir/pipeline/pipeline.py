import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from ..processors.registry import ProcessorRegistry
from ..utils import dict_topk, logger
from .aliases import PipelineAliasRegistry
from .cache import get_pipeline_cache
from .parser import CallSequence, ParallelCallSequences, PipelineComponent, SystemCall, expand_aliases, parser


# TODO: probably should unify it...
_role_to_service = {"search": "search", "rerank": "score", "expander": "decompose_query", "merger": "fuse"}


class SearchPipeline:
    """
    Executes a RoutIR pipeline defined by the pipeline DSL.

    A ``SearchPipeline`` is constructed from a parsed
    :class:`~routir.pipeline.parser.PipelineComponent` tree (or directly from a
    DSL string via :meth:`from_string`) and runs all stages recursively, passing
    results between them.

    **Pipeline DSL syntax** (see :mod:`routir.pipeline.parser` for full reference)::

        # Simple retrieval
        "bm25%20"

        # Sequential reranking: retrieve 100, rerank to top 20
        "bm25%100 >> my-reranker%20"

        # Parallel retrieval + fusion
        "{dense, sparse}RRF%100"

        # Query expansion + parallel + fusion
        "expander{dense, sparse}RRF%100"

        # Named alias for per-stage runtime_kwargs
        "dense[d]%100 >> reranker%20"

    **runtime_kwargs**

    Pass per-stage parameters at query time by giving a stage an alias with
    ``[alias]`` and providing a dict keyed by that alias::

        pipeline = SearchPipeline.from_string(
            "dense[d]%100 >> reranker%20",
            collection="my-docs",
            runtime_kwargs={"d": {"instruction": "Retrieve relevant passages"}},
        )

    **Verification**

    At construction time the pipeline checks that every referenced service is
    registered in :data:`~routir.processors.registry.ProcessorRegistry`.
    Reranking stages additionally require a content service for the collection
    (to fetch document text for rescoring).

    Attributes:
        pipeline (PipelineComponent): Parsed pipeline AST root node.
        collection (str): Collection name used to fetch document text during
            reranking stages.
        runtime_kwargs (dict): Per-alias extra parameters injected into service
            call payloads at execution time.  Keys are pipeline aliases (defaulting
            to the service name); values are dicts merged into the request.
        doc_content_cache (dict): Per-run cache mapping document IDs to their
            text content, populated lazily during reranking and shared across
            all stages within a single :meth:`run` call.
    """

    def __init__(
        self,
        pipeline: PipelineComponent,
        collection: Optional[str] = None,
        runtime_kwargs: Dict[str, Dict[str, Any]] = None,
        verify: bool = True,
        bytes_content_cache_max_bytes: Optional[int] = None,
    ):
        """
        Initialize search pipeline.

        Args:
            pipeline (PipelineComponent): Parsed pipeline AST, typically produced
                by :func:`~routir.pipeline.parser.parser.parse`.
            collection (str, optional): Document collection name. Required only
                when the pipeline contains a reranking stage (which fetches
                document text). Must be registered as a ``"content"`` service in
                :data:`~routir.processors.registry.ProcessorRegistry`.
            runtime_kwargs (dict, optional): Per-alias extra parameters injected
                into individual service call payloads at execution time. Keys are
                pipeline aliases; values are dicts.  Example::

                    {"dense": {"instruction": "Retrieve relevant passages"}}

            verify (bool): If ``True`` (default), validate the pipeline at
                construction time.  Raises ``ValueError`` for bad input (e.g.
                the pipeline reranks but no ``collection`` was provided) and
                ``RuntimeError`` for operational issues (referenced service or
                content processor not registered).
            bytes_content_cache_max_bytes (int, optional): Per-pipeline cap on
                the bytes payloads cached in ``doc_content_cache``.  When set
                and an insertion would push the accumulated bytes-payload size
                over the cap, the cache evicts in insertion order (FIFO) until
                under the cap.  Only bytes-view payloads count toward the cap;
                text entries are free.  ``None`` (default) disables eviction.
        """
        self.pipeline = pipeline
        self.collection = collection
        self.runtime_kwargs = runtime_kwargs or {}
        # Per-run document content cache, keyed by (view, doc_id) so different
        # views of the same document are stored independently.
        self.doc_content_cache: Dict[Tuple[str, str], Any] = {}
        # Bytes-payload cap on doc_content_cache; ``None`` => unbounded.
        self._bytes_cache_max: Optional[int] = bytes_content_cache_max_bytes
        self._bytes_cache_used: int = 0

        if verify:
            self.verify()
        alias_not_found = set(self.runtime_kwargs.keys()) - set([c.alias for c in self.pipeline.all_calls])
        if alias_not_found:
            raise RuntimeError(f"Runtime kwargs reference unknown pipeline aliases: {alias_not_found}")

        # ``view`` is structural — it lives in the DSL, not in runtime_kwargs.
        # Reject any attempt to override it through the runtime layer.
        for alias, kwargs in (self.runtime_kwargs or {}).items():
            if "view" in kwargs:
                raise ValueError(
                    f"runtime_kwargs for alias '{alias}' contains 'view'; "
                    "view is structural and must be set in the pipeline DSL"
                )

    def verify(self):
        """Verify that all required services exist and views resolve.

        PR4 reads view metadata from the registry slot rather than the
        content processor directly so the same code path validates both local
        :class:`~routir.collections.processor.ContentProcessor` and remote
        :class:`~routir.collections.relay.RelayContentProcessor` collections.
        Each rerank stage's slot-level ``view_kind`` is also checked against
        the view's declared kind so a bytes-modality view never gets routed
        to a text-only scorer (or vice versa).
        """
        if any(call.role == "rerank" for call in self.pipeline.all_calls):
            if not self.collection:
                raise ValueError("pipeline contains reranking stage(s) and requires a `collection` field")
            if not ProcessorRegistry.has_service(self.collection, "content"):
                raise RuntimeError(
                    f"Pipeline requires reranking but no content service found for collection '{self.collection}'"
                )
            # Source-of-truth: registry slot metadata.  Works uniformly for
            # local ContentProcessor and remote RelayContentProcessor since
            # both register the same ``views`` / ``default_view`` keys.
            content_meta = ProcessorRegistry.get_meta(self.collection, "content")
            collection_views = content_meta.get("views")  # Dict[name -> kind] or None
            default_view = content_meta.get("default_view")
            # Relays that haven't been hydrated yet may have an empty dict; skip
            # the view check and let runtime errors surface in that case.
            if collection_views:
                for call in self.pipeline.all_calls:
                    if call.role != "rerank":
                        continue
                    resolved = call.view or default_view
                    if resolved is None:
                        raise ValueError(
                            f"stage '{call.alias}' requires a view but none specified "
                            f"and collection '{self.collection}' has no default_view"
                        )
                    if resolved not in collection_views:
                        raise ValueError(
                            f"view '{resolved}' (requested by stage '{call.alias}') is not in "
                            f"collection '{self.collection}'; available views: {sorted(collection_views.keys())}"
                        )
                    # PR4 slot-kind check: the resolved view's kind must match
                    # the score service's declared ``view_kind``.
                    required_kind = collection_views[resolved]
                    slot_kind = ProcessorRegistry.get_meta(call.name, "score").get("view_kind", "text")
                    if required_kind != slot_kind:
                        raise ValueError(
                            f"stage '{call.alias}' (view '{resolved}', kind '{required_kind}') "
                            f"is routed to service '{call.name}' which accepts '{slot_kind}'"
                        )

        for call in self.pipeline.all_calls:
            if not ProcessorRegistry.has_service(call.name, _role_to_service[call.role]):
                raise RuntimeError(f"No {_role_to_service[call.role]} service registered under '{call.name}'")

    @classmethod
    def from_string(
        cls,
        pipeline_string: str,
        collection: Optional[str] = None,
        runtime_kwargs: Dict[str, Dict[str, Any]] = None,
        verify: bool = True,
        bytes_content_cache_max_bytes: Optional[int] = None,
    ) -> "SearchPipeline":
        """
        Create a pipeline from a DSL string.

        This is the primary constructor for most use cases.

        Args:
            pipeline_string (str): Pipeline specification. Examples::

                "bm25%20"                          # simple retrieval, top 20
                "bm25%100 >> reranker%20"          # sequential reranking
                "{dense, sparse}RRF%100"           # parallel + fusion
                "expander{dense, sparse}RRF%100"   # query expansion
                "dense[d]%100 >> reranker%20"      # with alias for runtime_kwargs

            collection (str, optional): Document collection name for content
                lookup during reranking. Required only when the pipeline
                contains a reranking stage.
            runtime_kwargs (dict, optional): Per-alias extra parameters injected
                into service payloads at query time.  Keys are pipeline aliases
                (the name inside ``[...]``; defaults to service name).  Example::

                    {"d": {"instruction": "Given a query, retrieve passages…"}}

            verify (bool): Verify that all referenced services exist in the
                registry at construction time (default: ``True``).

        Returns:
            SearchPipeline: Ready-to-run pipeline instance.
        """
        ast = parser.parse(pipeline_string)
        ast = expand_aliases(ast, PipelineAliasRegistry.expanded)
        return cls(
            ast,
            collection,
            runtime_kwargs,
            verify,
            bytes_content_cache_max_bytes=bytes_content_cache_max_bytes,
        )

    def _cache_key(self, query: str) -> str:
        """Build the cache key for this pipeline + query.

        The AST is canonicalised via the dataclass ``repr``, which is stable
        across runs for identical inputs and ignores whitespace differences in
        the original DSL string.  ``SystemCall.view`` is part of the dataclass
        fields, so ``repr(self.pipeline)`` automatically distinguishes runs
        that differ only by view.  ``runtime_kwargs`` is filtered down to the
        aliases that actually appear in the pipeline so that an unused key does
        not poison the cache.
        """
        used_aliases = {c.alias for c in self.pipeline.all_calls}
        filtered_kwargs = {k: v for k, v in self.runtime_kwargs.items() if k in used_aliases}
        return json.dumps(
            [repr(self.pipeline), query, self.collection, filtered_kwargs],
            sort_keys=True,
        )

    @classmethod
    async def cached_run(
        cls,
        pipeline_string: str,
        query: str,
        collection: Optional[str] = None,
        runtime_kwargs: Dict[str, Dict[str, Any]] = None,
        bytes_content_cache_max_bytes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a pipeline with optional process-wide result caching.

        Single entry point used by both REST (``/pipeline``) and gRPC
        (``RoutirServicer.Pipeline``) handlers.  Behaviour:

        1. Parse the DSL and construct a :class:`SearchPipeline`.  Any DSL or
           verification error propagates to the caller unchanged.
        2. If a pipeline cache is installed (see
           :func:`routir.pipeline.cache.set_pipeline_cache`), look up the
           result by canonical key.  On hit, return it with ``cached=True``
           overriding any stage-level flag.
        3. On miss, execute the pipeline; cache the response only when it has
           no ``"error"`` key, mirroring :class:`~routir.processors.abstract.Processor`.

        Args:
            pipeline_string: Pipeline DSL string.
            query: User query.
            collection: Collection name for reranking stages.
            runtime_kwargs: Optional per-alias extra parameters.
            bytes_content_cache_max_bytes: Per-run cap on the bytes payloads
                cached during pipeline execution; see :meth:`__init__` for
                details.  Typically sourced from
                :attr:`~routir.config.Config.bytes_content_cache_max_bytes`.

        Returns:
            Final pipeline result dict.  Includes a top-level ``cached`` flag
            indicating whether the *pipeline-level* cache was hit (stage-level
            cache flags are not surfaced when the pipeline cache is in use).
        """
        pipeline = cls.from_string(
            pipeline_string,
            collection,
            runtime_kwargs=runtime_kwargs,
            bytes_content_cache_max_bytes=bytes_content_cache_max_bytes,
        )

        cache = get_pipeline_cache()
        cache_key = None
        if cache is not None:
            cache_key = pipeline._cache_key(query)
            cached = await cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True}

        result = await pipeline.run(query)

        if cache is not None and cache_key is not None and "error" not in result:
            # eventually consistent; mirrors Processor.submit behaviour
            asyncio.create_task(cache.put(cache_key, {**result, "cached": True}))

        return {**result, "cached": False}

    @staticmethod
    def _value_bytes_size(value: Any) -> int:
        """Return the byte-payload size of a doc_content_cache value.

        Bytes-view entries are ``List[bytes]``; sum part lengths.  Anything
        else (e.g. ``str`` text payloads) is treated as size 0 so it doesn't
        count toward the cap.
        """
        if not isinstance(value, list):
            return 0
        total = 0
        for part in value:
            if isinstance(part, (bytes, bytearray)):
                total += len(part)
        return total

    def _record_cache_insert(self, key: Tuple[str, str], value: Any) -> None:
        """Insert into doc_content_cache; FIFO-evict bytes payloads over the cap.

        Text payloads (``str``) skip the size accounting entirely so the cap
        never evicts them.  When the just-inserted entry is itself larger than
        the cap, eviction stops at that entry rather than removing it.
        """
        if self._bytes_cache_max is None:
            self.doc_content_cache[key] = value
            return
        size = self._value_bytes_size(value)
        self.doc_content_cache[key] = value
        if size == 0:
            return
        self._bytes_cache_used += size
        while self._bytes_cache_used > self._bytes_cache_max and self.doc_content_cache:
            evict_key = next(iter(self.doc_content_cache))
            if evict_key == key:
                # Don't evict what we just added; that would defeat the call.
                break
            evict_val = self.doc_content_cache.pop(evict_key)
            self._bytes_cache_used -= self._value_bytes_size(evict_val)

    async def get_doc_content(self, doc_id: str, view: str):
        """
        Retrieve and cache document content for a specific view.

        The cache is keyed by ``(view, doc_id)`` so different views of the same
        document are stored independently.  Bytes payloads are subject to the
        optional per-pipeline cap configured via
        ``bytes_content_cache_max_bytes``.
        """
        key = (view, doc_id)
        if key not in self.doc_content_cache:
            ret = await ProcessorRegistry[self.collection]["content"].submit(
                {"id": doc_id, "view": view}
            )
            if "error" in ret:
                raise RuntimeError(
                    f"Failed to retrieve content for document '{doc_id}' view '{view}': {ret['error']}"
                )
            # Text views populate 'text'; bytes views (PR5+) populate 'data' (List[bytes]).
            # Store whichever is set so rerank stages get the right payload shape.
            value = ret.get("text") if "text" in ret else ret.get("data")
            self._record_cache_insert(key, value)
        return self.doc_content_cache[key]

    async def run(
        self,
        query: str,
        last_output: Any = None,
        current_node: PipelineComponent = None,
        scratch: Dict[tuple, Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Recursively execute the pipeline for a single query.

        For most callers, only ``query`` needs to be provided — the remaining
        parameters are used internally during recursive traversal of the AST.

        The method dispatches on the type of ``current_node``:

        * :class:`~routir.pipeline.parser.CallSequence` — iterates over stages,
          passing ``last_output`` forward at each step.
        * :class:`~routir.pipeline.parser.ParallelCallSequences` — optionally runs
          the expander first, then fans out to all branches concurrently, and
          finally calls the merger with all ranked lists.
        * :class:`~routir.pipeline.parser.SystemCall` — submits one request to the
          appropriate processor and applies the ``%limit`` if set.

        Args:
            query (str): The user query string.
            last_output (dict, optional): Output dict from the previous pipeline
                stage, threaded automatically during recursion.  Callers should
                leave this as ``None``.  After a search stage it contains
                ``{"scores": {docid: float, …}}``; after an expander stage it
                contains ``{"queries": [str, …]}``.
            current_node (PipelineComponent, optional): The AST node to execute.
                Defaults to the root of the pipeline.  Set automatically during
                recursion; callers should leave this as ``None``.
            scratch (dict, optional): Internal cache of intermediate stage
                results, keyed by ``(alias, role, view)`` tuples.  Populated
                during execution so the same stage is not re-run if referenced
                multiple times; including ``view`` keeps the same alias+role
                under two different views from collapsing.  Callers should
                leave this as ``None``.

        Returns:
            dict: Result from the final pipeline stage. Typically contains:

            * ``"scores"`` — ``{docid: float}`` ranked results.
            * ``"cached"`` — whether the result was served from cache.
            * ``"timestamp"`` — Unix timestamp of processing.
        """
        current_node = current_node or self.pipeline
        last_output = last_output or {}
        scratch = scratch or {}

        if isinstance(current_node, CallSequence):
            for stage in current_node.stages:
                last_output = await self.run(query, last_output, stage, scratch)
            return last_output

        if isinstance(current_node, ParallelCallSequences):
            expanded_queries = [query]
            if current_node.expander is not None:
                last_output = await self.run(query, last_output, current_node.expander, scratch)
                expanded_queries = last_output["queries"]

            concurrent_run_outputs = await asyncio.gather(
                *[self.run(q, last_output, seq, scratch) for seq in current_node.sequences for q in expanded_queries]
            )

            merger_result = await self.run(
                query, {"scores": [o["scores"] for o in concurrent_run_outputs], **last_output}, current_node.merger, scratch
            )
            if current_node.expander is not None:
                merger_result["expanded_queries"] = expanded_queries
            return merger_result

        if not isinstance(current_node, SystemCall):
            raise RuntimeError(f"Expected SystemCall node, got {type(current_node).__name__}")

        payload = {"query": query, **last_output, **scratch, **self.runtime_kwargs.get(current_node.alias, {})}
        if current_node.limit is not None:
            payload["limit"] = current_node.limit

        processor = ProcessorRegistry[current_node.name][_role_to_service[current_node.role]]

        if current_node.role == "search":
            ret = await processor.submit(payload)
            if "scores" not in ret or not isinstance(ret["scores"], dict):
                raise RuntimeError(
                    f"Service '{current_node.name}' (search) returned unexpected format; "
                    f"expected dict 'scores', got keys: {list(ret.keys())}"
                )

        if current_node.role == "merger":
            ret = await processor.submit(payload)
            if "scores" not in ret or not isinstance(ret["scores"], dict):
                raise RuntimeError(
                    f"Service '{current_node.name}' (merger) returned unexpected format; "
                    f"expected dict 'scores', got keys: {list(ret.keys())}"
                )

        if current_node.role == "rerank":
            docid_to_rerank: List[str] = list(last_output["scores"].keys())

            # Resolve view: per-stage @view wins; otherwise use the collection's default.
            view = current_node.view
            if view is None:
                cp = ProcessorRegistry[self.collection]["content"]
                view = getattr(cp, "default_view", None)
            # verify() should have caught a missing default_view already; defensive guard:
            if view is None:
                raise RuntimeError(
                    f"rerank stage '{current_node.alias}' has no view and collection "
                    f"'{self.collection}' has no default_view"
                )

            logger.info(f"Gathering doc content for {len(docid_to_rerank)} documents (view='{view}')")
            # Text views return List[str]; bytes views (PR5+) return List[List[bytes]].
            # Either shape is handed straight to the engine via ``payload['passages']``.
            passages = await asyncio.gather(*[self.get_doc_content(d, view) for d in sorted(docid_to_rerank)])
            payload["passages"] = passages
            ret = await processor.submit(payload)
            if "scores" not in ret or not isinstance(ret["scores"], list):
                raise RuntimeError(
                    f"Service '{current_node.name}' (rerank) returned unexpected format; "
                    f"expected list 'scores', got keys: {list(ret.keys())}"
                )
            ret["scores"] = dict(zip(docid_to_rerank, ret["scores"]))

        if current_node.role == "expander":
            ret = await processor.submit(payload)
            if "queries" not in ret or not isinstance(ret["queries"], list):
                raise RuntimeError(
                    f"Service '{current_node.name}' (expander) returned unexpected format; "
                    f"expected list 'queries', got keys: {list(ret.keys())}"
                )

        # apply limit here just to safe
        if current_node.limit is not None:
            if "scores" in ret:
                ret["scores"] = dict_topk(ret["scores"], current_node.limit)
            elif "queries" in ret:
                ret["queries"] = ret["queries"][: current_node.limit]

        # Include view in the scratch key so the same alias+role under two
        # different views (e.g. ``kf-rerank@kf >> kf-rerank@ocr``) does not
        # collapse into a single entry.
        scratch[(current_node.alias, current_node.role, current_node.view)] = ret
        return ret
