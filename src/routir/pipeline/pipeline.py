import asyncio
from typing import Any, Dict, List

from ..processors.registry import ProcessorRegistry
from ..utils import dict_topk, logger
from .parser import CallSequence, ParallelCallSequences, PipelineComponent, SystemCall, parser


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
        self, pipeline: PipelineComponent, collection: str, runtime_kwargs: Dict[str, Dict[str, Any]] = None, verify: bool = True
    ):
        """
        Initialize search pipeline.

        Args:
            pipeline (PipelineComponent): Parsed pipeline AST, typically produced
                by :func:`~routir.pipeline.parser.parser.parse`.
            collection (str): Document collection name. Must be registered as a
                ``"content"`` service in
                :data:`~routir.processors.registry.ProcessorRegistry` when any
                reranking stage is present.
            runtime_kwargs (dict, optional): Per-alias extra parameters injected
                into individual service call payloads at execution time. Keys are
                pipeline aliases; values are dicts.  Example::

                    {"dense": {"instruction": "Retrieve relevant passages"}}

            verify (bool): If ``True`` (default), raise ``RuntimeError`` at
                construction time if any referenced service is missing from the
                registry, or if a reranking stage is present but no content
                service is configured for the collection.
        """
        self.pipeline = pipeline
        self.collection = collection
        self.runtime_kwargs = runtime_kwargs or {}
        self.doc_content_cache = {}

        if verify:
            self.verify()
        alias_not_found = set(self.runtime_kwargs.keys()) - set([c.alias for c in self.pipeline.all_calls])
        if alias_not_found:
            raise RuntimeError(f"Runtime kwargs reference unknown pipeline aliases: {alias_not_found}")

    def verify(self):
        """Verify that all required services exist in the registry."""
        if any(call.role == "rerank" for call in self.pipeline.all_calls):
            if not ProcessorRegistry.has_service(self.collection, "content"):
                raise RuntimeError(
                    f"Pipeline requires reranking but no content service found for collection '{self.collection}'"
                )
        for call in self.pipeline.all_calls:
            if not ProcessorRegistry.has_service(call.name, _role_to_service[call.role]):
                raise RuntimeError(
                    f"No {_role_to_service[call.role]} service registered under '{call.name}'"
                )

    @classmethod
    def from_string(
        cls, pipeline_string: str, collection: str, runtime_kwargs: Dict[str, Dict[str, Any]] = None, verify: bool = True
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

            collection (str): Document collection name for content lookup during
                reranking.
            runtime_kwargs (dict, optional): Per-alias extra parameters injected
                into service payloads at query time.  Keys are pipeline aliases
                (the name inside ``[...]``; defaults to service name).  Example::

                    {"d": {"instruction": "Given a query, retrieve passages…"}}

            verify (bool): Verify that all referenced services exist in the
                registry at construction time (default: ``True``).

        Returns:
            SearchPipeline: Ready-to-run pipeline instance.
        """
        return cls(parser.parse(pipeline_string), collection, runtime_kwargs, verify)

    async def get_doc_content(self, doc_id: str):
        """
        Retrieve and cache document content.

        Args:
            doc_id: Document identifier

        Returns:
            Document text content
        """
        if doc_id not in self.doc_content_cache:
            ret = await ProcessorRegistry[self.collection]["content"].submit({"id": doc_id})
            if "error" in ret:
                raise RuntimeError(f"Failed to retrieve content for document '{doc_id}': {ret['error']}")
            self.doc_content_cache[doc_id] = ret["text"]
        return self.doc_content_cache[doc_id]

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
                results, keyed by ``(alias, role)`` tuples.  Populated during
                execution so the same stage is not re-run if referenced multiple
                times.  Callers should leave this as ``None``.

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
            logger.info(f"Gathering doc content for {len(docid_to_rerank)} documents")
            doc_text_list = await asyncio.gather(*[self.get_doc_content(d) for d in sorted(docid_to_rerank)])
            payload["passages"] = doc_text_list
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

        scratch[(current_node.alias, current_node.role)] = ret
        return ret
