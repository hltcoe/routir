import asyncio
from typing import Any, Dict, List

from ..processors.registry import ProcessorRegistry
from ..utils import dict_topk, logger
from .parser import CallSequence, ParallelCallSequences, PipelineComponent, SystemCall, parser


# TODO: probably should unify it...
_role_to_service = {"search": "search", "rerank": "score", "expander": "decompose_query", "merger": "fuse"}


class SearchPipeline:
    """
    Executes search pipelines defined using a custom DSL.

    Supports sequential and parallel execution of search, rerank, query expansion,
    and fusion operations.

    Attributes:
        pipeline: Parsed pipeline component tree
        collection: Document collection to search
        runtime_kwargs: Runtime parameters passed to pipeline components
        doc_content_cache: Cache for retrieved document content
    """

    def __init__(
        self, pipeline: PipelineComponent, collection: str, runtime_kwargs: Dict[str, Dict[str, Any]] = None, verify: bool = True
    ):
        """
        Initialize search pipeline.

        Args:
            pipeline: Parsed pipeline component
            collection: Collection name
            runtime_kwargs: Runtime parameters for pipeline components
            verify: Whether to verify all services exist
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
        Create pipeline from string specification.

        Args:
            pipeline_string: Pipeline DSL string
            collection: Collection name
            runtime_kwargs: Runtime parameters
            verify: Whether to verify services

        Returns:
            SearchPipeline instance
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
        Recursively execute the pipeline for a query.

        Args:
            query: Search query
            last_output: Output from previous stage
            current_node: Current pipeline component to execute
            scratch: Scratch space for intermediate results

        Returns:
            Final search results
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

            return await self.run(
                query, {"scores": [o["scores"] for o in concurrent_run_outputs], **last_output}, current_node.merger, scratch
            )

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
