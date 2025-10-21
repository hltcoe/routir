import asyncio
from typing import Any, Dict, List

from ..processors.registry import ProcessorRegistry
from ..utils import dict_topk
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
        assert len(alias_not_found) == 0, f"Cannot find alias {alias_not_found}"

    def verify(self):
        """Verify that all required services exist in the registry."""
        if any(call.role == "rerank" for call in self.pipeline.all_calls):
            assert ProcessorRegistry.has_service(self.collection, "content"), \
                f"Cannot find content service for collection `{self.collection}` but the pipeline involve reranking"
        for call in self.pipeline.all_calls:
            assert ProcessorRegistry.has_service(call.name, _role_to_service[call.role]), (
                f"Cannot find a {_role_to_service[call.role]} service under `{call.name}`"
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
            self.doc_content_cache[doc_id] = ret["text"]
        return self.doc_content_cache[doc_id]

    async def run(
        self,
        query: str,
        last_output: Dict[str, Any] = None,
        current_node: PipelineComponent = None,
        scratch: Dict[str, Dict[str, Any]] = None,
    ) -> List[Dict[str, float]]:
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

        assert isinstance(current_node, SystemCall)

        payload = {"query": query, **last_output, **scratch, **self.runtime_kwargs.get(current_node.alias, {})}
        if current_node.limit is not None:
            payload["limit"] = current_node.limit

        processor = ProcessorRegistry[current_node.name][_role_to_service[current_node.role]]

        if current_node.role == "search":
            ret = await processor.submit(payload)
            assert "scores" in ret and isinstance(ret["scores"], dict)

        if current_node.role == "merger":
            ret = await processor.submit(payload)
            assert "scores" in ret and isinstance(ret["scores"], dict)

        if current_node.role == "rerank":
            docid_to_rerank: List[str] = list(last_output["scores"].keys())
            doc_text_list = await asyncio.gather(*[self.get_doc_content(d) for d in docid_to_rerank])
            payload["passages"] = doc_text_list
            ret = await processor.submit(payload)
            assert "scores" in ret and isinstance(ret["scores"], list)
            ret["scores"] = dict(zip(docid_to_rerank, ret["scores"]))

        if current_node.role == "expander":
            ret = await processor.submit(payload)
            assert "queries" in ret and isinstance(ret["queries"], list)

        # apply limit here just to safe
        if current_node.limit is not None:
            if "scores" in ret:
                ret["scores"] = dict_topk(ret["scores"], current_node.limit)
            elif "queries" in ret:
                ret["queries"] = ret["queries"][: current_node.limit]

        scratch[(current_node.alias, current_node.role)] = ret
        return ret
