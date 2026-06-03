import time
from typing import Any, Dict, List

from ..models import Engine
from .abstract import BatchProcessor, Processor


class AsyncQueryProcessor(Processor):
    """Processor that serve every request independently through async calls.
    Should be used when all the engine does is issuing async calls
    """

    def __init__(self, engine: Engine, cache_size=1024, cache_ttl=600, cache_key=None, **kwargs):
        super().__init__(cache_size, cache_ttl, cache_key)
        self.engine = engine

    async def _submit(self, item: Dict[str, Any]):
        """
        Process one single item
        """

        query = item.pop("query")
        limit = int(item.pop("limit", 1000))
        # TODO: This is getting increasingly annoying... should just use subset instead of subsets
        subsets = item.pop("subset", None)

        ranking = (await self.engine.search_batch([query], limit=[limit], subsets=[subsets]))[0]

        # Process each item in the batch
        return {"query": query, "scores": ranking, "service": self.engine.name, "processed": True, "timestamp": time.time()}


class BatchQueryProcessor(BatchProcessor):
    def __init__(self, engine: Engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of items.
        Override this method for specific batch processing logic.
        """
        # Simulate processing time (e.g., model inference)
        # await asyncio.sleep(0.5)
        queries = [item.get("query", "") for item in batch]
        limits = [int(item.get("limit", 1000)) for item in batch]
        subsets = [item.get("subset", None) for item in batch]
        rankings = await self.engine.search_batch(queries, limit=limits, subsets=subsets)

        # Process each item in the batch
        results = []
        for item, ranking in zip(batch, rankings):
            # Example processing - just compute text length
            query = item.get("query", "")

            result = {"query": query, "scores": ranking, "service": self.engine.name, "processed": True, "timestamp": time.time()}
            results.append(result)

        return results


class BatchDecomposeQueryProcessor(BatchProcessor):
    def __init__(self, engine: Engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        queries = [item.get("query", "") for item in batch]
        limits = [int(item["limit"]) if "limit" in item else None for item in batch]
        # Collect extra kwargs (e.g. prompt) from the first item — batch items
        # originate from the same pipeline stage so they share the same kwargs.
        extra = {k: v for k, v in batch[0].items() if k not in ("query", "limit", "service", "subset", "scores")}

        sub_queries_list = await self.engine.decompose_query_batch(queries, limit=limits, **extra)

        return [
            {
                "query": item["query"],
                "queries": sub_queries,
                "service": self.engine.name,
                "processed": True,
                "timestamp": time.time(),
            }
            for item, sub_queries in zip(batch, sub_queries_list)
        ]
