import time
from typing import Any, Dict, List

from ..models import Engine
from .abstract import BatchProcessor, Processor


class AsyncPairwiseScoreProcessor(Processor):
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

        ranking = (await self.engine.score_batch([query], passages=item.pop("passages")))[0]

        # Process each item in the batch
        return {"query": query, "scores": ranking, "service": self.engine.name, "processed": True, "timestamp": time.time()}


class BatchPairwiseScoreProcessor(BatchProcessor):
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
        passages = [item.get("passages", []) for item in batch]

        batch_scores = await self.engine.score_batch(queries, sum(passages, []), list(map(len, passages)))

        # Process each item in the batch
        results = []
        for item, scores in zip(batch, batch_scores):
            result = {
                "meta": {"n_passages": len(item["passages"])},
                "query": item["query"],
                "scores": scores,
                "service": self.engine.name,
                "processed": True,
                "timestamp": time.time(),
            }
            results.append(result)

        return results
