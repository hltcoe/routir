import time
from typing import Dict, List

from ..models import Engine
from .abstract import BatchProcessor


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
