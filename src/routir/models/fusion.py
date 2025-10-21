import asyncio
from typing import Callable, Dict, List

from ..processors.registry import auto_register
from ..utils import dict_topk
from .abstract import Engine


def _rrf(score_list: List[Dict[str, float]], smoothing_k=0) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion.

    Args:
        score_list: List of result dicts to fuse
        smoothing_k: Smoothing parameter (default 0)

    Returns:
        Fused scores dict
    """
    fused = {}
    for scores in score_list:
        for i, (docid, s) in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
            if docid not in fused:
                fused[docid] = 0
            fused[docid] += 1 / (i + 1 + smoothing_k)

    return fused


def _score_fusion(score_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Simple score fusion by summing scores.

    Args:
        score_list: List of result dicts to fuse

    Returns:
        Fused scores dict
    """
    fused = {}
    for scores in score_list:
        for docid, s in scores.items():
            if docid not in fused:
                fused[docid] = 0
            fused[docid] += s

    return fused


_fusion_functions: Dict[str, Callable] = {
    "RRF": _rrf,
    "score": _score_fusion,
}


class Fusion(Engine):
    """
    Engine that fuses results from multiple upstream engines.

    Retrieves results from multiple engines and combines them using
    a fusion method (RRF or score-based).

    Attributes:
        upstream: List of upstream engines
        fusion_function: Function to use for fusion
        fusion_args: Arguments for the fusion function
    """

    def __init__(self, name=None, config=None, **kwargs):
        """
        Initialize fusion engine.

        Args:
            name: Engine name
            config: Must contain 'upstream_service' list
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        self.upstream: List[Engine] = []
        if not isinstance(self.config["upstream_service"], list):
            self.config["upstream_service"] = [self.config["upstream_service"]]
        for c in self.config["upstream_service"]:
            self.upstream.append(Engine.load(c["engine"], config=c["config"], **kwargs))

        assert len(self.upstream) > 0, "Need to have at least one upstream service"

        self.fusion_function = _fusion_functions[self.config.get("fusion_method", "RRF")]
        self.fusion_args = self.config.get("fusion_args", {})

    async def search_batch(self, queries, limit=20, **kwargs) -> List[Dict[str, float]]:
        if not isinstance(limit, list):
            limit = [limit] * len(queries)
        assert len(limit) == len(queries)

        upstream_results = await asyncio.gather(
            *[service.search_batch(queries, limit=limit, **kwargs) for service in self.upstream]
        )

        return [
            dict_topk(self.fusion_function([results[qid] for results in upstream_results], **self.fusion_args), k)
            for qid, k in enumerate(limit)
        ]


@auto_register("fuse")
class RRF(Engine):
    """Reciprocal Rank Fusion engine for combining result lists."""

    async def fuse_batch(self, queries, batch_rankings, **kwargs):
        """Fuse multiple rankings using RRF algorithm."""
        return [_rrf(rankings, int(kwargs.get("rrf_k", 0))) for rankings in batch_rankings]


@auto_register("fuse")
class ScoreFusion(Engine):
    """Score-based fusion engine for combining result lists."""

    async def fuse_batch(self, queries, batch_rankings, **kwargs):
        """Fuse multiple rankings by summing scores."""
        return [_score_fusion(rankings) for rankings in batch_rankings]
