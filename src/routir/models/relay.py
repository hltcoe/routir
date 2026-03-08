import asyncio
from typing import Any, Dict, List

import aiohttp

from ..processors.registry import ProcessorRegistry
from ..utils import logger, session_request
from .abstract import Engine


class Relay(Engine):
    """
    Relay engine that forwards requests to remote or local services.

    Can relay to either HTTP endpoints or local processors, enabling
    distributed search architectures and service composition.

    Attributes:
        other_kwargs: Additional parameters to include in forwarded requests
    """

    def __init__(self, name: str = None, config=None, **kwargs):
        """
        Initialize the relay engine.

        Args:
            name: Engine name
            config: Must contain 'service' key; optionally 'endpoint' for remote services
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        if "service" not in self.config:
            raise RuntimeError("Relay config is missing required 'service' field")

        self.timeout = aiohttp.ClientTimeout(total=self.config.get("timeout", 600))
        self.retries = self.config.get("retries", 10)
        self.other_kwargs = self.config.get("other_request_kwargs", {})
        # TODO: should support some runtime config like retry and timeout
        # TODO: support list of endpoints for load balancing
        # TODO: implement relay for collections (content endpoint) so remote document
        #       stores can be used transparently by pipelines and rerankers

    async def _submit_payload(self, service_type, payloads: List[Dict[str, Any]]):
        if "endpoint" in self.config:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                for iretry in range(self.retries):
                    resps = await asyncio.gather(
                        *[session_request(session, f"{self.config['endpoint']}/{service_type}", load) for load in payloads]
                    )
                    if all(r is not None for r in resps):
                        break
                    n_failed = sum(1 for r in resps if r is None)
                    logger.warning(f"Retrying ({iretry+1}/{self.retries}) {self.config['endpoint']}/{service_type}: {n_failed}/{len(resps)} requests failed")
                    await asyncio.sleep(0.01)
                else:
                    n_failed = sum(1 for r in resps if r is None)
                    raise RuntimeError(
                        f"{n_failed}/{len(resps)} requests to {self.config['endpoint']}/{service_type} "
                        f"failed after {self.retries} retries"
                    )
        else:
            if not ProcessorRegistry.has_service(self.config["service"], service_type):
                raise RuntimeError(f"Local service '{self.config['service']}' does not have type '{service_type}'")
            local_processor = ProcessorRegistry.get(self.config["service"], service_type)
            resps = await asyncio.gather(*[local_processor.submit(load) for load in payloads])

        for resp, payload in zip(resps, payloads):
            if resp["query"] != payload["query"]:
                raise RuntimeError(
                    f"Response/payload query mismatch from {self.config.get('endpoint', 'local')}: "
                    f"expected '{payload['query']}', got '{resp['query']}'"
                )
        return [
            # for backward compatiblity if the service is using `result` as key
            resp.get("scores", resp.get("result", {})) for resp in resps
        ]


    async def search_batch(self, queries, subsets=None, **kwargs):
        if subsets is None:
            subsets = ["none"] * len(queries)
        if len(subsets) != len(queries):
            raise RuntimeError(f"len(subsets)={len(subsets)} does not match len(queries)={len(queries)}")

        for key in kwargs:
            if isinstance(kwargs[key], list):
                if len(kwargs[key]) != len(queries):
                    raise RuntimeError(f"kwarg '{key}' has length {len(kwargs[key])} but expected {len(queries)} (one per query)")
            else:
                kwargs[key] = [kwargs[key]] * len(queries)

        return await self._submit_payload("search", [
            {
                "query": queries[i],
                "service": self.config["service"],
                "subset": subsets[i],
                **self.other_kwargs,
                **{k: kwargs[k][i] for k in kwargs},
            }
            for i in range(len(queries))
        ])


    async def score_batch(self, queries, passages, candidate_length = None, **kwargs):
        if candidate_length is None:
            candidate_length = [len(passages)]
        if len(candidate_length) != len(queries):
            raise RuntimeError(f"len(candidate_length)={len(candidate_length)} does not match len(queries)={len(queries)}")
        if sum(candidate_length) != len(passages):
            raise RuntimeError(f"sum(candidate_length)={sum(candidate_length)} does not match len(passages)={len(passages)}")

        payloads = []
        start = 0
        for query, l in zip(queries, candidate_length):
            payloads.append({
                "query": query,
                "service": self.config["service"],
                "passages": passages[start: start+l]
            })
            start = start + l

        return await self._submit_payload("score", payloads)
