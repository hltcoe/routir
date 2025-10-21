import asyncio
from typing import Any, Dict, List

import aiohttp

from ..processors.registry import ProcessorRegistry
from ..utils import session_request
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

        assert "service" in self.config

        self.other_kwargs = self.config.get("other_request_kwargs", {})
        # TODO: should support some runtime config like retry and timeout
        # TODO: support list of endpoints for load balancing

    async def _submit_payload(self, service_type, payloads: List[Dict[str, Any]]):
        if "endpoint" in self.config:
            async with aiohttp.ClientSession() as session:
                resps = await asyncio.gather(
                    *[session_request(session, f"{self.config['endpoint']}/{service_type}", load) for load in payloads]
                )
        else:
            assert ProcessorRegistry.has_service(self.config["service"], service_type)
            local_processor = ProcessorRegistry.get(self.config["service"], service_type)
            resps = await asyncio.gather(*[local_processor.submit(load) for load in payloads])

        for resp, payload in zip(resps, payloads):
            assert resp["query"] == payload['query']
        return [
            # for backward compatiblity if the service is using `result` as key
            resp.get("scores", resp.get("result", {})) for resp in resps
        ]


    async def search_batch(self, queries, subsets=None, **kwargs):
        if subsets is None:
            subsets = ["none"] * len(queries)
        assert len(subsets) == len(queries)

        for key in kwargs:
            if isinstance(kwargs[key], list):
                assert len(kwargs[key]) == len(queries)
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
        assert len(candidate_length) == len(queries)
        assert sum(candidate_length) == len(passages)

        payloads = []
        start = 0
        for query, l in zip(queries, candidate_length):
            payloads.append({
                "query": query,
                "service": self.config["service"],
                "passages": passages[start: start+l]
            })
            start = start + 1

        return await self._submit_payload("score", payloads)
