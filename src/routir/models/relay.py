import asyncio
from typing import Any, Dict, List, Optional

from ..client import AsyncClient
from ..processors.registry import ProcessorRegistry
from .abstract import Engine


class Relay(Engine):
    """Forward search/score requests to a remote (or local) RoutIR service.

    When ``endpoint`` is set in config, the relay uses an :class:`AsyncClient`
    that talks gRPC if available (``grpc_endpoint``) and otherwise REST.
    When ``endpoint`` is omitted, the relay dispatches to a local processor
    via :class:`ProcessorRegistry`.

    Recognized config fields:

    * ``service`` (required): name of the remote/local service to call.
    * ``endpoint``: REST base URL of a remote RoutIR server (e.g.
      ``http://host:5000``). If omitted, dispatch goes through the local
      :class:`ProcessorRegistry`.
    * ``grpc_endpoint``: optional gRPC target (e.g. ``host:50051``) used by the
      underlying :class:`AsyncClient`; falls back to REST if unreachable.
    * ``api_key``: Bearer token forwarded by the client.
    * ``transport``: one of ``"auto"`` (default), ``"grpc"``, ``"rest"``.
    * ``tls``: explicit TLS flag for the gRPC channel; inferred from the
      ``grpcs://`` scheme on ``grpc_endpoint`` if not set.
    * ``timeout``: per-request timeout in seconds (default 600).
    * ``retries``: client-level retry budget (default 10).
    * ``other_request_kwargs``: dict merged into every forwarded payload.
    """

    def __init__(self, name: str = None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        if "service" not in self.config:
            raise RuntimeError("Relay config is missing required 'service' field")

        self.other_kwargs = self.config.get("other_request_kwargs", {})

        # Remote relay: hold an AsyncClient. Started lazily on the first call so
        # constructing the Relay does not open network sockets.
        self._client: Optional[AsyncClient] = None
        if "endpoint" in self.config:
            self._client = AsyncClient(
                endpoint=self.config["endpoint"],
                grpc_endpoint=self.config.get("grpc_endpoint"),
                api_key=self.config.get("api_key"),
                transport=self.config.get("transport", "auto"),
                timeout=self.config.get("timeout", 600),
                retries=self.config.get("retries", 10),
                tls=self.config.get("tls"),
            )

    async def _submit_payload(self, service_type: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._client is not None:
            method = {"search": self._client.search_batch, "score": self._client.score_batch}.get(service_type)
            if method is None:
                raise RuntimeError(f"Relay does not support service_type '{service_type}'")
            resps = await method(payloads)
        else:
            if not ProcessorRegistry.has_service(self.config["service"], service_type):
                raise RuntimeError(
                    f"Local service '{self.config['service']}' does not have type '{service_type}'"
                )
            local_processor = ProcessorRegistry.get(self.config["service"], service_type)
            resps = await asyncio.gather(*[local_processor.submit(load) for load in payloads])

        for resp, payload in zip(resps, payloads):
            if resp.get("query") != payload["query"]:
                raise RuntimeError(
                    f"Response/payload query mismatch from {self.config.get('endpoint', 'local')}: "
                    f"expected '{payload['query']}', got '{resp.get('query')}'"
                )
        return [resp.get("scores", {}) for resp in resps]

    async def search_batch(self, queries, subsets=None, **kwargs):
        if subsets is None:
            subsets = ["none"] * len(queries)
        if len(subsets) != len(queries):
            raise RuntimeError(f"len(subsets)={len(subsets)} does not match len(queries)={len(queries)}")

        for key in kwargs:
            if isinstance(kwargs[key], list):
                if len(kwargs[key]) != len(queries):
                    raise RuntimeError(
                        f"kwarg '{key}' has length {len(kwargs[key])} but expected {len(queries)} (one per query)"
                    )
            else:
                kwargs[key] = [kwargs[key]] * len(queries)

        payloads = [
            {
                "query": queries[i],
                "service": self.config["service"],
                "subset": subsets[i],
                **self.other_kwargs,
                **{k: kwargs[k][i] for k in kwargs},
            }
            for i in range(len(queries))
        ]
        return await self._submit_payload("search", payloads)

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        if candidate_length is None:
            candidate_length = [len(passages)]
        if len(candidate_length) != len(queries):
            raise RuntimeError(
                f"len(candidate_length)={len(candidate_length)} does not match len(queries)={len(queries)}"
            )
        if sum(candidate_length) != len(passages):
            raise RuntimeError(
                f"sum(candidate_length)={sum(candidate_length)} does not match len(passages)={len(passages)}"
            )

        payloads = []
        start = 0
        for query, l in zip(queries, candidate_length):
            payloads.append({
                "query": query,
                "service": self.config["service"],
                "passages": passages[start : start + l],
            })
            start += l

        return await self._submit_payload("score", payloads)
