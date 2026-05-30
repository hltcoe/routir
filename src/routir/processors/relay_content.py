"""Processor that forwards ``/content`` lookups to a remote RoutIR server.

Registered by :func:`routir.config.load.auto_add_relay_services` for every
content service advertised by a remote endpoint in ``server_imports``.  This
lets a local pipeline rerank against a collection hosted elsewhere without any
collection config or doc files locally.

Local caching is important here: a single rerank stage can fetch the top-N
documents (often 100 or more), so without caching every query pays the full
network round-trip per doc.  The per-run ``doc_content_cache`` on
:class:`~routir.pipeline.SearchPipeline` only dedupes within a single request.
"""

from typing import Any, Dict, Optional

from ..client import AsyncClient
from ..utils import logger
from .abstract import Processor


class RelayContentProcessor(Processor):
    """Forward content lookups to a remote RoutIR endpoint via :class:`AsyncClient`.

    The processor mirrors :class:`~routir.processors.content_processors.ContentProcessor`
    on the wire: cache key is the document ``id``, request shape is
    ``{"id": str}``, response shape includes ``text`` (and optionally ``title``,
    ``language``) on success, or ``error`` on failure.

    Args:
        collection (str): Remote collection name to query.
        endpoint (str): REST URL of the remote RoutIR server.
        grpc_endpoint (str, optional): gRPC target for the data plane.
        api_key (str, optional): Bearer token forwarded by the client.
        transport (str): One of ``"auto"`` (default), ``"grpc"``, ``"rest"``.
        timeout (float): Per-request timeout in seconds (default 600).
        retries (int): Client-level retry budget (default 10).
        tls (bool, optional): Explicit TLS flag for the gRPC channel.
        cache_size (int): Local LRU cache capacity; ``<= 0`` disables.
        cache_ttl (int): Local cache TTL in seconds.
        redis_url (str, optional): Use Redis instead of LRU when set.
        redis_kwargs (dict): Extra kwargs forwarded to the Redis client.
    """

    def __init__(
        self,
        collection: str,
        endpoint: str,
        grpc_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        transport: str = "auto",
        timeout: float = 600,
        retries: int = 10,
        tls: Optional[bool] = None,
        cache_size: int = -1,
        cache_ttl: int = 600,
        redis_url: Optional[str] = None,
        redis_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            cache_size=cache_size,
            cache_ttl=cache_ttl,
            cache_key=lambda x: x["id"],
            redis_url=redis_url,
            redis_kwargs=redis_kwargs or {},
        )
        self.collection = collection
        self.endpoint = endpoint
        self._client = AsyncClient(
            endpoint=endpoint,
            grpc_endpoint=grpc_endpoint,
            api_key=api_key,
            transport=transport,
            timeout=timeout,
            retries=retries,
            tls=tls,
        )

    async def _submit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = item["id"]
        try:
            return await self._client.content(self.collection, doc_id)
        except Exception as e:
            logger.exception(
                f"RelayContentProcessor failed fetching id={doc_id!r} from {self.endpoint} (collection={self.collection!r})"
            )
            return {"error": f"{type(e).__name__}: {e}"}
