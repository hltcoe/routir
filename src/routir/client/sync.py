import asyncio
import threading
from typing import List, Optional

from .client import AsyncClient


class Client:
    """Synchronous facade around :class:`AsyncClient`.

    Owns a private asyncio loop running on a background thread. This avoids
    the ``asyncio.run`` pitfall of failing inside notebooks / already-running
    event loops, at the cost of one extra thread per client.
    """

    def __init__(
        self,
        endpoint: str,
        grpc_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        transport: str = "auto",
        timeout: float = 600,
        retries: int = 3,
        tls: Optional[bool] = None,
    ):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._async = AsyncClient(
            endpoint=endpoint,
            grpc_endpoint=grpc_endpoint,
            api_key=api_key,
            transport=transport,
            timeout=timeout,
            retries=retries,
            tls=tls,
        )

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    @property
    def transport(self) -> str:
        return self._async.transport

    def reset_transport(self) -> None:
        self._run(self._async.reset_transport())

    def search(self, service: str, query: str, **kwargs) -> dict:
        return self._run(self._async.search(service, query, **kwargs))

    def score(self, service: str, query: str, passages: list, **kwargs) -> dict:
        return self._run(self._async.score(service, query, passages, **kwargs))

    def content(self, collection: str, id: str, view: Optional[str] = None) -> dict:
        return self._run(self._async.content(collection, id, view=view))

    def pipeline(
        self,
        pipeline: str,
        query: str,
        collection: Optional[str] = None,
        runtime_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        return self._run(self._async.pipeline(pipeline, query, collection=collection, runtime_kwargs=runtime_kwargs, **kwargs))

    def avail(self) -> dict:
        return self._run(self._async.avail())

    def ping(self) -> dict:
        return self._run(self._async.ping())

    def search_batch(self, payloads: List[dict]) -> List[dict]:
        return self._run(self._async.search_batch(payloads))

    def score_batch(self, payloads: List[dict]) -> List[dict]:
        return self._run(self._async.score_batch(payloads))

    def close(self) -> None:
        try:
            self._run(self._async.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            if not self._loop.is_closed():
                self._loop.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
