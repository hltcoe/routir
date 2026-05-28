from typing import List, Optional

from ..utils import logger
from .rest import RestTransport
from .transport import RoutirClientError, Transport


_VALID_TRANSPORTS = ("auto", "grpc", "rest")


def _infer_tls(grpc_endpoint: Optional[str], explicit_tls: Optional[bool]) -> bool:
    """Decide whether to use TLS on the gRPC channel.

    Honors explicit_tls if not None; otherwise infers from a ``grpcs://``
    scheme prefix on grpc_endpoint. Default is False (plaintext h2c, which
    matches the user's nginx-fronts-TLS deployment).
    """
    if explicit_tls is not None:
        return explicit_tls
    if grpc_endpoint is None:
        return False
    return grpc_endpoint.startswith("grpcs://")


class AsyncClient:
    """Async client for a RoutIR endpoint.

    Supports REST and gRPC transports. ``transport='auto'`` will probe gRPC
    via a single ``Ping`` and fall back to REST if the probe fails for
    transport-level reasons (UNIMPLEMENTED, channel errors); auth failures
    (UNAUTHENTICATED, PERMISSION_DENIED) are raised, not papered over.
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
        if transport not in _VALID_TRANSPORTS:
            raise ValueError(
                f"transport must be one of {_VALID_TRANSPORTS}, got '{transport}'"
            )
        self.endpoint = endpoint
        self.grpc_endpoint = grpc_endpoint
        self.api_key = api_key
        self._transport_choice = transport
        self.timeout = timeout
        self.retries = retries
        self.tls = tls

        self._tx: Optional[Transport] = None
        self._transport_name: str = "unset"

    @property
    def transport(self) -> str:
        return self._transport_name

    async def _start(self) -> None:
        if self._tx is not None:
            return

        # Always have a REST transport available as the fallback target.
        rest_tx = RestTransport(
            endpoint=self.endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )
        await rest_tx.start()

        if self._transport_choice == "rest":
            self._tx = rest_tx
            self._transport_name = "rest"
            return

        # Both "grpc" and "auto" want gRPC if it's reachable.
        if not self.grpc_endpoint:
            if self._transport_choice == "grpc":
                await rest_tx.close()
                raise RoutirClientError(
                    "transport='grpc' requires grpc_endpoint to be set."
                )
            # auto, no grpc_endpoint → silent REST.
            self._tx = rest_tx
            self._transport_name = "rest"
            return

        try:
            from .grpc import GrpcTransport
        except ImportError as e:
            if self._transport_choice == "grpc":
                await rest_tx.close()
                raise RoutirClientError(
                    "transport='grpc' requested but grpcio is not installed. "
                    "Try: pip install 'routir[grpc]'"
                ) from e
            logger.warning(
                "grpcio not installed; falling back to REST. "
                "Install 'routir[grpc]' to enable the gRPC transport."
            )
            self._tx = rest_tx
            self._transport_name = "rest"
            return

        tls = _infer_tls(self.grpc_endpoint, self.tls)
        target = self.grpc_endpoint
        for prefix in ("grpcs://", "grpc://"):
            if target.startswith(prefix):
                target = target[len(prefix):]
                break

        grpc_tx = GrpcTransport(
            endpoint=target,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            tls=tls,
        )
        await grpc_tx.start()

        # Lazy probe — one Ping.
        try:
            await grpc_tx.ping()
        except RoutirClientError as e:
            # UNAUTHENTICATED / PERMISSION_DENIED is a config error, not a
            # transport problem; raise it instead of falling back. The
            # GrpcTransport contract is to embed the status code name in
            # the error message, so a simple substring match is sound.
            msg = str(e).lower()
            if "unauthenticated" in msg or "permission_denied" in msg:
                await grpc_tx.close()
                await rest_tx.close()
                raise
            # UNIMPLEMENTED / channel error → fall back to REST.
            if self._transport_choice == "grpc":
                await grpc_tx.close()
                await rest_tx.close()
                raise RoutirClientError(
                    f"transport='grpc' requested but probe failed: {e}"
                ) from e
            logger.warning(f"gRPC probe failed ({e}); falling back to REST.")
            await grpc_tx.close()
            self._tx = rest_tx
            self._transport_name = "rest"
            return

        await rest_tx.close()
        self._tx = grpc_tx
        self._transport_name = "grpc"

    async def reset_transport(self) -> None:
        if self._tx is not None:
            await self._tx.close()
            self._tx = None
        self._transport_name = "unset"

    async def close(self) -> None:
        if self._tx is not None:
            await self._tx.close()
            self._tx = None
        self._transport_name = "unset"

    async def __aenter__(self) -> "AsyncClient":
        await self._start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure(self) -> Transport:
        if self._tx is None:
            await self._start()
        return self._tx  # type: ignore[return-value]

    async def search(self, service: str, query: str, **kwargs) -> dict:
        tx = await self._ensure()
        payload = {"service": service, "query": query, **kwargs}
        return await tx.search(payload)

    async def score(self, service: str, query: str, passages: list, **kwargs) -> dict:
        tx = await self._ensure()
        payload = {"service": service, "query": query, "passages": passages, **kwargs}
        return await tx.score(payload)

    async def content(self, collection: str, id: str) -> dict:
        tx = await self._ensure()
        return await tx.content({"collection": collection, "id": id})

    async def pipeline(
        self,
        pipeline: str,
        query: str,
        collection: Optional[str] = None,
        runtime_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        tx = await self._ensure()
        payload: dict = {"pipeline": pipeline, "query": query, **kwargs}
        if collection is not None:
            payload["collection"] = collection
        if runtime_kwargs is not None:
            payload["runtime_kwargs"] = runtime_kwargs
        return await tx.pipeline(payload)

    async def avail(self) -> dict:
        tx = await self._ensure()
        return await tx.avail()

    async def ping(self) -> dict:
        tx = await self._ensure()
        return await tx.ping()

    async def search_batch(self, payloads: List[dict]) -> List[dict]:
        tx = await self._ensure()
        return await tx.search_batch(payloads)

    async def score_batch(self, payloads: List[dict]) -> List[dict]:
        tx = await self._ensure()
        return await tx.score_batch(payloads)


__all__ = ["AsyncClient", "RoutirClientError"]
