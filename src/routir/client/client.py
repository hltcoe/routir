from typing import List, Optional, Tuple
from urllib.parse import urlparse

from ..utils import logger
from .rest import RestTransport
from .transport import RoutirClientError, Transport


_VALID_TRANSPORTS = ("auto", "grpc", "rest")
_REST_SCHEMES = ("http://", "https://")
_GRPC_SCHEMES = ("grpc://", "grpcs://")


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


def _split_endpoint(
    endpoint: Optional[str], grpc_endpoint: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Route a single endpoint into (rest_url, grpc_target) by URL scheme.

    Rules:
        - ``http://`` / ``https://`` -> REST.
        - ``grpc://`` / ``grpcs://`` -> gRPC (only used if ``grpc_endpoint``
          wasn't already supplied explicitly).
        - No scheme -> treated as REST and warned, to preserve the original
          positional-arg semantics.

    Both outputs may be None; the caller validates that at least one
    transport is reachable.
    """
    rest_url: Optional[str] = None
    grpc_target: Optional[str] = grpc_endpoint

    if endpoint is not None:
        if endpoint.startswith(_REST_SCHEMES):
            rest_url = endpoint
        elif endpoint.startswith(_GRPC_SCHEMES):
            if grpc_target is None:
                grpc_target = endpoint
                logger.warning(
                    f"Inferred gRPC transport from URL scheme of endpoint={endpoint!r}."
                )
            else:
                logger.warning(
                    f"Ignoring endpoint={endpoint!r} (gRPC scheme) because "
                    f"grpc_endpoint={grpc_endpoint!r} was supplied explicitly."
                )
        else:
            rest_url = endpoint
            logger.warning(
                f"endpoint={endpoint!r} has no URL scheme; assuming REST. "
                f"Use http://, https://, grpc://, or grpcs:// to be explicit."
            )

    return rest_url, grpc_target


class AsyncClient:
    """Async client for a RoutIR endpoint.

    Either ``endpoint`` (REST) or ``grpc_endpoint`` must be supplied — when
    only one is given, the client uses just that transport. The first
    positional ``endpoint`` accepts any URL scheme: ``http://`` / ``https://``
    route to the REST slot, ``grpc://`` / ``grpcs://`` route to the gRPC slot.

    ``transport='auto'`` (default):
        - With both endpoints, probe gRPC via a single ``Ping`` and fall
          back to REST on transport-level failures (UNIMPLEMENTED, channel
          errors). Auth failures are surfaced, not papered over.
        - With only a REST endpoint, query ``/avail`` once; if the server
          advertises ``grpc_port`` and the REST URL is ``http://`` (i.e. not
          fronted by a reverse proxy), auto-discover the gRPC target and
          probe it. Logs a warning when this happens.
        - With only a gRPC endpoint, use gRPC unconditionally. No REST
          fallback target exists.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
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

        rest_url, grpc_target = _split_endpoint(endpoint, grpc_endpoint)
        if rest_url is None and grpc_target is None:
            raise ValueError(
                "must supply at least one of endpoint or grpc_endpoint"
            )

        self.endpoint = rest_url
        self.grpc_endpoint = grpc_target
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

    def _make_rest(self) -> RestTransport:
        return RestTransport(
            endpoint=self.endpoint,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
        )

    def _make_grpc(self):
        from .grpc import GrpcTransport

        tls = _infer_tls(self.grpc_endpoint, self.tls)
        target = self.grpc_endpoint
        for prefix in _GRPC_SCHEMES:
            if target.startswith(prefix):
                target = target[len(prefix):]
                break
        return GrpcTransport(
            endpoint=target,
            api_key=self.api_key,
            timeout=self.timeout,
            retries=self.retries,
            tls=tls,
        )

    async def _autodiscover_grpc(self, rest_tx: RestTransport) -> None:
        """Probe /avail for a server-advertised gRPC port; set self.grpc_endpoint.

        Skipped silently if the REST URL is ``https://`` (likely nginx-fronted,
        where the advertised port won't be externally reachable) or if the
        server doesn't advertise a port. Failures of the probe itself are
        debug-logged, not raised — auto-discovery is best-effort.
        """
        if self.endpoint is None or not self.endpoint.startswith("http://"):
            return
        try:
            avail = await rest_tx.avail()
        except Exception as e:
            logger.debug(f"Auto-discovery: /avail probe failed: {e}")
            return
        port = avail.get("grpc_port")
        if not port:
            return
        host = urlparse(self.endpoint).hostname
        if not host:
            return
        self.grpc_endpoint = f"{host}:{port}"
        logger.warning(
            f"Auto-discovered gRPC at {self.grpc_endpoint} via /avail; "
            f"using gRPC for this client. Pass transport='rest' to opt out."
        )

    async def _start(self) -> None:
        if self._tx is not None:
            return

        # transport='rest': REST endpoint must be set; build it directly.
        if self._transport_choice == "rest":
            if self.endpoint is None:
                raise RoutirClientError(
                    "transport='rest' requires a REST endpoint (http:// or https://)."
                )
            tx = self._make_rest()
            await tx.start()
            self._tx = tx
            self._transport_name = "rest"
            return

        # transport='grpc': gRPC endpoint must be set; no REST fallback.
        if self._transport_choice == "grpc":
            if self.grpc_endpoint is None:
                raise RoutirClientError(
                    "transport='grpc' requires a gRPC endpoint (grpc://, grpcs://, or grpc_endpoint=...)."
                )
            try:
                tx = self._make_grpc()
                await tx.start()
                await tx.ping()
            except RoutirClientError as e:
                if tx is not None:
                    await tx.close()
                raise RoutirClientError(
                    f"transport='grpc' requested but probe failed: {e}"
                ) from e
            except ImportError as e:
                raise RoutirClientError(
                    "transport='grpc' requested but grpcio is not installed. "
                    "Try: pip install 'routir[grpc]'"
                ) from e
            self._tx = tx
            self._transport_name = "grpc"
            return

        # transport='auto'
        rest_tx: Optional[RestTransport] = None
        grpc_tx = None

        try:
            # Auto-discover gRPC via /avail when only REST is set.
            if self.endpoint is not None and self.grpc_endpoint is None:
                rest_tx = self._make_rest()
                await rest_tx.start()
                await self._autodiscover_grpc(rest_tx)

            if self.grpc_endpoint:
                try:
                    grpc_tx = self._make_grpc()
                    await grpc_tx.start()
                    await grpc_tx.ping()
                except RoutirClientError as e:
                    msg = str(e).lower()
                    if "unauthenticated" in msg or "permission_denied" in msg:
                        # Auth failure is a config error; do not silently fall back.
                        if grpc_tx is not None:
                            await grpc_tx.close()
                        if rest_tx is not None:
                            await rest_tx.close()
                        raise
                    logger.warning(f"gRPC probe failed ({e}); falling back to REST.")
                    if grpc_tx is not None:
                        await grpc_tx.close()
                        grpc_tx = None
                except ImportError:
                    logger.warning(
                        "grpcio not installed; falling back to REST. "
                        "Install 'routir[grpc]' to enable the gRPC transport."
                    )
                    grpc_tx = None
                else:
                    # gRPC succeeded.
                    if rest_tx is not None:
                        await rest_tx.close()
                    self._tx = grpc_tx
                    self._transport_name = "grpc"
                    return

            # gRPC unavailable (not configured, probe failed, or import missing).
            if self.endpoint is None:
                raise RoutirClientError(
                    "gRPC endpoint is unreachable and no REST endpoint was given to fall back to."
                )
            if rest_tx is None:
                rest_tx = self._make_rest()
                await rest_tx.start()
            self._tx = rest_tx
            self._transport_name = "rest"
        except BaseException:
            # Make sure no transport is left half-open on any error path.
            if rest_tx is not None and self._tx is not rest_tx:
                try:
                    await rest_tx.close()
                except Exception:
                    pass
            if grpc_tx is not None and self._tx is not grpc_tx:
                try:
                    await grpc_tx.close()
                except Exception:
                    pass
            raise

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

    async def content(self, collection: str, id: str, view: Optional[str] = None) -> dict:
        tx = await self._ensure()
        payload = {"collection": collection, "id": id}
        if view is not None:
            payload["view"] = view
        return await tx.content(payload)

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
