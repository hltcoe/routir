import asyncio
import uuid
from typing import List, Optional

from ..utils import logger
from .transport import RoutirClientError, Transport


# Match RestTransport's retry shape so callers see the same wait pattern
# regardless of transport: 0.1s -> 0.2s -> 0.4s ... capped at 2s.
_BACKOFF_START = 0.1
_BACKOFF_CAP = 2.0


# These are the gRPC StatusCode names (we compare by name rather than by
# enum value so the lazy-import story stays clean — we only need to import
# grpc inside ``start``/``_call``).
_RETRYABLE_CODE_NAMES = {
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "RESOURCE_EXHAUSTED",
    "ABORTED",
}


class GrpcTransport(Transport):
    """gRPC transport.

    Mirrors :class:`RestTransport`'s call-shape: every method takes the same
    dict ``payload`` the REST API accepts and returns a dict identical to
    what REST returns. This lets :class:`AsyncClient` swap transports
    without callers caring.

    Channel construction (TLS vs insecure, message-size limits) happens in
    :meth:`start`; the constructor only stores config. ``grpcio`` is
    imported lazily so users without the ``[grpc]`` extra can still
    ``from routir.client.grpc import GrpcTransport`` without ImportError.

    Retry policy:
        Retries on UNAVAILABLE / DEADLINE_EXCEEDED / RESOURCE_EXHAUSTED /
        ABORTED with the same backoff as REST. Re-raises immediately on
        INVALID_ARGUMENT / NOT_FOUND / PERMISSION_DENIED / UNAUTHENTICATED
        / UNIMPLEMENTED / FAILED_PRECONDITION.

    Error-message contract:
        Every :class:`RoutirClientError` raised by this transport includes
        the gRPC status-code name (e.g. ``"gRPC UNAUTHENTICATED: ..."``).
        :class:`routir.client.client.AsyncClient` relies on this when
        deciding whether to fall back to REST.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 600,
        retries: int = 3,
        tls: bool = False,
        max_message_mb: int = 64,
    ):
        # gRPC channel targets are host:port, not URLs. Strip the schemes
        # callers might paste so they can use ``grpc://`` / ``grpcs://`` as
        # a hint without us choking on it.
        for prefix in ("grpcs://", "grpc://"):
            if endpoint.startswith(prefix):
                endpoint = endpoint[len(prefix):]
                break
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.tls = tls
        self.max_message_mb = max_message_mb

        self._channel = None
        self._stub = None

    async def start(self) -> None:
        if self._stub is not None:
            return

        try:
            import grpc  # noqa: F401
            from ..proto._generated import routir_pb2_grpc
        except ImportError as e:
            raise RoutirClientError(
                "GrpcTransport requires grpcio. Try: pip install 'routir[grpc]'"
            ) from e

        max_bytes = self.max_message_mb * 1024 * 1024
        options = [
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ]
        if self.tls:
            self._channel = grpc.aio.secure_channel(
                self.endpoint, grpc.ssl_channel_credentials(), options=options
            )
        else:
            self._channel = grpc.aio.insecure_channel(self.endpoint, options=options)
        self._stub = routir_pb2_grpc.RoutirStub(self._channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
        self._stub = None

    def _metadata(self) -> List[tuple]:
        md = [("x-request-id", str(uuid.uuid4()))]
        if self.api_key:
            md.append(("authorization", f"Bearer {self.api_key}"))
        return md

    async def _call(self, rpc_callable, request):
        """Run a unary RPC with the shared retry/error policy.

        ``rpc_callable`` is e.g. ``self._stub.Ping``. We pass the request,
        metadata, and timeout in here so each callsite stays compact.
        """
        if self._stub is None:
            raise RoutirClientError(
                "GrpcTransport used before start(); call await transport.start() first."
            )

        import grpc

        backoff = _BACKOFF_START
        total_attempts = self.retries + 1
        last_exc = None

        for attempt in range(total_attempts):
            try:
                return await rpc_callable(
                    request, metadata=self._metadata(), timeout=self.timeout
                )
            except grpc.aio.AioRpcError as e:
                code = e.code()
                code_name = code.name if code is not None else "UNKNOWN"
                details = e.details() or ""
                last_exc = e
                if code_name in _RETRYABLE_CODE_NAMES and attempt < total_attempts - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_CAP)
                    continue
                # Non-retryable status, or budget exhausted on a retryable
                # one — raise with the code name in the message so callers
                # (notably AsyncClient's fallback logic) can pattern-match.
                msg = f"gRPC {code_name}: {details}" if details else f"gRPC {code_name}"
                if code_name not in _RETRYABLE_CODE_NAMES:
                    raise RoutirClientError(msg) from e
                method_name = (
                    rpc_callable._method.decode() if hasattr(rpc_callable, "_method") else str(rpc_callable)
                )
                logger.error(
                    f"GrpcTransport: {method_name} failed after {total_attempts} attempts; "
                    f"last status {code_name}: {details}"
                )
                raise RoutirClientError(
                    f"{msg} (after {total_attempts} attempts)"
                ) from e
            except Exception as e:
                # Channel-level failure (DNS, connection refused before a
                # status arrives, etc.). Retry like UNAVAILABLE.
                last_exc = e
                if attempt < total_attempts - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_CAP)
                    continue
                logger.exception(
                    f"GrpcTransport: channel error after {total_attempts} attempts: {type(e).__name__}: {e}"
                )
                raise RoutirClientError(
                    f"gRPC channel error: {type(e).__name__}: {e}"
                ) from e

        # Defensive: loop should always return or raise above.
        raise RoutirClientError(
            f"gRPC call exhausted retries: {type(last_exc).__name__ if last_exc else 'unknown'}"
        )

    @staticmethod
    def _pack_extras(extras: dict):
        from google.protobuf.struct_pb2 import Struct

        s = Struct()
        s.update(extras)
        return s

    async def ping(self) -> dict:
        from ..proto._generated import routir_pb2 as pb

        resp = await self._call(self._stub.Ping, pb.PingRequest())
        return {"status": resp.status}

    async def avail(self) -> dict:
        from ..proto._generated import routir_pb2 as pb

        resp = await self._call(self._stub.Avail, pb.AvailRequest())
        out = {role: list(sl.items) for role, sl in resp.services.items()}
        out["pipeline_aliases"] = dict(resp.pipeline_aliases)
        if resp.HasField("grpc_port"):
            out["grpc_port"] = resp.grpc_port
        return out

    async def search(self, payload: dict) -> dict:
        from ..proto._generated import routir_pb2 as pb

        req = pb.SearchRequest(
            service=payload.get("service", ""),
            query=payload.get("query", ""),
        )
        if "limit" in payload and payload["limit"] is not None:
            req.limit = int(payload["limit"])
        if "subset" in payload and payload["subset"] is not None:
            req.subset = payload["subset"]
        if "instruction" in payload and payload["instruction"] is not None:
            req.instruction = payload["instruction"]

        extras = {
            k: v for k, v in payload.items()
            if k not in ("service", "query", "limit", "subset", "instruction")
        }
        if extras:
            req.extras.CopyFrom(self._pack_extras(extras))

        resp = await self._call(self._stub.Search, req)
        return {
            "query": resp.query,
            "scores": dict(resp.scores),
            "service": resp.service,
            "cached": resp.cached,
            "timestamp": resp.timestamp,
        }

    async def score(self, payload: dict) -> dict:
        from ..proto._generated import routir_pb2 as pb

        req = pb.ScoreRequest(
            service=payload.get("service", ""),
            query=payload.get("query", ""),
            passages=list(payload.get("passages", []) or []),
        )
        if "prompt" in payload and payload["prompt"] is not None:
            req.prompt = payload["prompt"]

        extras = {
            k: v for k, v in payload.items()
            if k not in ("service", "query", "passages", "prompt")
        }
        if extras:
            req.extras.CopyFrom(self._pack_extras(extras))

        resp = await self._call(self._stub.Score, req)
        result = {
            "query": resp.query,
            "scores": [float(s) for s in resp.scores],
            "service": resp.service,
            "cached": resp.cached,
            "timestamp": resp.timestamp,
        }
        if resp.HasField("meta"):
            result["meta"] = {"n_passages": resp.meta.n_passages}
        return result

    async def content(self, payload: dict) -> dict:
        from ..proto._generated import routir_pb2 as pb

        req = pb.ContentRequest(
            collection=payload.get("collection", ""),
            id=payload.get("id", ""),
        )
        resp = await self._call(self._stub.Content, req)
        return {
            "collection": resp.collection,
            "id": resp.id,
            "text": resp.text,
            "cached": resp.cached,
            "timestamp": resp.timestamp,
        }

    async def pipeline(self, payload: dict) -> dict:
        from ..proto._generated import routir_pb2 as pb

        req = pb.PipelineRequest(
            pipeline=payload.get("pipeline", ""),
            query=payload.get("query", ""),
        )
        if "collection" in payload and payload["collection"] is not None:
            req.collection = payload["collection"]

        for alias, kwargs_dict in (payload.get("runtime_kwargs") or {}).items():
            req.runtime_kwargs[alias].CopyFrom(self._pack_extras(kwargs_dict or {}))

        resp = await self._call(self._stub.Pipeline, req)
        result = {
            "query": resp.query,
            "scores": dict(resp.scores),
            "cached": resp.cached,
            "timestamp": resp.timestamp,
        }
        # REST omits ``expanded_queries`` when it's empty; mirror that so
        # the two transports return identical dicts.
        expanded = list(resp.expanded_queries)
        if expanded:
            result["expanded_queries"] = expanded
        return result
