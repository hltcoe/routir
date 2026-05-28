"""gRPC server interceptors for RoutIR.

Imported only when the gRPC transport is enabled (``--grpc``).  Importing this
module pulls in :mod:`grpc`, so it must not be referenced from the REST-only
code path.
"""

import hmac
import uuid

import grpc

from .utils import logger


# RPC method paths look like ``/routir.v1.Routir/<MethodName>``.  Methods listed
# here bypass Bearer-token authentication, mirroring REST's
# ``_AUTH_EXEMPT_PATHS = {"/ping"}`` exemption in :mod:`routir.serve`.
_AUTH_EXEMPT_METHODS = {"Ping"}


def _abort_handler(code: grpc.StatusCode, message: str):
    """Build a ``unary_unary`` handler that aborts with the given status."""

    async def _abort(request, context):  # noqa: ARG001 — request unused intentionally
        await context.abort(code, message)

    return grpc.unary_unary_rpc_method_handler(_abort)


class _BearerAuthInterceptor(grpc.aio.ServerInterceptor):
    """Reject RPCs that do not present the configured Bearer token.

    Mirrors the REST-side ``_check_bearer_auth`` behavior in
    :mod:`routir.serve`: ``Ping`` is exempt so liveness probes do not need
    to carry credentials.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key

    async def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method.rsplit("/", 1)[-1]
        if method_name in _AUTH_EXEMPT_METHODS:
            return await continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or ())
        auth_header = metadata.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return _abort_handler(
                grpc.StatusCode.UNAUTHENTICATED,
                "missing or invalid Authorization header",
            )
        token = auth_header[len("Bearer "):].strip()
        if not hmac.compare_digest(token, self._api_key):
            return _abort_handler(grpc.StatusCode.UNAUTHENTICATED, "invalid API key")

        return await continuation(handler_call_details)


class _RequestIdInterceptor(grpc.aio.ServerInterceptor):
    """Log each RPC with a request ID, generating one if absent.

    The interceptor does not modify the request, response, or metadata —
    it only logs at INFO level.
    """

    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or ())
        request_id = metadata.get("x-request-id") or str(uuid.uuid4())
        logger.info(f"[req={request_id}] gRPC {handler_call_details.method}")
        return await continuation(handler_call_details)
