import argparse
import asyncio
import hmac
import os
import signal
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, jsonify, request

from .config.load import load_config
from .pipeline import PipelineAliasRegistry, SearchPipeline
from .processors import ProcessorRegistry, ServiceNotFound
from .utils import logger


BANNER = r"""
█████▄   ▄▄▄  ▄▄ ▄▄ ▄▄▄▄▄▄ ██ █████▄
██▄▄██▄ ██▀██ ██ ██   ██   ██ ██▄▄██▄
██   ██ ▀███▀ ▀███▀   ██   ██ ██   ██
"""


def _get_version_info() -> str:
    """Return ``vX.Y.Z (abcdef1)`` when run from a git checkout, else ``vX.Y.Z``."""
    try:
        pkg_version = version("routir")
    except PackageNotFoundError:
        pkg_version = "unknown"
    git_hash = None
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or None
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return f"v{pkg_version} ({git_hash})" if git_hash else f"v{pkg_version}"


def _print_banner():
    blue, reset = "\033[94m", "\033[0m"
    for line in BANNER.strip("\n").splitlines():
        logger.info(f"{blue}{line}{reset}")
    logger.info(f"{blue}  {_get_version_info()}{reset}")


app = Quart(__name__)

if os.environ.get("CORS_ALLOWED", "False") == "True":
    from quart_cors import cors

    logger.warning("CORS_ALLOWED=True")
    app = cors(app, allow_origin="*")

config = None
api_key: str = None  # universal Bearer token; None disables auth
grpc_port_advertised: int = None  # set by main() when --grpc is on; surfaced via /avail

# Paths that bypass Bearer auth even when an API key is configured.
# `/ping` stays open so liveness probes don't need to carry credentials.
_AUTH_EXEMPT_PATHS = {"/ping"}


@app.before_serving
async def startup():
    """Initialize resources before the server starts."""
    global config
    await load_config(config)


@app.before_request
async def _check_bearer_auth():
    """Reject requests missing or carrying a wrong Bearer token, when configured.

    CORS preflight ``OPTIONS`` requests are exempt: browsers cannot attach
    an ``Authorization`` header on preflights, and the CORS layer needs to
    respond with the appropriate headers so the browser will send the real
    request.
    """
    if api_key is None or request.method == "OPTIONS" or request.path in _AUTH_EXEMPT_PATHS:
        return None
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "missing or invalid Authorization header"}), 401
    if not hmac.compare_digest(auth_header[len("Bearer ") :].strip(), api_key):
        return jsonify({"error": "invalid API key"}), 401
    return None


# TODO: standardize the API format with pydantic


@app.route("/search", methods=["POST"])
@app.route("/query", methods=["POST"])  # deprecated
async def process_query():
    """Retrieve ranked documents for a query.

    **Request** (JSON):

    .. code-block:: json

        {
            "service": "my-retriever",
            "query":   "what is machine learning?",
            "limit":   20
        }

    ``service`` (required) selects the registered search service.  All other
    fields are forwarded to the processor as-is; common extra fields include
    ``limit`` and ``subset``.

    **Response** (200 OK):

    .. code-block:: json

        {
            "scores":    {"doc1": 12.3, "doc2": 9.8},
            "cached":    false,
            "timestamp": 1700000000.0
        }

    Returns 400 for missing data or unknown service; 500 for engine errors.
    """

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        service = data.pop("service")
        try:
            result = await ProcessorRegistry.submit(service, "search", data)
        except ServiceNotFound as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /search")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/score", methods=["POST"])
async def process_scoring():
    """Score query-passage pairs (reranking).

    **Request** (JSON):

    .. code-block:: json

        {
            "service":  "my-reranker",
            "query":    "what is machine learning?",
            "passages": ["ML is a subset of AI", "Pizza is popular in Italy"]
        }

    ``passages`` is a flat list of text strings to score against the query.

    **Response** (200 OK):

    .. code-block:: json

        {
            "scores":    [0.95, 0.02],
            "cached":    false,
            "timestamp": 1700000000.0
        }

    ``scores`` is a list of floats in the same order as ``passages``.

    Returns 400 for missing data or unknown service; 500 for engine errors.
    """

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        service = data.pop("service")
        try:
            result = await ProcessorRegistry.submit(service, "score", data)
        except ServiceNotFound as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /score")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/content", methods=["POST"])
async def process_get_content():
    """Retrieve document text by ID from a registered collection.

    **Request** (JSON):

    .. code-block:: json

        {
            "collection": "my-corpus",
            "id":         "doc_42"
        }

    **Response** (200 OK):

    .. code-block:: json

        {
            "collection": "my-corpus",
            "id":         "doc_42",
            "text":       "Full document text here…"
        }

    Returns 400 for missing ``id``, unknown collection, or lookup failure;
    500 for unexpected errors.
    """
    try:
        data = await request.get_json()
        if not data or "id" not in data:
            return jsonify({"error": "No id provided"}), 400

        try:
            result = await ProcessorRegistry.submit(data["collection"], "content", data)
        except ServiceNotFound as e:
            return jsonify({"error": str(e)}), 400
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Error in /content")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/pipeline", methods=["POST"])
async def process_pipeline():
    """Execute a multi-stage pipeline defined by the pipeline DSL.

    **Request** (JSON):

    .. code-block:: json

        {
            "pipeline":       "bm25%100 >> my-reranker%20",
            "collection":     "my-corpus",
            "query":          "what is machine learning?",
            "runtime_kwargs": {"bm25": {"subset": "en"}}
        }

    Required fields: ``pipeline``, ``query``.  ``collection`` is required only
    when the pipeline contains a reranking stage (which needs document text).

    ``pipeline`` is a DSL string; see :mod:`routir.pipeline.parser` for
    syntax.  ``collection``, when supplied, must be a registered content
    service.  ``runtime_kwargs`` is optional and maps pipeline aliases to
    extra per-stage parameters.

    **Response** (200 OK):

    .. code-block:: json

        {
            "query":     "what is machine learning?",
            "scores":    {"doc1": 0.95, "doc2": 0.82},
            "cached":    false,
            "timestamp": 1700000000.0
        }

    ``expanded_queries`` is included when the pipeline contains a query
    decomposition stage.

    Returns 400 for missing required fields or DSL/service errors; 500 for
    unexpected errors.
    """
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        # required fields (collection is conditionally required; checked in verify)
        for field in ["pipeline", "query"]:
            if field not in data:
                return jsonify({"error": f"No {field} provided"}), 400

        try:
            result = await SearchPipeline.cached_run(
                data["pipeline"],
                data["query"],
                collection=data.get("collection"),
                runtime_kwargs=data.get("runtime_kwargs", {}),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Error in /pipeline")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/ping", methods=["GET"])
async def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "pong"})


@app.route("/avail", methods=["GET"])
async def get_avail_service():
    """List all services registered with the server, grouped by type.

    **Response** (200 OK):

    .. code-block:: json

        {
            "search":           ["bm25", "dense"],
            "score":            ["cross-encoder"],
            "fuse":             ["rrf"],
            "decompose_query":  [],
            "content":          ["my-corpus"],
            "pipeline_aliases": {"ragtime2": "{zho%100, rus%100, ...}ScoreFusion"}
        }

    Used by :func:`~routir.config.load.auto_add_relay_services` to discover
    services on remote servers.
    """
    payload = {
        **ProcessorRegistry.get_all_services(),
        "pipeline_aliases": PipelineAliasRegistry.source,
    }
    if grpc_port_advertised is not None:
        payload["grpc_port"] = grpc_port_advertised
    return jsonify(payload)


async def _build_grpc_server(args, api_key):
    """Build a gRPC asyncio server with RoutIR interceptors and servicer.

    Lazy-imports :mod:`grpc` and the servicer module so the base install
    (no ``[grpc]`` extra) keeps working when ``--grpc`` is not set.
    """
    import grpc

    from .grpc_interceptors import _BearerAuthInterceptor, _RequestIdInterceptor
    from .proto._generated import routir_pb2_grpc
    from .servicer import RoutirServicer

    max_bytes = args.grpc_max_message_mb * 1024 * 1024
    options = [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
    interceptors = [_RequestIdInterceptor()]
    if api_key is not None:
        interceptors.append(_BearerAuthInterceptor(api_key))

    server = grpc.aio.server(options=options, interceptors=interceptors)
    routir_pb2_grpc.add_RoutirServicer_to_server(RoutirServicer(grpc_port=args.grpc_port), server)
    server.add_insecure_port(f"{args.host}:{args.grpc_port}")
    return server


def _install_signal_handlers(grpc_server, shutdown_event):
    """Install loop-level SIGINT/SIGTERM handlers that drain both transports.

    When gRPC is enabled we take over signal handling so Hypercorn does not
    install its own (and so we can also stop the gRPC server).  The shared
    ``shutdown_event`` is passed to :func:`hypercorn.asyncio.serve` as its
    ``shutdown_trigger`` so REST drains cleanly when the event fires.
    In-flight gRPC RPCs get a 5-second grace window.

    ``loop.add_signal_handler`` raises ``NotImplementedError`` on Windows,
    so the call is wrapped defensively.
    """
    loop = asyncio.get_running_loop()

    def _shutdown():
        if shutdown_event.is_set():
            return
        logger.info("Signal received; stopping gRPC server (grace=5s) and draining REST")
        shutdown_event.set()
        # Fire-and-forget; the wait_for_termination task will then complete.
        asyncio.create_task(grpc_server.stop(grace=5.0))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            # add_signal_handler is unavailable on Windows; best-effort only.
            pass


async def _serve_all(app, hypercorn_config, args, api_key):
    """Run REST (always) and optionally gRPC on the same event loop."""
    if not args.grpc:
        # REST-only path: behavior is unchanged from before --grpc existed.
        await serve(app, hypercorn_config)
        return

    # gRPC enabled: we control signal handling, and pass a shutdown_trigger
    # to Hypercorn so it does not install its own handlers.
    shutdown_event = asyncio.Event()
    rest_task = asyncio.create_task(serve(app, hypercorn_config, shutdown_trigger=shutdown_event.wait))

    grpc_server = await _build_grpc_server(args, api_key)
    await grpc_server.start()
    logger.info(f"gRPC listening on {args.host}:{args.grpc_port}")
    grpc_task = asyncio.create_task(grpc_server.wait_for_termination())
    _install_signal_handlers(grpc_server, shutdown_event)
    await asyncio.gather(rest_task, grpc_task)


def main():
    """
    CLI entry point: parse arguments and start the Hypercorn ASGI server.

    Usage::

        routir config.json [--port 5000] [--host 0.0.0.0]

    The startup timeout is 600 s to accommodate slow model loading.

    Args (CLI):
        config: Path to the JSON config file (required positional argument).
        --port: TCP port to listen on (default 5000).
        --host: Interface to bind (default ``0.0.0.0`` for all interfaces).
        --cache_dir: Directory for local cache files (default ``./.cache``).
        --api_key: Bearer token required on every request (except ``/ping``).
            When unset, the server accepts unauthenticated requests.  Falls
            back to the ``ROUTIR_API_KEY`` env var, which is preferred since
            CLI arguments are visible in process listings.
        --grpc: When set, additionally start a gRPC server on the same event
            loop.  Off by default; the REST-only path is unchanged.
        --grpc-port: TCP port for the gRPC server (default 50051).
        --grpc-max-message-mb: Cap on inbound and outbound gRPC message size
            in megabytes (default 64).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--cache_dir", type=str, default="./.cache")
    parser.add_argument("--api_key", type=str, default=os.environ.get("ROUTIR_API_KEY"))
    parser.add_argument("--grpc", action="store_true", default=False, help="Also start a gRPC server alongside REST.")
    parser.add_argument("--grpc-port", type=int, default=50051, help="TCP port for the gRPC server (default 50051).")
    parser.add_argument(
        "--grpc-max-message-mb", type=int, default=64, help="Cap on gRPC send/receive message size in MB (default 64)."
    )

    args = parser.parse_args()

    _print_banner()

    global config, api_key, grpc_port_advertised
    config = args.config
    api_key = args.api_key
    if args.grpc:
        grpc_port_advertised = args.grpc_port
    if api_key:
        logger.info("Bearer-token authentication enabled")
    else:
        logger.warning("No --api_key set; server will accept unauthenticated requests")

    # app.run(host=args.host, port=args.port, use_reloader=False)
    hypercorn_config = Config()
    hypercorn_config.bind = [f"{args.host}:{args.port}"]
    hypercorn_config.startup_timeout = 600
    # hypercorn_config.keep_alive_timeout = 600
    logger.info(f"REST listening on {args.host}:{args.port}")
    asyncio.run(_serve_all(app, hypercorn_config, args, api_key))


if __name__ == "__main__":
    main()
