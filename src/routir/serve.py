import argparse
import asyncio
import os

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, jsonify, request

from .config.load import load_config
from .pipeline import PipelineAliasRegistry, SearchPipeline
from .processors import ProcessorRegistry
from .utils import logger


app = Quart(__name__)

if os.environ.get("CORS_ALLOWED", "False") == "True":
    from quart_cors import cors
    logger.warning("CORS_ALLOWED=True")
    app = cors(app, allow_origin="*")

config = None


@app.before_serving
async def startup():
    """Initialize resources before the server starts."""
    global config
    await load_config(config)


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
        if not ProcessorRegistry.has_service(service, "search"):
            return jsonify({"error": f"Service '{service}' not found or does not support search"}), 400

        result = await ProcessorRegistry.get(service, "search").submit(data)
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
        if not ProcessorRegistry.has_service(service, "score"):
            return jsonify({"error": f"Service '{service}' not found or does not support scoring"}), 400

        result = await ProcessorRegistry.get(service, "score").submit(data)
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

        if not ProcessorRegistry.has_service(data["collection"], "content"):
            return jsonify({"error": f"Collection '{data['collection']}' not found"}), 400

        result = await ProcessorRegistry.get(data["collection"], "content").submit(data)
        return jsonify({**data, **result}), 400 if "error" in result else 200

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

    Required fields: ``pipeline``, ``collection``, ``query``.

    ``pipeline`` is a DSL string; see :mod:`routir.pipeline.parser` for
    syntax.  ``collection`` must be a registered content service (needed for
    reranking stages).  ``runtime_kwargs`` is optional and maps pipeline
    aliases to extra per-stage parameters.

    **Response** (200 OK) — same fields as ``/search`` plus the echoed
    request fields:

    .. code-block:: json

        {
            "pipeline":   "bm25%100 >> my-reranker%20",
            "collection": "my-corpus",
            "query":      "what is machine learning?",
            "scores":     {"doc1": 0.95, "doc2": 0.82},
            "cached":     false,
            "timestamp":  1700000000.0
        }

    Returns 400 for missing required fields or DSL/service errors; 500 for
    unexpected errors.
    """
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        # required fields
        for field in ["pipeline", "collection", "query"]:
            if field not in data:
                return jsonify({"error": f"No {field} provided"}), 400

        pipeline = SearchPipeline.from_string(data["pipeline"], data["collection"], runtime_kwargs=data.get("runtime_kwargs", {}))
        result = await pipeline.run(data["query"])
        return jsonify({**data, **result}), 200

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
    return jsonify({
        **ProcessorRegistry.get_all_services(),
        "pipeline_aliases": PipelineAliasRegistry.source,
    })


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
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--cache_dir", type=str, default="./.cache")

    args = parser.parse_args()

    global config
    config = args.config

    # app.run(host=args.host, port=args.port, use_reloader=False)
    hypercorn_config = Config()
    hypercorn_config.bind = [f"{args.host}:{args.port}"]
    hypercorn_config.startup_timeout = 600
    # hypercorn_config.keep_alive_timeout = 600
    asyncio.run(serve(app, hypercorn_config))


if __name__ == "__main__":
    main()
