import argparse
import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, jsonify, request

from .config.load import load_config
from .pipeline import SearchPipeline
from .processors import ProcessorRegistry
from .utils import logger


app = Quart(__name__)
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
    """API endpoint for processing requests."""

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        service = data.pop("service")
        if not ProcessorRegistry.has_service(service, "search"):
            return jsonify({"error": "Unsupported service"}), 400

        result = await ProcessorRegistry.get(service, "search").submit(data)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/score", methods=["POST"])
async def process_scoring():
    """API endpoint for processing requests."""

    try:
        data = await request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        service = data.pop("service")
        if not ProcessorRegistry.has_service(service, "score"):
            return jsonify({"error": "Unsupported service"}), 400

        result = await ProcessorRegistry.get(service, "score").submit(data)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/content", methods=["POST"])
async def process_get_content():
    """API endpoint for retrieving document content by ID."""
    try:
        data = await request.get_json()
        if not data or "id" not in data:
            return jsonify({"error": "No id provided"}), 400

        if not ProcessorRegistry.has_service(data["collection"], "content"):
            return jsonify({"error": "Unsupported collection"}), 400

        result = await ProcessorRegistry.get(data["collection"], "content").submit(data)
        return jsonify({**data, **result}), 400 if "error" in result else 200

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/pipeline", methods=["POST"])
async def process_pipeline():
    """API endpoint for executing custom search pipelines."""
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
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ping", methods=["GET"])
async def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "pong"})


@app.route("/avail", methods=["GET"])
async def get_avail_service():
    """API endpoint for listing available services."""
    return jsonify(ProcessorRegistry.get_all_services())


def main():
    """
    Main entry point for the search service server.

    Parses command line arguments and starts the Hypercorn server.
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
    asyncio.run(serve(app, hypercorn_config))


if __name__ == "__main__":
    main()
