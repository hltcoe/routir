import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..client import AsyncClient
from ..models import Engine, Relay
from ..pipeline import PipelineAliasRegistry, set_pipeline_cache
from ..processors import (
    AsyncPairwiseScoreProcessor,
    AsyncQueryProcessor,
    BatchDecomposeQueryProcessor,
    BatchPairwiseScoreProcessor,
    ContentProcessor,
    Processor,
    ProcessorRegistry,
    RelayContentProcessor,
)
from ..processors.cache import LRUCache, RedisCache
from ..utils import logger
from ..utils.extensions import load_all_extensions
from .config import Config


def _normalize_server(s: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if isinstance(s, str):
        return {"endpoint": s}
    if isinstance(s, dict) and "endpoint" in s:
        return s
    raise ValueError(f"Bad server entry: {s!r}; expected str or dict with 'endpoint'")


async def auto_add_relay_services(
    servers: Union[str, List[Union[str, Dict[str, str]]]],
    content_cache_settings: Optional[Dict[str, Any]] = None,
):
    """
    Discover and register services from remote RoutIR servers as local proxies.

    Queries each server's ``/avail`` endpoint (via :class:`AsyncClient`) to list
    its available services, then creates :class:`~routir.models.Relay`-backed
    processors for every service not already registered locally.  This lets the
    local server transparently forward requests to the remote server.

    ``"search"``, ``"score"``, and ``"content"`` service types are proxied.
    Services already registered locally take precedence (remote services with
    the same name are skipped).

    Args:
        servers: Either a single server entry or a list of entries.  Each entry
            may be a string (REST base URL, e.g. ``"http://host:5000"``) or a
            dict with at least ``"endpoint"`` and optionally ``"grpc_endpoint"``,
            ``"api_key"``, etc.  These extra fields are forwarded into the
            created :class:`~routir.models.Relay` config (and into
            :class:`RelayContentProcessor` for content) so the data plane can
            use gRPC even though discovery always uses REST.
        content_cache_settings: Optional dict with keys ``cache_size``,
            ``cache_ttl``, ``redis_url``, ``redis_kwargs`` applied to every
            :class:`RelayContentProcessor` registered from this call.  Sourced
            from the ``relay_content_cache*`` fields on :class:`Config`.
    """
    if isinstance(servers, (str, dict)):
        servers = [servers]
    servers = [_normalize_server(s) for s in servers]
    content_cache_settings = content_cache_settings or {}

    async def _fetch_avail(entry: Dict[str, str]):
        async with AsyncClient(endpoint=entry["endpoint"], transport="rest") as c:
            try:
                return await c.avail()
            except Exception as e:
                logger.exception(f"Failed to fetch /avail from {entry['endpoint']}: {e}")
                return None

    resps = await asyncio.gather(*[_fetch_avail(s) for s in servers])

    # ensure backward compatible
    avail_services = {
        i: {
            "search": resp["search"] if "search" in resp else resp["query"],
            "score": resp["score"],
            "content": resp.get("content", []),
        }
        for i, resp in enumerate(resps)
        if resp is not None
    }

    for i, types in avail_services.items():
        entry = servers[i]
        for service_type, processor_cls in zip(["search", "score"], [AsyncQueryProcessor, AsyncPairwiseScoreProcessor]):
            for service_name in types[service_type]:
                if ProcessorRegistry.has_service(service_name, service_type):
                    continue
                logger.info(f"Adding auto Relay to {entry['endpoint']} for service `{service_name}` of type {service_type}")
                relay_config = {"service": service_name, **entry}
                processor = processor_cls(engine=Relay(name=service_name, config=relay_config))
                await processor.start()
                ProcessorRegistry.register(service_name, service_type, processor)

        # Fields on the per-server entry that the AsyncClient understands.
        client_kwargs = {
            k: entry[k] for k in ("grpc_endpoint", "api_key", "transport", "timeout", "retries", "tls") if k in entry
        }
        for collection_name in types["content"]:
            if ProcessorRegistry.has_service(collection_name, "content"):
                continue
            logger.info(f"Adding auto Relay to {entry['endpoint']} for collection `{collection_name}` of type content")
            processor = RelayContentProcessor(
                collection=collection_name,
                endpoint=entry["endpoint"],
                **client_kwargs,
                **content_cache_settings,
            )
            await processor.start()
            ProcessorRegistry.register(collection_name, "content", processor)


def load_index_from_hfds(repo_id: str):
    """
    Download an index from HuggingFace Datasets.

    Args:
        repo_id: Repository ID (with optional 'hfds:' prefix)

    Returns:
        Path to the downloaded index directory
    """
    from huggingface_hub import snapshot_download

    if repo_id.startswith("hfds:"):
        repo_id = repo_id.replace("hfds:", "")
    logger.info(f"Downloading {repo_id} from Huggingface Datasets")
    # TODO: could first load config from the repo and do some checking
    local_path = snapshot_download(repo_id=repo_id, repo_type="dataset") + "/index"
    logger.info(f"Replacing {repo_id} with {local_path}")
    return local_path


async def load_config(config: str):
    """
    Parse the service configuration and register all collections and services.

    This is the main initialization entry point called by the server at startup.
    It performs the following steps in order:

    1. Parse the JSON config (file path or raw string) into a
       :class:`~routir.config.config.Config` model.
    2. Load any Python files listed in ``file_imports`` (custom engine classes).
    3. For each collection, create and register a content processor.
    4. For each service, instantiate the engine and register search (and
       optionally score) processors.  Index paths with the ``hfds:`` prefix are
       downloaded from Hugging Face Datasets first.
    5. Discover and proxy services from remote servers listed in
       ``server_imports``.

    Args:
        config (str): Either a file path to a JSON config file or a raw JSON
            string.  File paths are read and parsed automatically.

    Note:
        This function modifies the global
        :data:`~routir.processors.registry.ProcessorRegistry` singleton in place.
        It is not safe to call concurrently.
    """
    if Path(config).exists():
        config = Path(config).read_text()

    config: Config = Config.model_validate_json(config)

    load_all_extensions(user_specified_files=config.file_imports)

    # Build the pipeline-level result cache, if enabled.  Mirrors the
    # ``cache_size > 0`` gate used by individual processors so ``-1``/``0``
    # disables it.
    if config.pipeline_cache > 0:
        if config.pipeline_cache_redis_url is not None:
            pipeline_cache = RedisCache(
                config.pipeline_cache,
                config.pipeline_cache_ttl,
                config.pipeline_cache_redis_url,
                key_prefix="routirpipeline:",
                **config.pipeline_cache_redis_kwargs,
            )
        else:
            pipeline_cache = LRUCache(config.pipeline_cache, config.pipeline_cache_ttl)
        set_pipeline_cache(pipeline_cache)
        logger.info(
            f"Pipeline-level cache enabled (capacity={config.pipeline_cache}, "
            f"ttl={config.pipeline_cache_ttl}s, "
            f"backend={'redis' if config.pipeline_cache_redis_url else 'lru'})"
        )

    for collection_config in config.collections:
        ProcessorRegistry.register(
            collection_config.name, "content", Processor.load(collection_config.processor, collection_config=collection_config)
        )
    logger.info("All collections are loaded")

    for service_config in config.services:

        def _cache_key(x):
            return tuple(x.get(k, "") for k in service_config.cache_key_fields)

        # load index from huggingface datasets
        if "index_path" in service_config.config and service_config.config["index_path"].startswith("hfds:"):
            service_config.config["index_path"] = load_index_from_hfds(service_config.config["index_path"])

        engine: Engine = Engine.load(service_config.engine, name=service_config.name, config=service_config.config)

        if engine.can_search:
            processor: Processor = Processor.load(
                service_config.processor,
                engine=engine,
                batch_size=service_config.batch_size,
                max_wait_time=service_config.max_wait_time,
                cache_size=service_config.cache,
                cache_ttl=service_config.cache_ttl,
                cache_key=_cache_key,
                redis_url=service_config.cache_redis_url,
                redis_kwargs=service_config.cache_redis_kwargs,
            )
            await processor.start()
            ProcessorRegistry.register(service_config.name, "search", processor)

        if engine.can_score and not service_config.scoring_disabled:
            processor = BatchPairwiseScoreProcessor(
                engine,
                batch_size=service_config.batch_size,
                max_wait_time=service_config.max_wait_time,
                cache_size=-1,  # turn off cache for now
            )
            await processor.start()
            ProcessorRegistry.register(service_config.name, "score", processor)

        if engine.can_decompose_query:
            processor = BatchDecomposeQueryProcessor(
                engine,
                batch_size=service_config.batch_size,
                max_wait_time=service_config.max_wait_time,
                cache_size=service_config.cache,
                cache_ttl=service_config.cache_ttl,
                cache_key=_cache_key,
            )
            await processor.start()
            ProcessorRegistry.register(service_config.name, "decompose_query", processor)

        logger.info(f"{service_config.name} initialized and ready")

    content_cache_settings = {
        "cache_size": config.relay_content_cache,
        "cache_ttl": config.relay_content_cache_ttl,
        "redis_url": config.relay_content_cache_redis_url,
        "redis_kwargs": config.relay_content_cache_redis_kwargs,
    }
    await auto_add_relay_services(config.server_imports, content_cache_settings)

    # Collections (role "content") never appear in the pipeline DSL, so they
    # cannot collide with aliases.  Only check against services callable from
    # within a pipeline string.
    callable_roles = {"search", "score", "fuse", "decompose_query"}
    reserved_names = {name for name, by_role in ProcessorRegistry.all_services.items() if callable_roles & by_role.keys()}
    PipelineAliasRegistry.register_all(config.pipeline_aliases, reserved_names)

    logger.info("All services are initialized")
