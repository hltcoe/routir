import asyncio
from pathlib import Path
from typing import List

import aiohttp

from ..models import Engine, Relay
from ..processors import (
    AsyncPairwiseScoreProcessor,
    AsyncQueryProcessor,
    BatchPairwiseScoreProcessor,
    ContentProcessor,
    Processor,
    ProcessorRegistry,
)
from ..utils import logger, session_request
from ..utils.extensions import load_all_extensions
from .config import Config


async def auto_add_relay_services(servers: List[str]):
    """
    Discover and register services from remote RoutIR servers as local proxies.

    Queries each server's ``/avail`` endpoint to list its available services,
    then creates :class:`~routir.models.Relay`-backed processors for every
    service not already registered locally.  This lets the local server
    transparently forward requests to the remote server.

    Only ``"search"`` and ``"score"`` service types are proxied.  Services
    already registered locally take precedence (remote services with the same
    name are skipped).

    Args:
        servers (list[str]): Base URLs of remote RoutIR servers to import from,
            e.g. ``["http://gpu-host-1:5000", "http://gpu-host-2:5000"]``.
            A single string is also accepted.
    """
    if isinstance(servers, str):
        servers = [servers]

    async with aiohttp.ClientSession() as session:
        resps = await asyncio.gather(
            *[session_request(session, url=f"{server}/avail", method="GET") for server in servers]
        )

    # ensure backward compatible
    avail_services = {
        server: {
            "search": resp['search'] if 'search' in resp else resp['query'],
            "score": resp['score']
        }
        for server, resp in zip(servers, resps)
        if resp is not None
    }

    for server in avail_services:
        for service_type, processor_cls in zip(["search", "score"], [AsyncQueryProcessor, AsyncPairwiseScoreProcessor]):
            for service_name in avail_services[server][service_type]:
                if ProcessorRegistry.has_service(service_name, service_type):
                    continue
                logger.info(f"Adding auto Relay to {server} for service `{service_name}` of type {service_type}")
                processor = processor_cls(
                    engine=Relay(name=service_name, config={"endpoint": server, "service": service_name})
                )
                await processor.start()
                ProcessorRegistry.register(service_name, service_type, processor)

def load_index_from_hfds(repo_id: str):
    """
    Download an index from HuggingFace Datasets.

    Args:
        repo_id: Repository ID (with optional 'hfds:' prefix)

    Returns:
        Path to the downloaded index directory
    """
    from huggingface_hub import snapshot_download
    if repo_id.startswith('hfds:'):
        repo_id = repo_id.replace('hfds:', '')
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

    for collection_config in config.collections:
        ProcessorRegistry.register(
            collection_config.name,
            "content",
            Processor.load(
                collection_config.processor,
                collection_config=collection_config
            )
        )
    logger.info("All collections are loaded")

    for service_config in config.services:
        def _cache_key(x):
            return tuple(x.get(k, "") for k in service_config.cache_key_fields)

        # load index from huggingface datasets
        if 'index_path' in service_config.config and service_config.config['index_path'].startswith('hfds:'):
            service_config.config['index_path'] = load_index_from_hfds(service_config.config['index_path'])

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

        logger.info(f"{service_config.name} initialized and ready")

    await auto_add_relay_services(config.server_imports)

    logger.info("All services are initialized")
