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
    }

    for server in servers:
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


async def load_config(config: str):
    if Path(config).exists():
        config = Path(config).read_text()

    config: Config = Config.model_validate_json(config)

    load_all_extensions(user_specified_files=config.file_imports)

    for collection_config in config.collections:
        ProcessorRegistry.register(collection_config.name, "content", ContentProcessor(collection_config))
    logger.info("All collections are loaded")

    for service_config in config.services:
        def _cache_key(x):
            return tuple(x.get(k, "") for k in service_config.cache_key_fields)

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
