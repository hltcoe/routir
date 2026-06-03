import asyncio
import gzip
import os
import re
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..client import AsyncClient
from ..models import Engine, Relay
from ..pipeline import PipelineAliasRegistry, set_pipeline_cache
from ..pipeline.cache import set_bytes_content_cache_max_bytes
from ..collections.processor import ContentProcessor
from ..collections.relay import RelayContentProcessor
from ..processors import (
    AsyncPairwiseScoreProcessor,
    AsyncQueryProcessor,
    BatchDecomposeQueryProcessor,
    BatchPairwiseScoreProcessor,
    Processor,
    ProcessorRegistry,
)
from ..processors.cache import LRUCache, RedisCache
from ..utils import logger
from ..utils.extensions import load_all_extensions
from .config import Config, TarSource


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

    # ``collection`` is a dict-of-collection-info (views map + default);
    # ``score_view_kinds`` lets us register relay score slots with the right
    # modality.  Fall back to empty dicts so older shapes degrade gracefully
    # within a single deployment generation.
    avail_services = {
        i: {
            "search": resp["search"] if "search" in resp else resp.get("query", []),
            "score": resp["score"],
            "score_view_kinds": resp.get("score_view_kinds", {}),
            "collection": resp.get("collection", {}),
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
                # ``score_view_kinds`` is only meaningful for score slots; for
                # search relays we default to text since the search interface
                # itself doesn't carry passages.
                view_kind = types["score_view_kinds"].get(service_name, "text") if service_type == "score" else "text"
                ProcessorRegistry.register(service_name, service_type, processor, view_kind=view_kind)

        # Fields on the per-server entry that the AsyncClient understands.
        client_kwargs = {
            k: entry[k] for k in ("grpc_endpoint", "api_key", "transport", "timeout", "retries", "tls") if k in entry
        }
        # ``types["collection"]`` is a dict { name: {"views": {...}, "default": ...} }.
        # Older deployments may still hand back a list of names; coerce both to
        # the new shape so the rest of the function stays uniform.
        collection_info = types["collection"]
        if isinstance(collection_info, list):
            collection_info = {name: {"views": {}, "default": None} for name in collection_info}
        for collection_name, info in collection_info.items():
            if ProcessorRegistry.has_service(collection_name, "content"):
                continue
            logger.info(f"Adding auto Relay to {entry['endpoint']} for collection `{collection_name}` of type content")
            processor = RelayContentProcessor(
                collection=collection_name,
                endpoint=entry["endpoint"],
                **client_kwargs,
                **content_cache_settings,
            )
            # Hydrate so ``SearchPipeline.verify()`` can resolve views remotely.
            processor.views = dict(info.get("views") or {})
            processor.default_view = info.get("default")
            await processor.start()
            ProcessorRegistry.register(
                collection_name,
                "content",
                processor,
                views=processor.views,
                default_view=processor.default_view,
            )


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


def _has_tar_gz_view(parsed_config) -> bool:
    """Cheap up-front check: does any collection's view declare a .tar.gz tar_template?"""
    for coll in parsed_config.collections:
        if not getattr(coll, "views", None):
            continue
        for view_spec in coll.views.values():
            if not isinstance(view_spec.source, TarSource):
                continue
            tmpl = view_spec.source.tar_template
            if tmpl.endswith(".gz") or tmpl.endswith(".tgz"):
                return True
    return False


def _decompress_tar_gz_views(tar_source, cache_dir: Path) -> str:
    """Decompress every concrete ``.tar.gz`` matching ``tar_source.tar_template``
    into ``cache_dir`` (idempotent) and return the rewritten ``.tar``
    template.

    Policy: ``<some_dir>/<basename>.tar.gz`` -> ``<cache_dir>/<basename>.tar``.
    The basename is preserved so ``{shard}`` interpolation continues to work
    naturally — only the directory and suffix change.
    """
    pattern = tar_source.tar_template
    glob_pattern = re.sub(r"\{shard(?::[^}]*)?\}", "*", pattern)
    found = sorted(glob(glob_pattern))

    for src in found:
        src_p = Path(src)
        name = src_p.name
        if name.endswith(".tar.gz"):
            dst_name = name[: -len(".gz")]
        elif name.endswith(".tgz"):
            dst_name = name[: -len(".tgz")] + ".tar"
        else:
            continue
        dst = cache_dir / dst_name
        if not dst.exists():
            logger.info(f"Decompressing {src} -> {dst} (one-time)")
            tmp = cache_dir / (dst_name + ".tmp")
            with gzip.open(src, "rb") as fr, tmp.open("wb") as fw:
                shutil.copyfileobj(fr, fw, length=1 << 20)
            os.replace(tmp, dst)

    # Rewrite the template's directory + suffix.
    p = Path(pattern)
    new_name = p.name
    if new_name.endswith(".tar.gz"):
        new_name = new_name[: -len(".gz")]
    elif new_name.endswith(".tgz"):
        new_name = new_name[: -len(".tgz")] + ".tar"
    return str(cache_dir / new_name)


def _resolve_tar_gz_in_views(parsed_config, decompress_cache_dir: Optional[str]) -> None:
    """Validate / resolve any .tar.gz references in TarSource view specs.

    Without ``decompress_cache_dir``, ``.tar.gz`` references are rejected at
    startup with a clear redirect (PR6 doesn't yet wire ``indexed_gzip`` into
    the runtime read path).  With the flag, each ``.tar.gz`` is decompressed
    once into ``decompress_cache_dir`` and the view's ``tar_template`` is
    rewritten to point at the ``.tar`` under the cache dir.  The basename is
    preserved so ``{shard}`` interpolation continues to work naturally.
    """
    found_gz: List[str] = []
    for coll in parsed_config.collections:
        if not getattr(coll, "views", None):
            continue
        for view_name, view_spec in coll.views.items():
            if not isinstance(view_spec.source, TarSource):
                continue
            tmpl = view_spec.source.tar_template
            # tar_template may contain {shard} — the suffix on the template
            # itself is enough to detect the policy collection-wide.
            if tmpl.endswith(".gz") or tmpl.endswith(".tgz"):
                found_gz.append(f"{coll.name}.{view_name}: {tmpl}")

    if not found_gz:
        return

    if decompress_cache_dir is None:
        raise RuntimeError(
            ".tar.gz tar_template references found in config but "
            "--allow-tar-gz-decompress-cache is not set:\n  "
            + "\n  ".join(found_gz)
            + "\n\nPass --allow-tar-gz-decompress-cache <dir> to decompress "
              "these shards once into <dir> at startup.  Native gzip random "
              "access (via indexed_gzip) is not yet wired into the runtime."
        )

    cache_dir = Path(decompress_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for coll in parsed_config.collections:
        if not getattr(coll, "views", None):
            continue
        for view_spec in coll.views.values():
            if not isinstance(view_spec.source, TarSource):
                continue
            if not (view_spec.source.tar_template.endswith(".gz") or
                    view_spec.source.tar_template.endswith(".tgz")):
                continue
            new_template = _decompress_tar_gz_views(view_spec.source, cache_dir)
            view_spec.source.tar_template = new_template


async def load_config(config: str, *, tar_gz_decompress_cache: Optional[str] = None):
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
        tar_gz_decompress_cache (str, optional): Directory into which any
            ``.tar.gz`` tar_template references should be decompressed once
            at startup.  When ``None`` (the default), ``.tar.gz`` references
            raise at startup — native gzip random access (via
            ``indexed_gzip``) is not yet wired into the runtime read path.

    Note:
        This function modifies the global
        :data:`~routir.processors.registry.ProcessorRegistry` singleton in place.
        It is not safe to call concurrently.
    """
    if Path(config).exists():
        config = Path(config).read_text()

    config: Config = Config.model_validate_json(config)

    if config.collections:
        _resolve_tar_gz_in_views(config, tar_gz_decompress_cache)

    load_all_extensions(user_specified_files=config.file_imports)

    # Build the pipeline-level result cache, if enabled.  Mirrors the
    # ``cache_size > 0`` gate used by individual processors so ``-1``/``0``
    # disables it.
    # Per-pipeline bytes-content cache cap (PR5a): wired here so REST/gRPC
    # handlers can pull the value from the same module-level holder as the
    # pipeline result cache.
    set_bytes_content_cache_max_bytes(config.bytes_content_cache_max_bytes)

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
        content_processor = Processor.load(
            collection_config.processor,
            collection_config=collection_config,
            cache_size=collection_config.cache,
            cache_ttl=collection_config.cache_ttl,
        )
        # Snapshot per-view kind at registration time.  Prefer the user-declared
        # ``ViewSpec.kind`` (what they wrote in the config) over the backend's
        # class attribute; this keeps the registry honest even when a synthetic
        # / stand-in backend doesn't yet expose the real modality
        # (e.g. PR5-pending bytes views still using a placeholder backend).
        declared_views = getattr(collection_config, "views", {}) or {}
        views_kind = {name: getattr(spec, "kind", "text") for name, spec in declared_views.items()}
        if not views_kind:
            # Processors without a ViewSpec list (e.g. IRDSProcessor) fall back
            # to the resolved backend's kind, or text if even that is absent.
            backends = getattr(content_processor, "backends", None) or {}
            views_kind = {name: backend.kind for name, backend in backends.items()}
        ProcessorRegistry.register(
            collection_config.name,
            "content",
            content_processor,
            views=views_kind,
            default_view=getattr(content_processor, "default_view", getattr(collection_config, "default_view", None)),
        )
    logger.info("All collections are loaded")

    for service_config in config.services:

        def _cache_key(x):
            return tuple(x.get(k, "") for k in service_config.cache_key_fields)

        # load index from huggingface datasets
        if "index_path" in service_config.config and service_config.config["index_path"].startswith("hfds:"):
            service_config.config["index_path"] = load_index_from_hfds(service_config.config["index_path"])

        engine: Engine = Engine.load(service_config.engine, name=service_config.name, config=service_config.config)

        engine_view_kind = getattr(engine, "accepts_view_kind", "text")

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
            ProcessorRegistry.register(service_config.name, "search", processor, view_kind=engine_view_kind)

        if engine.can_score and not service_config.scoring_disabled:
            processor = BatchPairwiseScoreProcessor(
                engine,
                batch_size=service_config.batch_size,
                max_wait_time=service_config.max_wait_time,
                cache_size=-1,  # turn off cache for now
            )
            await processor.start()
            ProcessorRegistry.register(service_config.name, "score", processor, view_kind=engine_view_kind)

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
            # decompose_query produces text sub-queries; modality flag is "text".
            ProcessorRegistry.register(service_config.name, "decompose_query", processor, view_kind="text")

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
