"""Pydantic models for service configuration."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """
    Configuration for a single search or ranking service.

    One ``ServiceConfig`` entry in the ``"services"`` list corresponds to one
    loaded engine, one search processor, and optionally one scoring processor
    registered in :data:`~routir.processors.registry.ProcessorRegistry`.

    Example JSON:

    .. code-block:: json

        {
            "name": "my-retriever",
            "engine": "Qwen3",
            "config": {
                "index_path": "/data/qwen3-index",
                "embedding_model_name": "Qwen/Qwen3-Embedding-8B"
            },
            "cache": 4096,
            "cache_ttl": 600,
            "batch_size": 16,
            "max_wait_time": 0.05
        }

    Attributes:
        name (str): Service identifier used in API requests (the ``"service"``
            field) and as the key in
            :data:`~routir.processors.registry.ProcessorRegistry`.
        engine (str): Class name of the :class:`~routir.models.abstract.Engine`
            subclass to instantiate.  Must be importable at startup — either a
            built-in engine or one loaded via ``file_imports``.
        config (dict): Engine-specific parameters passed as ``config=`` to the
            engine constructor.  Content varies by engine; see the engine's
            ``__init__`` for accepted keys.  The special key ``"index_path"``
            with the ``hfds:<repo>`` prefix triggers automatic download from
            Hugging Face Datasets.
        processor (str): Class name of the
            :class:`~routir.processors.abstract.BatchProcessor` subclass for the
            **search** role.  Default ``"BatchQueryProcessor"`` works for most
            engines.  Override only if you need custom batching or request logic.
        cache (int): In-memory LRU cache size for search results (number of
            entries).  ``-1`` (default) disables the cache entirely.  Ignored
            when ``cache_redis_url`` is set.
        cache_ttl (int): Cache entry time-to-live in seconds (default 600).
            Applies to both LRU and Redis caches.
        batch_size (int): Maximum requests accumulated into one batch before the
            engine processes them (default 32).
        max_wait_time (float): Maximum seconds to wait for a batch to fill before
            processing a partial batch (default 0.05 s).  Lower values reduce
            latency; higher values improve GPU utilisation.
        cache_key_fields (list[str]): Request fields included in the cache key.
            Default ``["query", "limit"]``.  Add extra fields (e.g.
            ``"subset"``) whenever they affect the results.
        cache_redis_url (str, optional): Redis connection URL for distributed
            caching (e.g. ``"redis://localhost:6379"``).  When set, Redis
            replaces the in-memory LRU cache.
        cache_redis_kwargs (dict, optional): Additional keyword arguments
            forwarded to the Redis client (e.g. ``{"password": "…", "db": 1}``).
        scoring_disabled (bool): When ``True``, the scoring/reranking processor
            for this service is not registered even if the engine implements
            ``score_batch``.  Useful when you want search-only access to an
            engine that also supports reranking.
    """

    name: str
    engine: str
    # collection: str # mostly for book keeping purpose and allow service name to be cleaner
    config: Dict[str, Any]
    processor: str = "BatchQueryProcessor"
    cache: int = -1
    batch_size: int = 32
    cache_ttl: int = 600
    max_wait_time: float = 0.05
    cache_key_fields: List[str] = Field(default_factory=lambda: ["query", "limit"])
    cache_redis_url: Optional[str] = None
    cache_redis_kwargs: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    scoring_disabled: bool = False


class CollectionConfig(BaseModel):
    """
    Configuration for a document collection.

    Collections expose the ``/content`` endpoint and are used by reranking
    pipeline stages to fetch document text by ID.

    .. note::

        The class name has a historical typo (three ``l``\\ s).  It is kept as-is
        to avoid breaking existing imports.

    Example JSON:

    .. code-block:: json

        {
            "name": "my-corpus",
            "doc_path": "/data/corpus.jsonl",
            "id_field": "docid",
            "content_field": "text"
        }

    Attributes:
        name (str): Collection identifier used in API requests (the
            ``"collection"`` field) and as the key in
            :data:`~routir.processors.registry.ProcessorRegistry`.
        doc_path (str, optional): Path to the JSONL (or gzip-compressed) document
            file.  Each line must be a JSON object.  May be a local path or an
            ``hfds:<repo>`` URL for Hugging Face Datasets.
        processor (str): Class name of the content-processor to use.
            Default ``"ContentProcessor"`` provides offset-based random access.
            Use ``"IRDSProcessor"`` to load from an ``ir_datasets`` dataset ID.
        offset_source (str): Strategy for random document access:

            * ``"offsetfile"`` *(default)* — builds a byte-offset map
              (``.offsetmap`` sidecar file) for fast O(1) lookup in a JSONL file.
            * ``"msmarco_seg"`` — reads from sharded gzipped files in the
              MSMARCO v2.1 segmented document format, using embedded byte offsets
              from the document ID.

        id_field (str): JSON key whose value is the document ID
            (default ``"id"``).
        content_field (str or list[str]): JSON key(s) whose values are
            concatenated (space-joined) to form the document text returned by
            ``/content``.  Accepts a single string or a list for multi-field
            concatenation (e.g. ``["title", "body"]``).  Always stored as a list
            internally after validation.
        id_to_lang_mapping (str, optional): Path to a pickle file mapping
            document IDs to language codes.  Used by processors that serve
            multilingual corpora.
        cache_path (str, optional): Directory for the ``.offsetmap`` cache file.
            Defaults to the same directory as ``doc_path``.
        force_load_all_documents (bool): When ``True``, all documents are loaded
            into memory at startup for maximum throughput.  Only suitable for
            small corpora; default ``False`` uses on-demand offset-based access.
    """

    name: str
    doc_path: Optional[str] = None
    processor: str = "ContentProcessor"
    offset_source: Literal["msmarco_seg", "offsetfile"] = "offsetfile"
    id_field: str = "id"
    content_field: Union[str, List[str]] = "text"
    id_to_lang_mapping: Optional[str] = None
    cache_path: Optional[str] = None
    force_load_all_documents: bool = False

    def model_post_init(self, __context):
        """Ensure content_field is always a list."""
        if not isinstance(self.content_field, list):
            self.content_field = [self.content_field]


class Config(BaseModel):
    """
    Top-level configuration for the RoutIR service.

    Passed as a JSON file (or JSON string) to ``routir <config.json>``.
    Parsed by :func:`~routir.config.load.load_config`, which initialises all
    collections and services and registers them with
    :data:`~routir.processors.registry.ProcessorRegistry`.

    Example JSON skeleton:

    .. code-block:: json

        {
            "file_imports": ["./my_engine.py"],
            "collections": [
                {
                    "name": "my-corpus",
                    "doc_path": "/data/corpus.jsonl"
                }
            ],
            "services": [
                {
                    "name": "my-retriever",
                    "engine": "MyEngine",
                    "config": {"index_path": "/data/index"}
                }
            ],
            "server_imports": ["http://other-host:5000"]
        }

    Attributes:
        services (list[ServiceConfig]): Search/ranking services to load and
            register.  Each entry loads one engine and creates its processors.
        collections (list[CollectionConfig]): Document collections to register
            as content services.  Required for reranking pipeline stages.
        server_imports (list[str]): URLs of remote RoutIR servers whose services
            are proxied locally via :class:`~routir.models.Relay`.  Discovered
            automatically from each server's ``/avail`` endpoint at startup.
        file_imports (list[str]): Paths to Python files loaded before any service
            is initialised.  Use this to register custom
            :class:`~routir.models.abstract.Engine` subclasses.
        dynamic_pipeline (bool): When ``True`` (default), the ``/pipeline``
            endpoint accepts arbitrary pipeline DSL strings at request time.
            Set to ``False`` to restrict the server to pre-defined services only.
    """

    services: List[ServiceConfig] = Field(default_factory=list)
    collections: List[CollectionConfig] = Field(default_factory=list)
    server_imports: List[str] = Field(default_factory=list)  # not yet implemented
    file_imports: List[str] = Field(default_factory=list)
    dynamic_pipeline: bool = True
