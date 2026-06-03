"""Pydantic models for service configuration."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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


class TextJsonlSource(BaseModel):
    """Text-from-JSONL view source: read content fields from a JSONL line.

    Concatenates one or more JSON fields with ``sep`` to form the text payload.

    Attributes:
        source: Discriminator literal ``"text_jsonl"``.
        doc_path (str): Path to the JSONL (or gzip-compressed) document file.
            Each line must be a JSON object.  May be a local path or an
            ``hfds:<repo>`` URL for Hugging Face Datasets.
        id_field (str): JSON key whose value is the document ID
            (default ``"id"``).
        content_fields (str or list[str]): JSON key(s) whose values are
            concatenated with ``sep`` to form the text payload.  Accepts a
            single string or a list for multi-field concatenation (e.g.
            ``["title", "body"]``).  Always normalized to a list after
            validation.
        sep (str): Separator inserted between concatenated content fields
            (default ``"\\n"``).
        offset_source (str): Strategy for random document access:

            * ``"offsetfile"`` *(default)* - builds a byte-offset map
              (``.offsetmap`` sidecar file) for fast O(1) lookup in a JSONL file.
            * ``"msmarco_seg"`` - reads from sharded gzipped files in the
              MSMARCO v2.1 segmented document format, using embedded byte offsets
              from the document ID.

        cache_dir (str, optional): Directory for the ``.offsetmap`` sidecar.
            When the dataset mount is read-only (common on shared compute
            filesystems), set this to a writable scratch path.  If unset,
            RoutIR tries to write next to the source file; on
            ``PermissionError`` it falls back to
            ``${XDG_CACHE_HOME:-~/.cache}/routir/offsetmap/``.  Inside the
            directory, the sidecar filename is
            ``<basename>.<hash16>.offsetmap`` so multiple corpora sharing
            one cache dir never collide.
    """

    source: Literal["text_jsonl"] = "text_jsonl"
    doc_path: str
    id_field: str = "id"
    content_fields: Union[str, List[str]] = "text"
    sep: str = "\n"
    offset_source: Literal["msmarco_seg", "offsetfile"] = "offsetfile"
    cache_dir: Optional[str] = None

    def model_post_init(self, __context):
        if not isinstance(self.content_fields, list):
            self.content_fields = [self.content_fields]


class LocalPathSource(BaseModel):
    """Bytes-from-filesystem view source: one file per id (or many via glob).

    Two modes:
      * Single file per id — set ``path_template`` (e.g. ``/data/kf/{id}.jpg``).
      * Multiple files per id — set ``path_glob`` (e.g. ``/data/kf/{id}_*.jpg``).
        All matching files are returned, sorted by filename.

    Exactly one of ``path_template`` / ``path_glob`` must be set.

    Attributes:
        source: Discriminator literal ``"local_path"``.
        path_template (str, optional): Format string producing exactly one
            path per id.  Resolved via ``str.format(id=...)``.
        path_glob (str, optional): Format string producing a glob expanded
            against the filesystem.  Resolved via ``str.format(id=...)``;
            the result is passed to :func:`glob.glob`.  Use this when one
            id maps to many files (multi-frame keyframes, multi-segment audio).
        base_dir (str, optional): Restrict all resolved paths to live under
            this directory (resolved against ``os.path.realpath``).  Resolved
            paths outside this root are rejected with a clear error so that
            user-supplied ids cannot escape the configured tree via ``../``.
        mime (str, optional): Content type hint forwarded to engines and
            clients (e.g. ``"image/jpeg"``, ``"audio/wav"``).  Per-view,
            applies to every match.
    """

    source: Literal["local_path"] = "local_path"
    path_template: Optional[str] = None
    path_glob: Optional[str] = None
    base_dir: Optional[str] = None
    mime: Optional[str] = None

    def model_post_init(self, __context):
        if (self.path_template is None) == (self.path_glob is None):
            raise ValueError(
                "LocalPathSource requires exactly one of path_template or path_glob"
            )


class GlobMatcher(BaseModel):
    """Anchored glob match for tar member names.

    ``pattern`` contains ``{id}`` which is substituted (re.escape'd) at lookup
    time.  fnmatch metacharacters in the surrounding text (``*``, ``?``, ``[``
    classes) translate to regex via :func:`fnmatch.translate`; the resulting
    regex is anchored at both ends so a partial substring match is impossible.
    """
    kind: Literal["glob"] = "glob"
    pattern: str       # e.g. "{id}.kf_uni5s.t*.jpg"


class RegexMatcher(BaseModel):
    """Anchored regex match for tar member names.

    ``pattern`` contains ``{id}`` which is substituted (re.escape'd) at lookup
    time.  The pattern is wrapped with ``^...$`` so a partial substring match
    is impossible.  Use this when fnmatch isn't expressive enough (e.g. you
    need to anchor a digit count or alternate extensions).
    """
    kind: Literal["regex"] = "regex"
    pattern: str       # e.g. r"{id}\.kf_uni5s\.t\d+\.jpg"


class ShardManifest(BaseModel):
    """CSV/TSV file mapping id -> shard token.

    Loaded once at backend init; cached in a process-wide singleton keyed by
    (path, id_column, shard_column).  Shard column values are str or int —
    both work with ``str.format(shard=...)`` (int allows ``{shard:06d}``).
    """
    kind: Literal["manifest"] = "manifest"
    path: str
    id_column: str = "id"
    shard_column: str = "shard"


class ShardModulo(BaseModel):
    """Derive shard from a deterministic hash of the id mod ``n``.

    Uses ``sha256(id.encode())[:8]`` interpreted as a big-endian uint64 —
    deterministic across processes and Python versions (Python's built-in
    ``hash()`` is salted per interpreter and would not be safe here).
    """
    kind: Literal["modulo"] = "modulo"
    n: int
    width: int = 4     # zero-pad in tar_template via {shard:0Nd}


class ShardSubstring(BaseModel):
    """Slice an id literally to derive its shard token.

    ``id[start:end]`` is the shard token (str).  Use this for layouts where
    a fixed prefix of the id is the bucket key.
    """
    kind: Literal["substring"] = "substring"
    start: int
    end: int


class TarSource(BaseModel):
    """Bytes-from-tar view source: lookup members in plain (uncompressed) tar shards.

    PR5b ships plain ``.tar`` support.  ``.tar.gz`` requires ``indexed_gzip``
    and lands in PR6.  Random access is via a sidecar ``.taridx`` file
    (member-name -> (offset, size)) built on first open, stamped with
    ``(size, mtime, head_sha256)`` of the tar so stale indexes rebuild
    automatically.  Reads use ``os.pread`` so concurrent fetches don't need
    a per-shard lock.

    Attributes:
        source: Discriminator literal ``"tar"``.
        tar_template (str): ``str.format(shard=...)`` template.  May reference
            ``{shard}`` with an optional format spec (e.g. ``{shard:06d}``).
        shard_resolver (Union[ShardManifest, ShardModulo, ShardSubstring], optional):
            How to derive the shard token from an id.  Omit when
            ``tar_template`` references no ``{shard}`` placeholder (single-tar
            collections).
        matcher (Union[GlobMatcher, RegexMatcher]): Member-name pattern.
            Substitutes ``{id}`` (re.escape'd) at lookup time, anchored.
        mime (str, optional): Content-type hint forwarded to engines / clients.
        cache_dir (str, optional): Directory for the ``.taridx`` sidecar.
            When the dataset mount is read-only (common on shared compute
            filesystems), set this to a writable scratch path.  If unset,
            RoutIR tries to write next to the tar; on ``PermissionError`` it
            falls back to ``${XDG_CACHE_HOME:-~/.cache}/routir/taridx/``.
            Inside the directory, the sidecar filename is
            ``<basename>.<hash16>.taridx`` so multiple shards sharing one
            cache dir never collide.
    """
    source: Literal["tar"] = "tar"
    tar_template: str
    shard_resolver: Optional[Union[ShardManifest, ShardModulo, ShardSubstring]] = Field(
        default=None, discriminator="kind"
    )
    matcher: Union[GlobMatcher, RegexMatcher] = Field(discriminator="kind")
    mime: Optional[str] = None
    cache_dir: Optional[str] = None


class ViewSpec(BaseModel):
    """One named view of a collection.

    A view describes how to materialize one modality of a document (e.g. OCR
    text, ASR transcript, keyframes).  Selected at pipeline-call time via
    the DSL ``@view`` suffix on a stage (PR3); collection swap then preserves
    the pipeline as long as the target collection exposes the same view names.

    Attributes:
        kind: ``"text"`` or ``"bytes"`` - declares the payload modality.
        source: The source/backend configuration for this view.  Discriminated
            on the inner ``"source"`` field: ``"text_jsonl"`` ->
            :class:`TextJsonlSource` (text view); ``"local_path"`` ->
            :class:`LocalPathSource` (bytes view); ``"tar"`` ->
            :class:`TarSource` (bytes view).
    """

    kind: Literal["text", "bytes"] = "text"
    source: Union[TextJsonlSource, LocalPathSource, TarSource] = Field(discriminator="source")

    def model_post_init(self, __context):
        if isinstance(self.source, TextJsonlSource) and self.kind != "text":
            raise ValueError("TextJsonlSource is only valid for kind='text' views")
        if isinstance(self.source, (LocalPathSource, TarSource)) and self.kind != "bytes":
            raise ValueError(
                f"{type(self.source).__name__} is only valid for kind='bytes' views"
            )


class CollectionConfig(BaseModel):
    """
    Configuration for a document collection.

    Collections expose the ``/content`` endpoint and are used by reranking
    pipeline stages to fetch document text by ID.

    A collection is a bundle of one or more **views**, each addressing a
    distinct modality or projection of the documents.  Most collections have
    a single text view; multi-view collections support per-stage view
    selection via the pipeline DSL (PR3).

    Example JSON (single-view, modern form)::

        {
            "name": "my-corpus",
            "views": {
                "text": {
                    "kind": "text",
                    "source": {
                        "source": "text_jsonl",
                        "doc_path": "/data/corpus.jsonl",
                        "id_field": "docid",
                        "content_fields": "text"
                    }
                }
            }
        }

    Example JSON (multi-view)::

        {
            "name": "my-corpus",
            "default_view": "ocr",
            "views": {
                "ocr": {"kind": "text", "source": {"source": "text_jsonl",
                        "doc_path": "/data/corpus.jsonl",
                        "id_field": "docid", "content_fields": "ocr"}},
                "asr": {"kind": "text", "source": {"source": "text_jsonl",
                        "doc_path": "/data/corpus.jsonl",
                        "id_field": "docid", "content_fields": "asr"}}
            }
        }

    Example JSON (legacy single-view, still accepted)::

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
        processor (str): Class name of the content-processor to use.
            Default ``"ContentProcessor"`` dispatches to the view backends.
            Use ``"IRDSProcessor"`` to load from an ``ir_datasets`` dataset ID.
        views (dict[str, ViewSpec]): Named views of this collection.  Each
            entry maps a view name to its :class:`ViewSpec` (kind + source).
            If left empty and the legacy ``doc_path`` field is set, a single
            text view named ``"text"`` is synthesized from the deprecated
            ``doc_path`` / ``id_field`` / ``content_field`` / ``offset_source``
            / ``cache_path`` fields.
        default_view (str, optional): Name of the view used when a request
            does not specify one.  Auto-elected when exactly one view exists;
            required to be set explicitly when multiple views are defined.
        id_to_lang_mapping (str, optional): Path to a pickle file mapping
            document IDs to language codes.  Used by processors that serve
            multilingual corpora.
        force_load_all_documents (bool): When ``True``, all documents are
            loaded into memory at startup for maximum throughput.  Only
            suitable for small corpora; default ``False`` uses on-demand
            offset-based access.

    Deprecated attributes (kept for back-compat synthesis into ``views``;
    new configs should use ``views`` directly):

        doc_path (str, optional): Legacy: path to the JSONL document file.
            Synthesizes a single text view named ``"text"`` when ``views`` is
            empty.
        id_field (str, optional): Legacy: JSON id field name.
        content_field (str or list[str], optional): Legacy: JSON content
            field(s).
        offset_source (str, optional): Legacy: random-access strategy.
        cache_path (str, optional): Legacy: path to the ``.offsetmap`` sidecar.
    """

    name: str
    processor: str = "ContentProcessor"
    views: Dict[str, ViewSpec] = Field(default_factory=dict)
    default_view: Optional[str] = None
    id_to_lang_mapping: Optional[str] = None
    force_load_all_documents: bool = False
    # In-memory LRU cache fronting per-id lookups.  Keyed by ``(view, id)`` so
    # the same id served across multiple views stays distinct.  Default-on
    # because bytes views (keyframes, tar shards) pay a real I/O cost per
    # fetch; set ``cache: 0`` to disable.  ``cache_ttl`` is in seconds.
    cache: int = 256
    cache_ttl: int = 600

    # ----- legacy / deprecated fields (back-compat for single-view configs) -----
    doc_path: Optional[str] = None
    id_field: Optional[str] = None
    content_field: Optional[Union[str, List[str]]] = None
    offset_source: Optional[Literal["msmarco_seg", "offsetfile"]] = None
    cache_path: Optional[str] = None

    def model_post_init(self, __context):
        # Back-compat synthesis: legacy single-view fields -> views={"text": ...}
        if not self.views and self.doc_path is not None:
            self.views = {
                "text": ViewSpec(
                    kind="text",
                    source=TextJsonlSource(
                        source="text_jsonl",
                        doc_path=self.doc_path,
                        id_field=self.id_field or "id",
                        content_fields=self.content_field or "text",
                        offset_source=self.offset_source or "offsetfile",
                        # Legacy top-level ``cache_path`` is folded into the new
                        # per-view ``cache_dir`` (PR7).  Same path, new name.
                        cache_dir=self.cache_path,
                    ),
                )
            }

        # Default-view rules
        if self.default_view is None and len(self.views) == 1:
            self.default_view = next(iter(self.views))
        if self.default_view is not None and self.default_view not in self.views:
            raise ValueError(
                f"default_view {self.default_view!r} not in views {list(self.views.keys())}"
            )
        if len(self.views) > 1 and self.default_view is None:
            raise ValueError(
                f"collection {self.name!r} has multiple views but no default_view set"
            )


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
        pipeline_aliases (dict[str, str]): Named shortcuts for pipeline DSL
            fragments.  An alias is a single identifier that expands to a
            (possibly multi-stage) pipeline at request time.  Aliases may
            reference other aliases; cycles raise an error at startup.  Each
            alias name must not collide with any service or collection name.

            Example::

                {
                    "ragtime2":    "{zho%100, rus%100, spa%100, eng%100}ScoreFusion",
                    "ragtime2-rr": "ragtime2%100 >> nemotron%20"
                }

            With these aliases, ``ragtime2%20`` expands to
            ``{zho%100, rus%100, spa%100, eng%100}ScoreFusion%20`` (the
            call-site limit is applied to the last stage of the alias body).
        pipeline_cache (int): Capacity for the pipeline-level result cache
            (number of entries).  ``-1`` (default) or ``0`` disables it.
            When enabled, the ``/pipeline`` endpoint reuses the final response
            for identical requests, keyed on the canonical AST of the pipeline
            (after alias expansion), the query, the collection, and the subset
            of ``runtime_kwargs`` that refer to aliases actually used.  Ignored
            when ``pipeline_cache_redis_url`` is set.
        pipeline_cache_ttl (int): Pipeline-cache entry TTL in seconds
            (default 600).
        pipeline_cache_redis_url (str, optional): Redis URL for the pipeline
            cache.  When set, Redis replaces the in-memory LRU cache.
        pipeline_cache_redis_kwargs (dict, optional): Extra kwargs forwarded
            to the Redis client for the pipeline cache.
        relay_content_cache (int): Capacity for the local cache that fronts
            remote ``content`` services registered via ``server_imports``.
            ``-1`` (default) or ``0`` disables it.  Without this cache,
            reranking against a remote collection re-fetches every document
            text on every query (the per-run ``doc_content_cache`` on
            :class:`SearchPipeline` only dedupes within a single request).
        relay_content_cache_ttl (int): TTL in seconds for the relay-content
            cache (default 600).
        relay_content_cache_redis_url (str, optional): Redis URL for the
            relay-content cache.
        relay_content_cache_redis_kwargs (dict, optional): Extra kwargs
            forwarded to the Redis client for the relay-content cache.
        bytes_content_cache_max_bytes (int, optional): Per-request cap on the
            size of the bytes-view payloads cached inside one
            :class:`~routir.pipeline.pipeline.SearchPipeline` run
            (``doc_content_cache``).  When set, the cache evicts in insertion
            order (FIFO) once the inserted bytes exceed the cap.  Only
            bytes-view payloads count toward the cap; text-view entries stay
            free.  ``None`` (default) disables eviction.
    """

    model_config = ConfigDict(extra="forbid")

    services: List[ServiceConfig] = Field(default_factory=list)
    collections: List[CollectionConfig] = Field(default_factory=list)
    # Each entry is either a REST URL string or a dict like
    # ``{"endpoint": "http://...", "grpc_endpoint": "host:50051", ...}``.
    # Extra fields (``grpc_endpoint``, ``api_key``, etc.) are passed through
    # to the auto-created :class:`~routir.models.Relay` config so the data
    # plane can use gRPC even though discovery always uses REST.
    server_imports: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    file_imports: List[str] = Field(default_factory=list)
    dynamic_pipeline: bool = True
    pipeline_aliases: Dict[str, str] = Field(default_factory=dict)

    pipeline_cache: int = -1
    pipeline_cache_ttl: int = 600
    pipeline_cache_redis_url: Optional[str] = None
    pipeline_cache_redis_kwargs: Dict[str, Any] = Field(default_factory=dict)

    relay_content_cache: int = -1
    relay_content_cache_ttl: int = 600
    relay_content_cache_redis_url: Optional[str] = None
    relay_content_cache_redis_kwargs: Dict[str, Any] = Field(default_factory=dict)

    bytes_content_cache_max_bytes: Optional[int] = None
