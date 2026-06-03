import time
from typing import Any, Dict, List, Union

from ..models.abstract import Engine
from .abstract import Processor


_callable_service_types = {"search", "score", "decompose_query", "fuse"}
_uncallable_service_types = {"content"}

# TODO: why am I creating this mess...
_output_keys = {"search": "scores", "score": "scores", "decompose_query": "queries", "fuse": "scores"}


class ServiceNotFound(KeyError):
    """Raised when a (name, service_type) combo is not registered.

    KeyError so existing code that catches KeyError still works, but the
    distinct type lets transport layers map cleanly to HTTP 400 / gRPC NOT_FOUND.
    """

    def __str__(self) -> str:
        # KeyError wraps its message in repr() quotes by default; override so
        # str(e) yields the plain message that REST/gRPC surface to callers.
        return self.args[0] if self.args else ""


class _ProcessorRegistry:
    """
    Global registry for all processors and services.

    Manages processor instances indexed by service name and type.

    Attributes:
        all_services: Nested dict of service_name -> service_type -> processor
        slot_meta: Nested dict of service_name -> service_type -> metadata dict.
            Mirrors :attr:`all_services` and carries per-slot metadata such as
            ``view_kind`` (for score/search slots) and ``views`` / ``default_view``
            (for content slots).  Empty dict if no metadata was supplied at
            registration time.
    """

    def __init__(self):
        """Initialize empty registry."""
        # TODO: make collection name part of the namespace
        self.all_services: Dict[str, Dict[str, Processor]] = {}
        # Parallel metadata store: same outer/inner keys as ``all_services``.
        self.slot_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @property
    def valid_service_types(self):
        """Get set of valid service types."""
        return _callable_service_types | _uncallable_service_types

    def __getitem__(self, name: str):
        """Get all processors for a service name."""
        return self.all_services[name]

    def register(self, name: str, service_type: str, processor: Processor, **meta: Any):
        """
        Register a processor.

        Args:
            name: Service name
            service_type: Service type ('search', 'score', 'content', etc.)
            processor: Processor instance
            **meta: Optional per-slot metadata.  Recognised keys include
                ``view_kind`` (str, for score/search slots) and ``views`` /
                ``default_view`` (for content slots).  The registry is agnostic
                to specific keys — callers read them back via :meth:`get_meta`.

        Raises:
            ValueError: If service type is invalid or service already exists.
        """
        if service_type not in self.valid_service_types:
            raise ValueError(f"Invalid service type '{service_type}'. Valid types: {self.valid_service_types}")
        if name in self.all_services and service_type in self.all_services[name]:
            raise ValueError(f"Service type '{service_type}' of name '{name}' already exists.")

        if name not in self.all_services:
            self.all_services[name] = {}
            self.slot_meta[name] = {}
        self.all_services[name][service_type] = processor
        self.slot_meta[name][service_type] = dict(meta) if meta else {}

    def get_meta(self, name: str, service_type: str) -> Dict[str, Any]:
        """Return the metadata dict for a registered slot (empty dict if absent)."""
        return self.slot_meta.get(name, {}).get(service_type, {})

    def get_slot_meta(self, name: str, service_type: str, key: str, default: Any = None) -> Any:
        """Convenience accessor: return one metadata value for a slot."""
        return self.slot_meta.get(name, {}).get(service_type, {}).get(key, default)

    def has_service(self, name: str, service_type: str):
        """Check if a service exists."""
        return name in self.all_services and service_type in self.all_services[name]

    def get(self, name: str, service_type: str):
        """Get a processor by service name and type."""
        if self.has_service(name, service_type):
            return self.all_services[name][service_type]

    async def submit(self, name: str, service_type: str, data: dict) -> dict:
        if not self.has_service(name, service_type):
            raise ServiceNotFound(
                f"Service '{name}' not found or does not support {service_type}"
            )
        return await self.get(name, service_type).submit(data)

    def get_all_services(self):
        """
        Get all registered services grouped by type.

        Returns:
            Dict mapping service types to lists of service names
        """
        services = {t: [] for t in self.valid_service_types}
        for name, processors in self.all_services.items():
            for s in processors:
                services[s].append(name)
        return services


class DummyProcessor(Processor):
    """A minimal, no-cache processor that wraps a single engine method.

    Used internally by :func:`auto_register` to expose an engine method
    (``search``, ``fuse``, etc.) as a :class:`Processor` without any batching
    or caching overhead.  Requests are dispatched directly to the engine method
    on every call.

    Not intended for production workloads where batching matters; prefer the
    config-based loading path via :func:`~routir.config.load.load_config`,
    which creates :class:`~routir.processors.abstract.BatchProcessor` instances.
    """

    def __init__(self, engine: Engine, method: str):
        super().__init__(cache_size=0)
        if not hasattr(engine, method):
            raise ValueError(f"Engine {engine.__class__.__name__} does not have method '{method}'")
        self.service = getattr(engine, method)
        self.result_key = _output_keys[method]

    async def submit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {self.result_key: (await self.service(**item)), "processed": True, "timestamp": time.time()}


# singleton
ProcessorRegistry = _ProcessorRegistry()


def auto_register(methods: Union[str, List[str]], view_kind: str = None, **default_init_kwargs):
    """
    Class decorator that instantiates an engine and registers it as a processor.

    This is a lightweight alternative to the JSON config-based loading path,
    useful for **built-in** or **singleton** engines (e.g. stateless fusion rules
    like RRF) that do not need per-service batching or caching.

    .. warning::

        ``auto_register`` creates a single engine instance with ``default_init_kwargs``
        at decoration time using :class:`DummyProcessor` (no batching, no caching).
        For production engines with GPU models, use the config-based path instead so
        that :class:`~routir.processors.abstract.BatchProcessor` and LRU/Redis caching
        are applied.

    The engine class must implement the ``{method}_batch`` method for every
    *method* listed.  The decorator checks the corresponding ``can_{method}``
    property (e.g. ``can_fuse`` for ``"fuse"``) and raises ``TypeError`` if it
    returns ``False``.

    The service is registered under the engine's *class name* in
    :data:`ProcessorRegistry`.

    Args:
        methods (str or list[str]): Service type(s) to register. Valid values:
            ``"search"``, ``"score"``, ``"fuse"``, ``"decompose_query"``.
        view_kind (str, optional): Modality of the engine's ``score_batch`` /
            ``search_batch`` inputs (``"text"`` or ``"bytes"``).  When omitted,
            falls back to the engine class's ``accepts_view_kind`` attribute
            (which defaults to ``"text"``).  Only ``score`` / ``search`` slots
            store this metadata; ``fuse`` and ``decompose_query`` always
            register as ``"text"``.
        **default_init_kwargs: Keyword arguments forwarded to ``engine_cls()``
            at decoration time.

    Returns:
        The unmodified engine class (decorator is transparent).

    Example:
        .. code-block:: python

            @auto_register("fuse")
            class RRFFusion(Engine):
                async def fuse_batch(self, queries, batch_scores, **kwargs):
                    ...

            # RRFFusion is now accessible as ProcessorRegistry["RRFFusion"]["fuse"]
    """
    if isinstance(methods, str):
        methods = [methods]

    def engine_dec(engine_cls: type[Engine]):
        """Decorator function that registers the engine."""
        if not issubclass(engine_cls, Engine):
            raise TypeError(f"{engine_cls.__name__} must be a subclass of Engine")
        for method in methods:
            if not getattr(engine_cls, f"can_{method}"):
                raise TypeError(
                    f"Engine {engine_cls.__name__} does not implement '{method}' "
                    f"(can_{method} is False - did you override {method}_batch?)"
                )
        resolved_view_kind = view_kind or getattr(engine_cls, "accepts_view_kind", "text")
        # register each
        for method in methods:
            slot_meta = {}
            # Only score/search slots care about modality; fuse / decompose_query
            # always work on text scores or strings.
            if method in {"score", "search"}:
                slot_meta["view_kind"] = resolved_view_kind
            else:
                slot_meta["view_kind"] = "text"
            ProcessorRegistry.register(
                engine_cls.__name__,
                method,
                processor=DummyProcessor(engine_cls(**default_init_kwargs), method),
                **slot_meta,
            )

        return engine_cls

    return engine_dec
