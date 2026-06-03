"""View-dispatch entry point for local document collections.

Houses :class:`ContentProcessor` (view-dispatching, backed by one
:class:`ViewBackend` per declared view) and :class:`IRDSProcessor`
(``ir_datasets`` backed) - the two ways a RoutIR server resolves
``(collection, view, doc_id) -> payload`` for collections that live on the
local machine.  Remote collections are handled by :mod:`routir.collections.relay`.

Dependency direction:
    This module imports :class:`Processor` from
    :mod:`routir.processors.abstract`.  The reverse must never happen -
    :mod:`routir.processors` provides the online request/cache plumbing and
    must not depend on collection storage.
"""

from typing import Any, Dict, Tuple, Union

from ..config import CollectionConfig, LocalPathSource, TarSource, TextJsonlSource
from ..processors.abstract import Processor
from ..utils import load_singleton
from .views import LocalPathBackend, TarBackend, TextJsonlBackend, ViewBackend


# Map source pydantic model -> backend class.  PR1 ships one entry; PR5a adds
# ``LocalPathSource``; PR5b adds ``TarSource``.  We're intentionally not
# using FactoryEnabled.load for this mapping because backends are matched by
# source type, not by class name.
_BACKEND_FOR_SOURCE = {
    TextJsonlSource: TextJsonlBackend,
    LocalPathSource: LocalPathBackend,
    TarSource:       TarBackend,
}


def _backend_for(source) -> type:
    try:
        return _BACKEND_FOR_SOURCE[type(source)]
    except KeyError:
        raise ValueError(f"No backend registered for source type {type(source).__name__}")


def _view_id_key(item: Dict[str, Any], default_view: str) -> Tuple[str, str]:
    """Cache-key helper: (view, doc_id) tuple.

    Falls back to ``default_view`` when the request omits ``view``.
    """
    return (item.get("view") or default_view, item["id"])


class ContentProcessor(Processor):
    """
    Processor for retrieving document content by ID, dispatched per view.

    Builds one :class:`ViewBackend` per entry in
    :attr:`CollectionConfig.views` at startup; per-request dispatch picks
    the right backend by view name (defaulting to ``default_view`` when the
    request omits ``view``).

    Attributes:
        config: Collection configuration.
        default_view: View used when a request omits ``view``.
        backends: Mapping of view name to :class:`ViewBackend` instance.
        lang_mapping: Optional mapping of document IDs to language codes.
    """

    def __init__(self, collection_config: CollectionConfig, cache_size=256, cache_ttl=600):
        """
        Initialize content processor.

        Args:
            collection_config: Collection configuration with at least one view
                (either declared directly via ``views`` or synthesized from
                the deprecated ``doc_path`` / ``content_field`` / ... fields).
            cache_size: Maximum cache entries.  Defaults to a small in-memory
                LRU so per-id lookups (especially bytes views like keyframes,
                which incur real disk/tar I/O) don't pay the I/O cost on every
                repeat fetch.  Set to ``0`` to disable.
            cache_ttl: Cache TTL in seconds.
        """
        # Cache key includes the view name so two requests for the same id but
        # different views are distinct entries.  default_view fallback keeps
        # legacy ``{"id": ...}``-only requests working.
        default_view = collection_config.default_view
        super().__init__(cache_size, cache_ttl, lambda x, _dv=default_view: _view_id_key(x, _dv))

        self.config = collection_config
        self.default_view = default_view

        self.backends: Dict[str, ViewBackend] = {}
        for view_name, view_spec in collection_config.views.items():
            backend_cls = _backend_for(view_spec.source)
            self.backends[view_name] = backend_cls(
                name=view_name,
                spec=view_spec.source,
                collection_config=collection_config,
            )

        self.lang_mapping = None
        if collection_config.id_to_lang_mapping is not None:
            self.lang_mapping: Dict[str, str] = load_singleton(collection_config.id_to_lang_mapping)

    def _resolve_key(self, key: Union[str, Tuple[str, str]]) -> Tuple[str, str]:
        if isinstance(key, tuple):
            return key
        return (self.default_view, key)

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> Dict[str, Any]:
        view, doc_id = self._resolve_key(key)
        backend = self.backends[view]
        payload = dict(backend[doc_id])
        payload["view"] = view
        payload["kind"] = backend.kind
        if self.lang_mapping is not None:
            payload["language"] = self.lang_mapping.get(doc_id, "")
        return payload

    def __contains__(self, key: Union[str, Tuple[str, str]]) -> bool:
        view, doc_id = self._resolve_key(key)
        backend = self.backends.get(view)
        if backend is None:
            return False
        return doc_id in backend

    async def _submit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        view = item.get("view") or self.default_view
        if view not in self.backends:
            return {"error": f"View {view!r} is not defined for collection {self.config.name!r}."}
        doc_id = item["id"]
        backend = self.backends[view]
        if doc_id not in backend:
            return {"error": f"ID {doc_id} is not found."}
        payload = dict(backend[doc_id])
        payload["view"] = view
        payload["kind"] = backend.kind
        if self.lang_mapping is not None:
            payload["language"] = self.lang_mapping.get(doc_id, "")
        return payload


class IRDSProcessor(Processor):
    """
    Processor for retrieving document content by ID from IRDS format.

    Inherits from ContentProcessor and uses IRDS-specific line reader.
    """

    def __init__(self, collection_config: CollectionConfig, cache_size=256, cache_ttl=600):
        """
        Initialize content processor.

        Args:
            collection_config: Collection configuration with doc_path, id_field, etc.
            cache_size: Maximum cache entries (default 256).
            cache_ttl: Cache TTL in seconds.
        """
        # always use `id` from the request as the key, this is different from id_field in config
        super().__init__(cache_size, cache_ttl, lambda x: x["id"])

        import ir_datasets as irds

        self.config = collection_config
        # IRDSProcessor still reads content_field from the legacy top-level
        # field; multi-view IRDS support is out of scope for PR1.
        cf = collection_config.content_field
        if cf is None:
            cf = ["text"]
        elif not isinstance(cf, list):
            cf = [cf]
        self.content_field = cf
        self.ds = irds.load(collection_config.name).docs

    def __getitem__(self, idx: str):
        doc: dict[str, str] = self.ds.lookup(idx)._asdict()
        results = {"text": "\n".join(doc[c] for c in self.content_field)}
        if "title" in doc:
            results["title"] = doc["title"]

        return results

    def __contains__(self, idx: str):
        try:
            _ = self.ds.lookup(idx)
            return True
        except KeyError:
            return False

    async def _submit(self, item: Dict[str, Any]) -> Dict[str, str]:
        return (
            self[self.cache_key(item)] if self.cache_key(item) in self else {"error": f"ID {self.cache_key(item)} is not found."}
        )
