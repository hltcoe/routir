"""ViewBackend ABC: resolves (doc_id) -> payload for one named view.

A view backend is the storage strategy behind a single named view of a
collection (e.g. ``ocr`` text or ``keyframe`` bytes).  ``ContentProcessor``
dispatches to the right backend by view name.

PR1 ships only ``TextJsonlBackend``; PR5a/5b add local-path and tar backends.
Third-party backends are loaded via ``file_imports`` and discovered through
``FactoryEnabled`` - concrete backend classes should never need to be
imported from ``processors/``.
"""

from typing import TYPE_CHECKING, Any, Dict

from ...utils import FactoryEnabled

if TYPE_CHECKING:
    from ...config import CollectionConfig  # avoid circular


class ViewBackend(FactoryEnabled):
    """Abstract storage backend for one view of a collection.

    Subclasses set the class attribute ``kind`` to ``"text"`` or ``"bytes"``
    and implement ``__getitem__`` / ``__contains__``.

    Attributes:
        kind: ``"text"`` or ``"bytes"``; declares the payload modality.
        name: View name (the key in ``CollectionConfig.views``).
        spec: The pydantic source model for this view.
        collection_config: The parent collection config (read-only).
    """

    kind: str = "text"

    def __init__(self, name: str, spec, collection_config: "CollectionConfig"):
        self.name = name
        self.spec = spec
        self.collection_config = collection_config

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        """Return the payload dict for ``doc_id``.

        For ``kind="text"``: at minimum ``{"text": <str>}``; may include
        ``"title"`` and other view-specific fields.
        For ``kind="bytes"``: ``{"data": List[bytes]}`` - even when only one
        blob is present, wrap it in a list of length 1.

        Raises ``KeyError`` if the id is not present.
        """
        raise NotImplementedError

    def __contains__(self, doc_id: str) -> bool:
        raise NotImplementedError
