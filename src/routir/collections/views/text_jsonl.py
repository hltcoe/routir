"""TextJsonlBackend: text-jsonl view storage backend.

Wraps the legacy ``OffsetFile`` / ``MSMARCOSegOffset`` random-access readers
behind the :class:`ViewBackend` interface.  Two views over the same jsonl
file (e.g. ``ocr`` field and ``asr`` field of the same corpus) share a
single underlying reader, keyed by ``(doc_path, id_field)``.
"""

import json
from typing import Any, Dict, Tuple

from ...utils import logger
from ..indexing.offset_file import MSMARCOSegOffset, OffsetFile, RandomAccessReader
from .abstract import ViewBackend


# Module-level cache: (doc_path, id_field) -> RandomAccessReader.
# Shared across all TextJsonlBackend instances within a process so two views
# over one jsonl don't each build their own .offsetmap.
_READER_CACHE: Dict[Tuple[str, str], RandomAccessReader] = {}


def _get_or_build_reader(spec, collection_config) -> RandomAccessReader:
    key = (str(spec.doc_path), spec.id_field)
    if key not in _READER_CACHE:
        if spec.offset_source == "offsetfile":
            _READER_CACHE[key] = OffsetFile(
                spec.doc_path,
                key=lambda line, _id=spec.id_field: json.loads(line)[_id],
                cache_dir=spec.cache_dir,
                id_field=spec.id_field,
            )
        elif spec.offset_source == "msmarco_seg":
            _READER_CACHE[key] = MSMARCOSegOffset(
                spec.doc_path,
                force_load_all=collection_config.force_load_all_documents,
            )
        else:
            raise ValueError(f"unknown offset_source: {spec.offset_source!r}")
    else:
        logger.debug(f"Reusing cached reader for {key}")
    return _READER_CACHE[key]


class TextJsonlBackend(ViewBackend):
    """View backend for text fields stored in a JSONL document file.

    Returns ``{"text": "<concatenated content_fields>"}`` plus a ``"title"``
    field when present in the underlying document (preserved for back-compat
    with the legacy ContentProcessor response shape).
    """

    kind = "text"

    def __init__(self, name, spec, collection_config):
        super().__init__(name, spec, collection_config)
        self.reader = _get_or_build_reader(spec, collection_config)
        cf = spec.content_fields
        self.content_fields = cf if isinstance(cf, list) else [cf]
        self.sep = spec.sep

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        line = self.reader[doc_id]
        if not line:
            raise KeyError(doc_id)
        doc = json.loads(line)
        payload: Dict[str, Any] = {
            "text": self.sep.join(doc[c] for c in self.content_fields),
        }
        if "title" in doc:
            payload["title"] = doc["title"]
        return payload

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.reader
