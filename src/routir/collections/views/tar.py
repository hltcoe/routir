"""TarBackend: bytes-from-tar view backend.

Reads byte blobs from members inside plain ``.tar`` shards using a sidecar
``.taridx`` index for O(1) lookup.  Concurrent reads use ``os.pread`` on a
per-shard file descriptor — no lock, no recompressing.

PR5b: plain .tar only.  ``.tar.gz`` support lives in PR6 behind ``indexed_gzip``.

Shard resolution: ``shard_resolver`` derives a shard token from the doc id,
which is interpolated into ``tar_template`` (``str.format(shard=...)``).
Single-tar collections may omit ``shard_resolver`` if ``tar_template``
references no ``{shard}`` placeholder.

Anchored matchers: ``{id}`` in the pattern is ``re.escape``'d before being
compiled, and the resulting regex is anchored at both ends (``^...$``).
This prevents ``id="abc"`` from spuriously matching ``abcd_*.jpg``.
"""

import csv as _csv
import fnmatch
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...config.config import (
    GlobMatcher, RegexMatcher,
    ShardManifest, ShardModulo, ShardSubstring,
)
from ..indexing.tar_index import build_or_load_taridx
from .abstract import ViewBackend


# Module-level singletons (per-process, shared across all TarBackend instances):

# tar path -> {member: (offset, size)}
_INDEX_CACHE: Dict[str, Dict[str, Tuple[int, int]]] = {}
# tar path -> sorted member names
_INDEX_KEYS_CACHE: Dict[str, List[str]] = {}
# tar path -> os.open() fd (read-only)
_FD_CACHE: Dict[str, int] = {}
# (manifest_path, id_col, shard_col) -> id->shard
_MANIFEST_CACHE: Dict[Tuple[str, str, str], Dict[str, str]] = {}

# Dedicated IO executor so tar reads don't starve the default thread pool.
# Worker count from ROUTIR_TAR_IO_WORKERS env var (default 32).
_TAR_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    global _TAR_EXECUTOR
    if _TAR_EXECUTOR is None:
        n = int(os.environ.get("ROUTIR_TAR_IO_WORKERS", "32"))
        _TAR_EXECUTOR = ThreadPoolExecutor(max_workers=n, thread_name_prefix="routir-tar-io")
    return _TAR_EXECUTOR


def _load_manifest(spec: ShardManifest) -> Dict[str, str]:
    key = (spec.path, spec.id_column, spec.shard_column)
    if key not in _MANIFEST_CACHE:
        with open(spec.path, newline="") as fp:
            # tab- or comma-separated: sniff by suffix.
            dialect = "excel-tab" if spec.path.endswith(".tsv") else "excel"
            reader = _csv.DictReader(fp, dialect=dialect)
            _MANIFEST_CACHE[key] = {
                row[spec.id_column]: row[spec.shard_column] for row in reader
            }
    return _MANIFEST_CACHE[key]


def _resolve_shard(resolver, doc_id: str):
    if resolver is None:
        return None
    if isinstance(resolver, ShardManifest):
        mapping = _load_manifest(resolver)
        if doc_id not in mapping:
            raise KeyError(f"id '{doc_id}' not found in shard manifest {resolver.path}")
        val = mapping[doc_id]
        # Convert int-like strings so ``{shard:06d}`` works.
        try:
            return int(val)
        except (TypeError, ValueError):
            return val
    if isinstance(resolver, ShardModulo):
        h = int.from_bytes(hashlib.sha256(doc_id.encode()).digest()[:8], "big")
        return h % resolver.n
    if isinstance(resolver, ShardSubstring):
        return doc_id[resolver.start:resolver.end]
    raise TypeError(f"unknown shard resolver: {type(resolver).__name__}")


def _compile_matcher(spec, doc_id: str) -> re.Pattern:
    """Anchored regex with ``re.escape``'d id."""
    if isinstance(spec, GlobMatcher):
        placeholder = "__ROUTIR_ID_PLACEHOLDER__"
        skel = spec.pattern.replace("{id}", placeholder)
        translated = fnmatch.translate(skel)
        # fnmatch.translate already anchors; substitute the placeholder for the escaped id.
        regex = translated.replace(re.escape(placeholder), re.escape(doc_id))
        return re.compile(regex)
    if isinstance(spec, RegexMatcher):
        pat = spec.pattern.replace("{id}", re.escape(doc_id))
        if not pat.startswith("^"):
            pat = "^" + pat
        if not pat.endswith("$"):
            pat = pat + "$"
        return re.compile(pat)
    raise TypeError(f"unknown matcher: {type(spec).__name__}")


class TarBackend(ViewBackend):
    """View backend for bytes stored inside plain ``.tar`` shards.

    See module docstring for design notes.  Return shape mirrors
    :class:`LocalPathBackend`: ``{"data": List[bytes], "mime": <hint>}``.
    Zero matches return an empty data list rather than raising — same as
    ``LocalPathBackend``'s ``path_glob`` mode.
    """

    kind = "bytes"

    def __init__(self, name, spec, collection_config):
        super().__init__(name, spec, collection_config)
        self.tar_template = spec.tar_template
        self.shard_resolver = spec.shard_resolver
        self.matcher_spec = spec.matcher
        self.mime = spec.mime
        self.cache_dir = spec.cache_dir

    def _shard_to_tar_path(self, shard) -> str:
        if shard is None:
            return self.tar_template
        return self.tar_template.format(shard=shard)

    def _ensure_open(self, tar_path: str) -> Tuple[int, Dict[str, Tuple[int, int]], List[str]]:
        """Ensure the index and fd are cached for *tar_path*; return ``(fd, index, sorted_keys)``."""
        if tar_path not in _INDEX_CACHE:
            if tar_path.endswith(".gz"):
                raise NotImplementedError(
                    f".tar.gz random access requires indexed_gzip (PR6); "
                    f"got {tar_path}"
                )
            _INDEX_CACHE[tar_path] = build_or_load_taridx(
                Path(tar_path), cache_dir=self.cache_dir
            )
            _INDEX_KEYS_CACHE[tar_path] = sorted(_INDEX_CACHE[tar_path].keys())
        if tar_path not in _FD_CACHE:
            _FD_CACHE[tar_path] = os.open(tar_path, os.O_RDONLY)
        return _FD_CACHE[tar_path], _INDEX_CACHE[tar_path], _INDEX_KEYS_CACHE[tar_path]

    def _matching_members(self, doc_id: str, sorted_keys: List[str]) -> List[str]:
        """Return all member names that match the matcher for *doc_id*, sorted by name."""
        pat = _compile_matcher(self.matcher_spec, doc_id)
        # bisect would narrow the search space when the prefix is fixed; for simplicity
        # we full-scan the per-shard key list.  PR-future: prefix-bucket the index.
        return [k for k in sorted_keys if pat.match(k)]

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        shard = _resolve_shard(self.shard_resolver, doc_id)
        tar_path = self._shard_to_tar_path(shard)
        fd, index, sorted_keys = self._ensure_open(tar_path)
        members = self._matching_members(doc_id, sorted_keys)
        parts: List[bytes] = []
        for name in members:
            offset, size = index[name]
            parts.append(os.pread(fd, size, offset))
        payload: Dict[str, Any] = {"data": parts}
        if self.mime:
            payload["mime"] = self.mime
        return payload

    def __contains__(self, doc_id: str) -> bool:
        try:
            shard = _resolve_shard(self.shard_resolver, doc_id)
        except KeyError:
            return False
        tar_path = self._shard_to_tar_path(shard)
        try:
            _, _, sorted_keys = self._ensure_open(tar_path)
        except (FileNotFoundError, OSError):
            return False
        return bool(self._matching_members(doc_id, sorted_keys))
