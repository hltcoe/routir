"""LocalPathBackend: filesystem-bytes view backend.

Reads byte blobs from local files.  Supports two modes:
* ``path_template`` — single file per id (length-1 ``data`` list).
* ``path_glob`` — multiple files per id, sorted by filename (length-N).
* Zero matches return ``data=[]``; this is NOT an error.

Path-traversal guard: when ``base_dir`` is set, every resolved path's
``os.path.realpath`` must live under ``os.path.realpath(base_dir)``; ids
that escape that root raise ``ValueError``.  When ``base_dir`` is not set,
no traversal check is performed — the caller is responsible for trusting
the id space.

PR5b adds the tar-shard equivalent (``TarBackend``).  Same return shape:
``{"data": List[bytes], "mime": <hint>}``.
"""

import glob as _glob
import os
from typing import Any, Dict, List, Optional

from .abstract import ViewBackend


class LocalPathBackend(ViewBackend):
    """Filesystem-bytes view backend.

    Two modes selected by which field of :class:`~routir.config.LocalPathSource`
    is set:

    * ``path_template`` — one file per id; missing file raises ``KeyError``.
    * ``path_glob`` — zero or more files per id; zero matches is a legal empty
      payload (``data=[]``) and ``__contains__`` returns ``True``.

    The "always present" semantics for glob mode are what makes audio-only
    chunks (zero keyframes) work without raising in the rerank stage; the
    bytes engine receives an empty inner list and handles it.
    """

    kind = "bytes"

    def __init__(self, name, spec, collection_config):
        super().__init__(name, spec, collection_config)
        self.path_template = spec.path_template
        self.path_glob = spec.path_glob
        self.mime = spec.mime
        # Resolve base_dir up front so the per-id check is a string-prefix test.
        self.base_dir_real: Optional[str] = (
            os.path.realpath(spec.base_dir) if spec.base_dir is not None else None
        )

    def _resolve(self, doc_id: str) -> List[str]:
        if self.path_template is not None:
            paths = [self.path_template.format(id=doc_id)]
        else:
            pat = self.path_glob.format(id=doc_id)
            paths = sorted(_glob.glob(pat))

        if self.base_dir_real is not None:
            sep = os.sep
            root = self.base_dir_real.rstrip(sep) + sep
            for p in paths:
                real = os.path.realpath(p)
                if real != self.base_dir_real and not real.startswith(root):
                    raise ValueError(
                        f"resolved path {real!r} for id {doc_id!r} escapes base_dir "
                        f"{self.base_dir_real!r}"
                    )
        return paths

    def __getitem__(self, doc_id: str) -> Dict[str, Any]:
        paths = self._resolve(doc_id)
        # path_template: file must exist (raise KeyError).
        # path_glob: zero matches is legal (empty data list).
        if self.path_template is not None:
            if not paths or not os.path.exists(paths[0]):
                raise KeyError(doc_id)
        parts: List[bytes] = []
        for p in paths:
            if os.path.exists(p):
                with open(p, "rb") as fp:
                    parts.append(fp.read())
        payload: Dict[str, Any] = {"data": parts}
        if self.mime:
            payload["mime"] = self.mime
        return payload

    def __contains__(self, doc_id: str) -> bool:
        try:
            paths = self._resolve(doc_id)
        except ValueError:
            return False
        if self.path_template is not None:
            return bool(paths) and os.path.exists(paths[0])
        # path_glob: present even when zero matches (legal "empty" result).
        return True
