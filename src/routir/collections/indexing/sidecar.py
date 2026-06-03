"""Sidecar path resolution with a writable-fallback chain.

Used by both ``OffsetFile`` (``.offsetmap``) and ``tar_index``
(``.taridx``) so the two index types behave identically when the
dataset directory is read-only -- a very common production case.

Resolution priority for both load and write (first viable wins):

  1. User-specified ``cache_dir`` (per-view ``cache_dir`` field on the
     source spec).  Filename inside it is ``<basename>.<hash16>.<suffix>``
     where the hash is the first 16 hex chars of
     ``sha256(realpath(source))`` -- so multiple sources sharing one cache
     dir never collide.
  2. Adjacent to the source: ``<source_path>.<suffix>``.  Unhashed name
     (back-compat with the original PR1/PR5b layout for writable mounts).
  3. ``${XDG_CACHE_HOME:-~/.cache}/routir/<suffix without leading dot>/<basename>.<hash16>.<suffix>``.

On load, the helper returns the highest-priority candidate that exists.
On write, it walks the list and writes to the first candidate whose
``mkstemp`` + ``os.replace`` succeeds; lower-priority candidates that
raise ``PermissionError`` / ``OSError`` are skipped silently.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from ...utils import logger


def _hash16(source: Path) -> str:
    return hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]


def _xdg_cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "routir"


def resolve_sidecar_candidates(
    source: Path,
    suffix: str,
    user_cache_dir: Optional[str] = None,
) -> List[Path]:
    """Return sidecar candidates in priority order.

    ``suffix`` includes the leading dot (``".taridx"``, ``".offsetmap"``).
    """
    source = Path(source)
    out: List[Path] = []
    if user_cache_dir:
        h = _hash16(source)
        out.append(Path(user_cache_dir) / f"{source.name}.{h}{suffix}")
    out.append(source.parent / (source.name + suffix))
    h = _hash16(source)
    out.append(_xdg_cache_root() / suffix.lstrip(".") / f"{source.name}.{h}{suffix}")
    return out


def find_existing_sidecar(candidates: List[Path]) -> Optional[Path]:
    """Return the first candidate that exists on disk, or ``None``."""
    for c in candidates:
        if c.exists():
            return c
    return None


def atomic_write_sidecar(
    candidates: List[Path],
    writer: Callable[[Path], None],
) -> Path:
    """Walk ``candidates`` and write to the first one that succeeds.

    ``writer`` is invoked with the tmp file path created via ``mkstemp``
    (after the fd is closed) so the writer can open it for writing,
    serialise the payload, flush, and fsync -- this helper stays agnostic
    about the payload format.

    Returns the final sidecar path on success.  Raises ``PermissionError``
    with a combined message when every candidate is unwritable.  Non-IO
    exceptions raised by ``writer`` (e.g. pickle errors) propagate
    immediately -- those aren't a "try another location" case.
    """
    errors = []
    for target in candidates:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            errors.append(f"  {target}: mkdir failed: {e}")
            continue
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=target.name + ".tmp.",
                dir=str(target.parent),
            )
        except (PermissionError, OSError) as e:
            errors.append(f"  {target}: mkstemp failed: {e}")
            continue
        tmp_path_p = Path(tmp_path)
        try:
            # Caller-supplied writer takes the tmp path, opens it as needed,
            # writes, flushes, and fsyncs.  Close the fd we already have
            # before handing the path off, since the writer will reopen it.
            os.close(fd)
            writer(tmp_path_p)
            os.replace(str(tmp_path_p), str(target))
            logger.debug(f"Wrote sidecar {target}")
            return target
        except (PermissionError, OSError) as e:
            errors.append(f"  {target}: write/replace failed: {e}")
            try:
                tmp_path_p.unlink()
            except FileNotFoundError:
                pass
            continue
        except Exception:
            # Non-permission failures (e.g. pickle errors) -- surface them;
            # we don't want to silently try the next location for those.
            try:
                tmp_path_p.unlink()
            except FileNotFoundError:
                pass
            raise
    raise PermissionError(
        "Could not write sidecar to any candidate location:\n" + "\n".join(errors)
    )
