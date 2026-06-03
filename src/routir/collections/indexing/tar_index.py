"""Sidecar ``.taridx`` build & verify for plain ``.tar`` shards.

The index is a pickled tuple::

    (
        {"size": int, "mtime": float, "head_sha256": str},   # stamp
        {"member_name": (offset_to_first_data_byte, size)},
    )

``offset_to_first_data_byte`` is the byte position of the file content (i.e.
``offset_data`` from :class:`tarfile.TarInfo`), pre-computed at index time so
reads do a single seek into the tar.  The stamp is recomputed at open time;
if any field mismatches, the index is rebuilt and atomically replaced.

Plain ``.tar`` only.  ``.tar.gz`` raises :class:`NotImplementedError` and
will land in PR6 via ``indexed_gzip``.

The write protocol mirrors :meth:`OffsetFile.create_offsetmap`: write to
``mkstemp`` in the same directory, fsync, then ``os.replace`` -- leaving no
partial sidecar on failure and making concurrent builds race-safe.  PR7
extends this with a fallback chain so a read-only dataset mount can still
land its sidecar in a writable location (see :mod:`.sidecar`).
"""

import hashlib
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...utils import logger
from .sidecar import (
    atomic_write_sidecar,
    find_existing_sidecar,
    resolve_sidecar_candidates,
)


def _stamp(tar_path: Path) -> Dict[str, Any]:
    """Compute the freshness stamp for a tar file."""
    st = tar_path.stat()
    with tar_path.open("rb") as fp:
        head = fp.read(256)
    return {
        "size": st.st_size,
        "mtime": st.st_mtime,
        "head_sha256": hashlib.sha256(head).hexdigest(),
    }


def _build_index(tar_path: Path) -> Dict[str, Tuple[int, int]]:
    """Scan the tar with :class:`tarfile.TarFile`, return ``{name: (data_offset, size)}``.

    Uses ``TarFile`` iteration; each member's ``offset_data`` is the start of
    the file content.  ``size`` is from the header.  Directory entries and
    zero-size members are skipped.
    """
    index: Dict[str, Tuple[int, int]] = {}
    with tarfile.open(tar_path, mode="r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if member.size <= 0:
                continue
            index[member.name] = (member.offset_data, member.size)
    return index


def build_or_load_taridx(
    tar_path: Path,
    cache_dir: Optional[str] = None,
) -> Dict[str, Tuple[int, int]]:
    """Return the member -> ``(offset, size)`` index for *tar_path*.

    Loads the ``.taridx`` sidecar if it exists and the stamp matches the tar's
    current ``(size, mtime, head_sha256)``.  Otherwise scans the tar, builds
    the index, writes it atomically (``mkstemp`` + ``os.replace``), and
    returns it.

    The sidecar is resolved via the standard fallback chain
    (:func:`resolve_sidecar_candidates`): the user-supplied ``cache_dir``
    first, then adjacent to the tar, then ``${XDG_CACHE_HOME}/routir/taridx/``.

    Plain ``.tar`` only.  ``.tar.gz`` raises :class:`NotImplementedError`.
    """
    tar_path = Path(tar_path)
    if str(tar_path).endswith(".gz"):
        raise NotImplementedError(
            f".tar.gz random access requires indexed_gzip (PR6); got {tar_path}"
        )

    candidates = resolve_sidecar_candidates(tar_path, ".taridx", cache_dir)
    current_stamp = _stamp(tar_path)

    existing = find_existing_sidecar(candidates)
    if existing is not None:
        try:
            with existing.open("rb") as fp:
                stamp, index = pickle.load(fp)
            if stamp == current_stamp:
                logger.debug(
                    f"Loaded taridx for {tar_path} from {existing} ({len(index):,} members)"
                )
                return index
            logger.info(f"Stale taridx {existing} for {tar_path}, rebuilding")
        except Exception as e:
            logger.warning(f"Failed to load taridx {existing}: {e}; rebuilding")

    logger.info(f"Building taridx for {tar_path}")
    index = _build_index(tar_path)

    def _writer(path: Path) -> None:
        with path.open("wb") as fw:
            pickle.dump((current_stamp, index), fw, protocol=pickle.HIGHEST_PROTOCOL)
            fw.flush()
            os.fsync(fw.fileno())

    written = atomic_write_sidecar(candidates, _writer)
    logger.info(f"Wrote taridx {written} ({len(index):,} members)")
    return index


def verify_or_build(tar_path: Path, cache_dir: Optional[str] = None) -> Path:
    """Same as :func:`build_or_load_taridx` but returns the sidecar path
    (useful for the CLI)."""
    build_or_load_taridx(tar_path, cache_dir=cache_dir)
    existing = find_existing_sidecar(
        resolve_sidecar_candidates(Path(tar_path), ".taridx", cache_dir)
    )
    if existing is None:
        raise RuntimeError(
            f"sidecar disappeared immediately after build for {tar_path}"
        )
    return existing


def cli_build_all(
    directory: Path,
    force: bool = False,
    cache_dir: Optional[str] = None,
) -> None:
    """Walk *directory* for ``*.tar`` files and build/verify each ``.taridx``.

    When ``force=True``, delete any existing sidecar (at any candidate
    location) first so the index is rebuilt unconditionally and no stale
    lower-priority copies are left behind.
    """
    directory = Path(directory)
    for tar_path in sorted(directory.rglob("*.tar")):
        if force:
            for c in resolve_sidecar_candidates(tar_path, ".taridx", cache_dir):
                if c.exists():
                    try:
                        c.unlink()
                    except (PermissionError, OSError):
                        logger.warning(f"Could not delete stale sidecar {c}")
        build_or_load_taridx(tar_path, cache_dir=cache_dir)


def main():
    """CLI: build .taridx sidecars for every .tar under a directory."""
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m routir.collections.indexing.tar_index",
        description="Build .taridx sidecar indexes for plain .tar shards.",
    )
    p.add_argument("directory", type=Path, help="Directory to scan for .tar files (recursive).")
    p.add_argument("--force", action="store_true",
                   help="Delete existing .taridx sidecars and rebuild from scratch.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-file progress logs.")
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for sidecars when the dataset dir is read-only "
             "(falls back to ~/.cache/routir/taridx/ if also unset/unwritable).",
    )
    args = p.parse_args()

    if args.quiet:
        import logging
        logging.getLogger("search-service").setLevel(logging.WARNING)

    cli_build_all(args.directory, force=args.force, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
