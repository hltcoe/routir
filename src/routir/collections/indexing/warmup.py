"""Pre-build all view-backend sidecars from a RoutIR config.

Usage::

    python -m routir.collections.indexing.warmup config.json
    python -m routir.collections.indexing.warmup config.json --quiet
    python -m routir.collections.indexing.warmup config.json --force
    python -m routir.collections.indexing.warmup config.json --workers 16

For each collection in the config:
  * Every ``TextJsonlSource`` triggers an :class:`OffsetFile` build
    (load via the module-level reader cache, same path the server uses).
    Always serial — each text view is a single file.
  * Every ``TarSource`` enumerates concrete shards via filesystem glob
    (replacing ``{shard}`` with ``*`` in the template) and calls
    :func:`build_or_load_taridx` for each.  Multi-shard work parallelises
    cleanly via ``--workers`` — each shard build is independent (different
    source file, different sidecar destination).

Sidecars land per the standard fallback chain: per-view ``cache_dir``
if set, else adjacent to the source, else
``${XDG_CACHE_HOME:-~/.cache}/routir/...``.

Run this once after writing a config (or after the source data changes)
so the first server request doesn't pay the build cost.
"""

import argparse
import multiprocessing
import re
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

from ...config.config import Config, TarSource, TextJsonlSource
from ...utils import logger
from ..views import text_jsonl as _tj
from .sidecar import resolve_sidecar_candidates
from .tar_index import build_or_load_taridx


_TarTask = Tuple[str, Optional[str]]  # (tar_path, cache_dir)


def _build_one_taridx(task: _TarTask) -> Tuple[str, bool, Optional[str]]:
    """Worker-process entry point.  Builds one shard's taridx and returns a
    ``(tar_path, ok, error_msg)`` tuple.  Exceptions are caught so a single
    bad shard doesn't kill the whole pool."""
    tar_path, cache_dir = task
    try:
        build_or_load_taridx(Path(tar_path), cache_dir=cache_dir)
        return (tar_path, True, None)
    except Exception as e:
        return (tar_path, False, f"{type(e).__name__}: {e}")


def _clear_sidecars(source_paths: List[str], suffix: str, cache_dir: Optional[str]) -> None:
    for sp in source_paths:
        for c in resolve_sidecar_candidates(Path(sp), suffix, cache_dir):
            if c.exists():
                try:
                    c.unlink()
                except (PermissionError, OSError):
                    logger.warning(f"Could not delete {c}")


def warmup_config(
    config: Config,
    force: bool = False,
    workers: int = 1,
    only_view: Optional[str] = None,
) -> None:
    """Walk every view in every collection and ensure sidecars exist.

    Args:
        config: Parsed Config object.
        force: When True, clear existing sidecars at every candidate location
            before rebuilding.
        workers: Process-pool size for taridx builds.  ``1`` runs serially;
            higher values parallelise across shards.  Text-jsonl offsetmap
            builds stay serial (one file per view).
        only_view: When set, restrict warmup to the named view (matches
            ``view_name`` in any collection).  Useful for running one slurm
            job per view with internal multiprocess fan-out across that
            view's shards.
    """
    text_view_count = 0
    tar_tasks: List[_TarTask] = []

    # First pass: text views run serially (one file each, in-process so the
    # _READER_CACHE singleton is shared with later imports).
    for coll in config.collections:
        if not coll.views:
            continue
        logger.info(f"Warming collection {coll.name!r} ({len(coll.views)} views)")
        for view_name, view_spec in coll.views.items():
            if only_view is not None and view_name != only_view:
                continue
            src = view_spec.source
            if isinstance(src, TextJsonlSource):
                if force:
                    _tj._READER_CACHE.pop((str(src.doc_path), src.id_field), None)
                    _clear_sidecars([src.doc_path], ".offsetmap", src.cache_dir)
                _tj._get_or_build_reader(src, coll)
                text_view_count += 1
                logger.info(f"  {view_name}: offsetmap ready for {src.doc_path}")
            elif isinstance(src, TarSource):
                glob_pat = re.sub(r"\{shard(?::[^}]*)?\}", "*", src.tar_template)
                shards = sorted(glob(glob_pat))
                if not shards:
                    logger.warning(
                        f"  {view_name}: no shards matched {glob_pat!r}; skipping"
                    )
                    continue
                if force:
                    _clear_sidecars(shards, ".taridx", src.cache_dir)
                tar_tasks.extend((s, src.cache_dir) for s in shards)
                logger.info(f"  {view_name}: queued {len(shards)} shard(s)")
            else:
                logger.debug(
                    f"  {view_name}: source type {type(src).__name__} has no "
                    "sidecar; skipping"
                )

    # Second pass: tar shards in parallel.
    n_taridx_ok = 0
    n_taridx_err = 0
    if tar_tasks:
        logger.info(
            f"Building {len(tar_tasks)} taridx sidecar(s) with workers={workers}"
        )
        if workers <= 1:
            results = (_build_one_taridx(t) for t in tar_tasks)
        else:
            # imap_unordered streams results as they finish so progress logs
            # interleave; chunksize=1 keeps work balanced when individual
            # shards vary in size.
            pool = multiprocessing.Pool(processes=workers)
            try:
                results = pool.imap_unordered(_build_one_taridx, tar_tasks, chunksize=1)
                results = list(results)  # materialise inside the pool context
            finally:
                pool.close()
                pool.join()
        # Common consumption loop (works for generator or list).
        for tar_path, ok, err in results:
            if ok:
                n_taridx_ok += 1
                # Per-shard progress at INFO is verbose for 4k+ shards; log every 50.
                if n_taridx_ok % 50 == 0 or n_taridx_ok == len(tar_tasks):
                    logger.info(
                        f"  taridx progress: {n_taridx_ok + n_taridx_err}/{len(tar_tasks)} "
                        f"({n_taridx_err} errors)"
                    )
            else:
                n_taridx_err += 1
                logger.error(f"  taridx FAILED for {tar_path}: {err}")

    logger.info(
        f"Warmup complete: {text_view_count} offsetmap(s), {n_taridx_ok} taridx sidecar(s)"
        + (f", {n_taridx_err} error(s)" if n_taridx_err else "")
    )
    if n_taridx_err:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(
        prog="python -m routir.collections.indexing.warmup",
        description="Pre-build all view-backend sidecars (offsetmaps + taridx) "
                    "from a RoutIR config.",
    )
    p.add_argument("config", type=Path, help="Path to RoutIR JSON config.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every sidecar from scratch (clears existing first).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-view progress logs.")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool size for taridx builds (default 1; text-jsonl stays serial).",
    )
    p.add_argument(
        "--view",
        type=str,
        default=None,
        help="Restrict warmup to one view name (matches across all collections). "
             "Lets you run one slurm job per view with internal fan-out.",
    )
    args = p.parse_args()

    if args.quiet:
        import logging
        logging.getLogger("search-service").setLevel(logging.WARNING)

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = Config.model_validate_json(args.config.read_text())
    warmup_config(cfg, force=args.force, workers=args.workers, only_view=args.view)


if __name__ == "__main__":
    main()
