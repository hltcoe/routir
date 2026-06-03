"""Random-access reader machinery used by text-jsonl views (and similar).

Houses :class:`RandomAccessReader` (the abstract O(1) lookup interface) plus the
two concrete backends shipped today: :class:`OffsetFile` for plain JSONL
collections (with a ``.offsetmap`` sidecar) and :class:`MSMARCOSegOffset` for
MSMARCO v2.1 sharded gzip corpora where IDs encode shard and byte offset
directly.

These primitives know nothing about :class:`Processor` or the request/response
shapes -- they are pure storage adapters consumed by content view backends.
"""

import io
import json
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ...utils import logger, pbar
from .sidecar import (
    atomic_write_sidecar,
    find_existing_sidecar,
    resolve_sidecar_candidates,
)


class RandomAccessReader:
    """Abstract base for O(1) document lookup by string ID.

    Subclasses wrap different on-disk formats (plain JSONL, sharded gzip) and
    expose a uniform ``reader[doc_id]`` interface that returns the raw JSON
    line for a document.

    Attributes:
        path (Path): Root path of the document store (file or directory).
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def __getitem__(self, idx: str) -> str:
        """Return the raw JSON line for document *idx*."""
        raise NotImplementedError

    def __contains__(self, idx: str) -> bool:
        """Return ``True`` if *idx* is present in this reader."""
        raise NotImplementedError


class OffsetFile(RandomAccessReader):
    """Fast random-access reader for JSONL document files using a byte-offset map.

    On first access, scans the JSONL file and builds a ``{doc_id: byte_offset}``
    mapping that is pickled to a ``.offsetmap`` sidecar file beside the corpus.
    Subsequent startups load the sidecar directly (O(1) per document lookup).

    Documents are retrieved by seeking to the stored byte offset and reading one
    line -- no need to load the entire corpus into memory.

    The sidecar location is resolved via the shared fallback chain in
    :mod:`.sidecar`: per-view ``cache_dir`` first, then adjacent to the JSONL,
    then ``${XDG_CACHE_HOME}/routir/offsetmap/``.

    Example::

        reader = OffsetFile("/data/corpus.jsonl", id_field="docid")
        line = reader["msmarco_v2_doc_00_123456"]  # raw JSON string
        doc = json.loads(line)
    """

    def __init__(
        self,
        path: Path,
        key: Callable = None,
        cache_dir: Optional[str] = None,
        id_field: str = "id",
    ):
        """
        Initialize OffsetFile and build or load the byte-offset map.

        Args:
            path (Path): Path to the JSONL corpus file.
            key (callable, optional): Legacy ``(line: str) -> doc_id`` function.
                Prefer ``id_field`` instead.  When both are provided, ``key``
                takes precedence.
            cache_dir (str, optional): Directory for the ``.offsetmap`` sidecar
                file (PR7 -- interpreted as a directory, not a single file).
                If unset, RoutIR writes adjacent to the corpus; on
                ``PermissionError`` it falls back to
                ``${XDG_CACHE_HOME:-~/.cache}/routir/offsetmap/``.  Inside the
                directory, the sidecar name is
                ``<basename>.<hash16>.offsetmap`` where the hash is the first
                16 hex chars of ``sha256(realpath(corpus))``, preventing
                collisions across multiple corpora that share one cache dir.
            id_field (str): JSON field name used as the document ID when ``key``
                is not provided (default ``"id"``).
        """
        super().__init__(path)

        self.id_field = id_field
        self.key_func = key

        candidates = resolve_sidecar_candidates(
            self.path, ".offsetmap", cache_dir
        )
        existing = find_existing_sidecar(candidates)
        if existing is None:
            logger.info(f"Building offset map for {self.path}...")
            self.create_offsetmap(self.path, candidates, key or self._default_key)
            existing = find_existing_sidecar(candidates)
            if existing is None:
                raise RuntimeError(
                    f"offsetmap disappeared immediately after build for {self.path}"
                )
        else:
            logger.info(f"Loading existing offset map from {existing}")

        try:
            loaded_fn, self.pointer_dict = pickle.load(existing.open("rb"))
            logger.info(f"Loaded offset map with {len(self.pointer_dict):,} entries")
        except Exception as e:
            logger.error(f"Failed to load offset map from {existing}: {e}")
            raise

        self.fread = self.path.open("rt")

    def _default_key(self, line: str) -> str:
        """Default key extraction using id_field."""
        data = json.loads(line)
        return str(data[self.id_field]).strip()

    def create_offsetmap(
        self,
        fn: Path,
        candidates: List[Path],
        key: Callable[[str], str],
    ):
        """Scan *fn* line-by-line and write a ``{doc_id: byte_offset}`` pickle.

        The pickle is written to the first writable entry in *candidates*
        (see :func:`atomic_write_sidecar`).  This is called once at first
        startup and may take several minutes for large corpora.  Raises if
        more than 10 consecutive lines fail to parse.
        """
        mapping = {}

        # Use larger buffer size for reading
        buffer_size = 1024 * 1024  # 1MB buffer

        with fn.open("rt", buffering=buffer_size) as fr:
            count_err = 0
            line_count = 0

            # Update progress less frequently
            update_frequency = 10000

            with pbar(desc=f"Building offset map for {fn} (sequential)") as progress:
                while True:
                    loc = fr.tell()
                    line = fr.readline()
                    if line == "":
                        break

                    try:
                        mapping[key(line).strip()] = loc
                        count_err = 0
                    except Exception as e:
                        logger.warning(f"Offset #{loc} decode error: {e}")
                        count_err += 1

                    if count_err > 10:
                        raise Exception("Too many errors")

                    line_count += 1
                    if line_count % update_frequency == 0:
                        progress.update(update_frequency)

                # Update remaining lines
                progress.update(line_count % update_frequency)

        def _writer(path: Path) -> None:
            with path.open("wb") as fw:
                pickle.dump((str(fn), mapping), fw, protocol=pickle.HIGHEST_PROTOCOL)
                fw.flush()
                os.fsync(fw.fileno())

        written = atomic_write_sidecar(candidates, _writer)
        logger.info(f"Wrote offset map to {written}")

    def __getitem__(self, idx: str):
        if idx not in self.pointer_dict:
            return {}
        self.fread.seek(self.pointer_dict[idx])
        return self.fread.readline()

    def __iter__(self):
        with self.path.open("rt") as fr: # make sure it keeps its own fp
            yield from fr

    def __contains__(self, idx: str):
        return idx in self.pointer_dict

    def __len__(self):
        return len(self.pointer_dict)

    def __del__(self):
        # Defensive: if __init__ raised before self.fread was assigned (e.g.
        # the offset-map build failed), there is nothing to close.
        fread = getattr(self, "fread", None)
        if fread is not None:
            fread.close()


class MSMARCOSegOffset(RandomAccessReader):
    """Random-access reader for MSMARCO v2.1 sharded gzip document files.

    The MSMARCO v2.1 segmented document corpus is stored as multiple gzip
    shards.  Document IDs encode both the shard number and the byte offset
    within that shard (e.g. ``msmarco_v2.1_doc_segmented_00_123456_0_789``),
    enabling direct O(1) seeks without an external offset map.

    Uses ``rapidgzip`` for parallel decompression when available, falling back
    to the standard ``gzip`` module.  Open file handles per shard are cached
    in ``cached_fps`` for the lifetime of the reader.

    Example::

        reader = MSMARCOSegOffset("/data/msmarco-v2.1-doc-segmented/")
        line = reader["msmarco_v2.1_doc_segmented_00_123_0_456"]
        doc = json.loads(line)
    """

    def __init__(
        self, path: Path, num_workers: int = 32, filename_pattern="msmarco_v2.1_doc_segmented_{shard}.json.gz", id_parser=None,
        force_load_all: bool = False
    ):
        """
        Initialize the MSMARCO segmented document reader.

        Args:
            path (Path): Directory containing the sharded ``.json.gz`` files.
            num_workers (int): Parallelism passed to ``rapidgzip`` for
                decompression (default 32).  Ignored when falling back to
                native ``gzip``.
            filename_pattern (str): Glob/format pattern for shard filenames.
                ``{shard}`` is replaced with the shard identifier extracted
                from the document ID.
            id_parser (callable, optional): ``(doc_id: str) -> (shard, offset)``
                function.  Defaults to :meth:`_parse_idx`, which handles the
                standard MSMARCO v2.1 ID format
                ``msmarco_v2.1_doc_segmented_<shard>_<…>_<…>_<offset>``.
                Override for custom ID formats.
            force_load_all (bool): When ``True``, all documents across all
                shards are loaded into memory at startup.  Trades startup time
                and memory for maximum throughput.  Default ``False`` uses
                on-demand seek-based access.
        """
        super().__init__(path)

        try:
            from rapidgzip import RapidgzipFile
            self.opener = partial(RapidgzipFile, parallelization=num_workers)
        except Exception as e:
            logger.warning(f"Failed loading rapidgzip for .gz collection. Falling back to native gzip, which is slower: {e}")
            import gzip

            self.opener = gzip.open

        self.filename_pattern = filename_pattern
        self.id_parser = id_parser if id_parser is not None else self._parse_idx

        self.cached_fps: Dict[str, io.BytesIO] = {}
        self.loaded: dict[str, str] | None = None
        if force_load_all:
            self._load_all_docs()

    def _load_all_docs(self):
        self.loaded = {}
        for fn in pbar(list(self.path.glob(self.filename_pattern.format(shard="*"))), desc='loading all docs'):
            with self.opener(fn) as fp:
                for line in pbar(fp, leave=False):
                    line = line.decode()
                    self.loaded[json.loads(line)['docid']] = line

    def _parse_idx(self, idx: str):
        """Parse a MSMARCO v2.1 segmented document ID into ``(shard, offset)``.

        The standard ID format is::

            msmarco_v2.1_doc_segmented_<shard>_<n>_<m>_<byte_offset>
            # fields after split("_"): [0..2]=prefix  [3]=shard  [4..5]=indices  [6]=offset

        Returns:
            tuple: ``(shard_str, byte_offset_int)`` used to locate the document
            in ``<path>/<filename_pattern.format(shard=shard_str)>`` at the
            given byte offset.
        """
        idx = idx.split("_")
        return idx[3], int(idx[5])

    def __getitem__(self, idx: str):
        if self.loaded is not None:
            return self.loaded[idx]

        shard, off = self.id_parser(idx)
        fn = str(self.path / self.filename_pattern.format(shard=shard))
        if fn not in self.cached_fps:
            self.cached_fps[fn] = self.opener(fn)
        fp = self.cached_fps[fn]
        fp.seek(off)
        return fp.readline().decode()

    def __contains__(self, idx: str):
        return self[idx] != ""

    def __del__(self):
        for fp in self.cached_fps.values():
           fp.close()
