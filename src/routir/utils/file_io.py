import io
import json
import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Dict

from . import logger, pbar


class RandomAccessReader:
    def __init__(self, path: Path):
        self.path = Path(path)

    def __getitem__(self, idx: str) -> str:
        raise NotImplementedError

    def __contains__(self, idx: str) -> bool:
        raise NotImplementedError


class OffsetFile(RandomAccessReader):
    def __init__(self, path: Path, key: Callable = None, offset_fn: Path = None, id_field: str = "id"):
        """
        Initialize OffsetFile.

        Args:
            fn: Path to the document file
            key: Callable to extract key from line (legacy support)
            offset_fn: Path to store the offset map
            num_workers: Number of parallel workers
            id_field: Field name to use as key (preferred over key function)
        """
        super().__init__(path)

        if offset_fn is None:
            offset_fn = self.path.parent / (self.path.name + ".offsetmap")
        else:
            offset_fn = Path(offset_fn)

        # Prefer id_field over key function for parallel processing
        self.id_field = id_field
        self.key_func = key

        if not offset_fn.exists():
            logger.info(f"Building offset map for {self.path}...")
            self.create_offsetmap(self.path, offset_fn, key or self._default_key)
        else:
            logger.info(f"Loading existing offset map from {offset_fn}")

        try:
            loaded_fn, self.pointer_dict = pickle.load(offset_fn.open("rb"))
            logger.info(f"Loaded offset map with {len(self.pointer_dict):,} entries")
        except Exception as e:
            logger.error(f"Failed to load offset map from {offset_fn}: {e}")
            raise

        self.fread = self.path.open("rt")

    def _default_key(self, line: str) -> str:
        """Default key extraction using id_field."""
        data = json.loads(line)
        return str(data[self.id_field]).strip()

    def create_offsetmap(self, fn: Path, offset_fn: Path, key: Callable[[str], str]):
        """Fallback sequential implementation with optimizations."""
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

        logger.info(f"Writing offset map to {offset_fn}")
        with open(offset_fn, "wb") as fw:
            pickle.dump((str(fn), mapping), fw, protocol=pickle.HIGHEST_PROTOCOL)
            fw.flush()

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
        self.fread.close()


class MSMARCOSegOffset(RandomAccessReader):
    def __init__(
        self, path: Path, num_workers: int = 8, filename_pattern="msmarco_v2.1_doc_segmented_{shard}.json.gz", id_parser=None
    ):
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

    def _parse_idx(self, idx: str):
        idx = idx.split("_")
        return idx[3], int(idx[5])

    def __getitem__(self, idx: str):
        shard, off = self.id_parser(idx)
        fn = str(self.path / self.filename_pattern.format(shard=shard))
        if fn not in self.cached_fps:
            self.cached_fps[fn] = self.opener(fn)
        fp = self.cached_fps[fn]
        fp.seek(off)
        return fp.readline().decode()

    def __contains__(self, idx: str):
        return self[idx] != ""
