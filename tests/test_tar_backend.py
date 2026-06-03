"""Tests for the PR5b bytes view: ``TarSource`` + ``TarBackend``.

Covers:

* Single-tar, single-match & multi-match (sorted) reads via ``GlobMatcher``.
* Zero-match returns ``data=[]`` (not an error).
* Anchor enforcement for both glob and regex matchers (``id="abc"`` must not
  bleed into ``abcd_*.jpg``).
* All three shard resolvers (manifest, modulo, substring).
* Stale-index detection and rebuild after tar mutation.
* Race-safe atomic build (no ``*.taridx.tmp.*`` on failure).
* ``.tar.gz`` rejection with a clear PR6 message.
* End-to-end :class:`SearchPipeline` with TarSource view (multivent-style
  config: chunk ids, manifest, multi-match glob).
* Hermetic per-test isolation of module-level caches (incl. closing fds).
"""

import asyncio
import io
import os
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from routir.collections.indexing import tar_index as _tarmod
from routir.collections.processor import ContentProcessor
from routir.collections.views import TarBackend
from routir.collections.views import tar as _tar_backend
from routir.config import (
    CollectionConfig,
    GlobMatcher,
    RegexMatcher,
    ShardManifest,
    ShardModulo,
    ShardSubstring,
    TarSource,
    ViewSpec,
)
from routir.pipeline import SearchPipeline
from routir.processors.abstract import Processor
from routir.processors.registry import ProcessorRegistry


# ----------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _clear_tar_caches():
    """Per-test isolation: clear module-level caches and close all cached fds.

    Critical for tar tests: ``_FD_CACHE`` holds raw OS file descriptors that
    leak across tests if not closed.  Also clears the singleton manifest and
    index caches so each test starts from a clean slate.
    """
    def _close_fds():
        for fd in _tar_backend._FD_CACHE.values():
            try:
                os.close(fd)
            except OSError:
                pass
        _tar_backend._FD_CACHE.clear()
        _tar_backend._INDEX_CACHE.clear()
        _tar_backend._INDEX_KEYS_CACHE.clear()
        _tar_backend._MANIFEST_CACHE.clear()

    _close_fds()
    yield
    _close_fds()


# ----------------------------------------------------------------- helpers


def _make_tar(path: Path, members: Dict[str, bytes]) -> None:
    """Create a plain (uncompressed) tar at *path* with the given members."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_backend(spec: TarSource, name: str = "kf") -> TarBackend:
    cfg = CollectionConfig(
        name="dummy",
        views={name: ViewSpec(kind="bytes", source=spec)},
    )
    return TarBackend(name=name, spec=spec, collection_config=cfg)


# ----------------------------------------------------------------- single-tar single match


def test_single_tar_single_match(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"foo.jpg": b"\x89PNG\x00FOO-bytes"})
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=GlobMatcher(pattern="{id}.jpg"),
        mime="image/jpeg",
    )
    backend = _make_backend(spec)
    out = backend["foo"]
    assert out["data"] == [b"\x89PNG\x00FOO-bytes"]
    assert out["mime"] == "image/jpeg"


# ----------------------------------------------------------------- multi-match sorted ordering


def test_multi_match_chronological_ordering(tmp_path):
    tar_path = tmp_path / "x.tar"
    # Add in a deliberately scrambled order; result must be sorted (chronological for
    # the multivent layout because filenames embed time indices).
    _make_tar(tar_path, {
        "foo_t000010.jpg": b"ten",
        "foo_t000000.jpg": b"zero",
        "foo_t000005.jpg": b"five",
    })
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=GlobMatcher(pattern="{id}_t*.jpg"),
    )
    backend = _make_backend(spec)
    out = backend["foo"]
    assert out["data"] == [b"zero", b"five", b"ten"]


# ----------------------------------------------------------------- zero match


def test_zero_match_returns_empty_data(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"other_t0.jpg": b"unused"})
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=GlobMatcher(pattern="{id}_t*.jpg"),
    )
    backend = _make_backend(spec)
    out = backend["foo"]
    assert out["data"] == []
    # __contains__ goes through the matcher; nothing matched.
    assert "foo" not in backend


# ----------------------------------------------------------------- anchor enforcement


def test_glob_matcher_anchored(tmp_path):
    """``id='abc'`` against pattern ``'{id}_t*.jpg'`` must NOT match ``'abcd_t0.jpg'``."""
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {
        "abcd_t0.jpg": b"longer-id",
        "abc_t0.jpg": b"exact-match",
    })
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=GlobMatcher(pattern="{id}_t*.jpg"),
    )
    backend = _make_backend(spec)
    out = backend["abc"]
    assert out["data"] == [b"exact-match"]
    # The 'abcd' member must be excluded.
    assert b"longer-id" not in out["data"]


def test_regex_matcher_anchored(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {
        "abcd_t0.jpg": b"longer-id",
        "abc_t0.jpg": b"exact-match",
    })
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=RegexMatcher(pattern=r"{id}_t\d+\.jpg"),
    )
    backend = _make_backend(spec)
    out = backend["abc"]
    assert out["data"] == [b"exact-match"]


# ----------------------------------------------------------------- shard manifest


def test_shard_manifest(tmp_path):
    _make_tar(tmp_path / "shard_0.tar", {"a.jpg": b"AAA"})
    _make_tar(tmp_path / "shard_1.tar", {"b.jpg": b"BBB"})
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,shard\na,0\nb,1\n")
    spec = TarSource(
        source="tar",
        tar_template=str(tmp_path / "shard_{shard}.tar"),
        shard_resolver=ShardManifest(
            kind="manifest", path=str(manifest), id_column="id", shard_column="shard",
        ),
        matcher=GlobMatcher(pattern="{id}.jpg"),
    )
    backend = _make_backend(spec)
    assert backend["a"]["data"] == [b"AAA"]
    assert backend["b"]["data"] == [b"BBB"]


def test_shard_manifest_int_format_spec(tmp_path):
    """Manifest values that parse as int must support ``{shard:06d}`` format specs."""
    _make_tar(tmp_path / "shard_000000.tar", {"a.jpg": b"AAA"})
    _make_tar(tmp_path / "shard_000001.tar", {"b.jpg": b"BBB"})
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,shard\na,0\nb,1\n")
    spec = TarSource(
        source="tar",
        tar_template=str(tmp_path / "shard_{shard:06d}.tar"),
        shard_resolver=ShardManifest(
            kind="manifest", path=str(manifest), id_column="id", shard_column="shard",
        ),
        matcher=GlobMatcher(pattern="{id}.jpg"),
    )
    backend = _make_backend(spec)
    assert backend["a"]["data"] == [b"AAA"]
    assert backend["b"]["data"] == [b"BBB"]


# ----------------------------------------------------------------- shard modulo


def test_shard_modulo(tmp_path):
    # Build 4 tars; place each id's file in its deterministic shard.
    import hashlib
    n = 4

    def _shard_for(doc_id: str) -> int:
        return int.from_bytes(hashlib.sha256(doc_id.encode()).digest()[:8], "big") % n

    ids_payloads = {f"doc_{i}": f"data_{i}".encode() for i in range(8)}
    by_shard: Dict[int, Dict[str, bytes]] = {i: {} for i in range(n)}
    for doc_id, payload in ids_payloads.items():
        by_shard[_shard_for(doc_id)][f"{doc_id}.bin"] = payload
    for s, members in by_shard.items():
        # Each tar must exist even if empty — but tar's open won't write an empty
        # tar with zero entries; only create those with members.
        if members:
            _make_tar(tmp_path / f"shard_{s:04d}.tar", members)
        else:
            _make_tar(tmp_path / f"shard_{s:04d}.tar", {"_placeholder": b"x"})
    spec = TarSource(
        source="tar",
        tar_template=str(tmp_path / "shard_{shard:04d}.tar"),
        shard_resolver=ShardModulo(kind="modulo", n=n, width=4),
        matcher=GlobMatcher(pattern="{id}.bin"),
    )
    backend = _make_backend(spec)
    for doc_id, payload in ids_payloads.items():
        out = backend[doc_id]
        assert out["data"] == [payload], f"mismatch for {doc_id}"


# ----------------------------------------------------------------- shard substring


def test_shard_substring(tmp_path):
    _make_tar(tmp_path / "prefix_AB.tar", {"AB_001.bin": b"alpha"})
    _make_tar(tmp_path / "prefix_CD.tar", {"CD_002.bin": b"beta"})
    spec = TarSource(
        source="tar",
        tar_template=str(tmp_path / "prefix_{shard}.tar"),
        shard_resolver=ShardSubstring(kind="substring", start=0, end=2),
        matcher=GlobMatcher(pattern="{id}.bin"),
    )
    backend = _make_backend(spec)
    assert backend["AB_001"]["data"] == [b"alpha"]
    assert backend["CD_002"]["data"] == [b"beta"]


# ----------------------------------------------------------------- stale stamp


def test_stale_index_rebuild(tmp_path):
    """Mutate the tar after building the index; the next open must detect the
    stale stamp, rebuild, and surface the new member."""
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"foo.jpg": b"FOO"})
    spec = TarSource(
        source="tar",
        tar_template=str(tar_path),
        matcher=GlobMatcher(pattern="{id}.jpg"),
    )
    backend = _make_backend(spec)
    # Touch the backend so the index is built.
    assert backend["foo"]["data"] == [b"FOO"]
    sidecar = tar_path.parent / (tar_path.name + ".taridx")
    assert sidecar.exists()

    # Clear backend-level caches so the next access re-reads the sidecar.
    _tar_backend._INDEX_CACHE.clear()
    _tar_backend._INDEX_KEYS_CACHE.clear()
    for fd in _tar_backend._FD_CACHE.values():
        os.close(fd)
    _tar_backend._FD_CACHE.clear()

    # Append a new member.  Use 'a' append-mode on the plain tar.
    with tarfile.open(tar_path, mode="a") as tf:
        info = tarfile.TarInfo(name="bar.jpg")
        data = b"BAR-bytes"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    # Bump mtime to ensure stamp differs even on coarse FS timestamps.
    new_mtime = tar_path.stat().st_mtime + 5
    os.utime(tar_path, (new_mtime, new_mtime))

    # New backend instance forces a fresh _ensure_open call; the sidecar's
    # stamp won't match the tar -> rebuild.
    backend2 = _make_backend(spec)
    out = backend2["bar"]
    assert out["data"] == [b"BAR-bytes"]
    # And the old member still works.
    out_foo = backend2["foo"]
    assert out_foo["data"] == [b"FOO"]


# ----------------------------------------------------------------- race-safe build


def test_race_safe_build_no_tmp_leftover(tmp_path):
    """If pickle.dump fails during atomic write, no .taridx.tmp.* survives."""
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"foo.jpg": b"FOO"})

    with patch.object(_tarmod.pickle, "dump", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _tarmod.build_or_load_taridx(tar_path)

    # No tmp files remain.
    leftovers = list(tmp_path.glob("x.tar.taridx*"))
    assert leftovers == [], f"leftover sidecar/tmp files: {leftovers}"


# ----------------------------------------------------------------- .tar.gz rejection


def test_tar_gz_rejected(tmp_path):
    """Plain .tar only in PR5b; .tar.gz raises NotImplementedError."""
    gz_path = tmp_path / "x.tar.gz"
    gz_path.write_bytes(b"not-a-real-gzip-but-extension-is-what-matters")
    spec = TarSource(
        source="tar",
        tar_template=str(gz_path),
        matcher=GlobMatcher(pattern="{id}.jpg"),
    )
    backend = _make_backend(spec)
    with pytest.raises(NotImplementedError, match="PR6"):
        backend["foo"]


# ----------------------------------------------------------------- cli_build_all helper


def test_cli_build_all_walks_directory(tmp_path):
    _make_tar(tmp_path / "a.tar", {"x.bin": b"X"})
    _make_tar(tmp_path / "sub" / "b.tar", {"y.bin": b"Y"})
    _tarmod.cli_build_all(tmp_path)
    assert (tmp_path / "a.tar.taridx").exists()
    assert (tmp_path / "sub" / "b.tar.taridx").exists()


def test_cli_build_all_force_rebuilds(tmp_path):
    tar_path = tmp_path / "a.tar"
    _make_tar(tar_path, {"x.bin": b"X"})
    _tarmod.cli_build_all(tmp_path)
    sidecar = tmp_path / "a.tar.taridx"
    mtime_first = sidecar.stat().st_mtime
    # Move sidecar mtime back so the unconditional rebuild is detectable.
    os.utime(sidecar, (mtime_first - 100, mtime_first - 100))
    _tarmod.cli_build_all(tmp_path, force=True)
    assert sidecar.stat().st_mtime > mtime_first - 100


# ----------------------------------------------------------------- ContentProcessor dispatch


async def test_content_processor_dispatches_tar_view(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"a.jpg": b"\x89PNG\x00A"})
    cfg = CollectionConfig(
        name="tarcoll",
        default_view="kf",
        views={
            "kf": ViewSpec(
                kind="bytes",
                source=TarSource(
                    source="tar",
                    tar_template=str(tar_path),
                    matcher=GlobMatcher(pattern="{id}.jpg"),
                    mime="image/jpeg",
                ),
            ),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)
    payload = await cp._submit({"id": "a", "view": "kf"})
    assert payload["data"] == [b"\x89PNG\x00A"]
    assert payload["view"] == "kf"
    assert payload["kind"] == "bytes"
    assert payload["mime"] == "image/jpeg"


# ----------------------------------------------------------------- end-to-end pipeline


class _StubSearch(Processor):
    def __init__(self, scores):
        super().__init__(cache_size=-1)
        self._scores = scores

    async def _submit(self, item):
        return {"scores": dict(self._scores)}


class _RecordingBytesReranker(Processor):
    def __init__(self):
        super().__init__(cache_size=-1)
        self.received_passages = None

    async def _submit(self, item):
        self.received_passages = item.get("passages")
        n = len(self.received_passages or [])
        return {"scores": [float(i) for i in range(n)]}


async def test_end_to_end_pipeline_multivent_style(tmp_path):
    """Multivent-shaped config: chunk ids, shard manifest, multi-match glob.

    Two chunks, each with N frames inside their respective tar shard; the
    bytes reranker must receive ``List[List[bytes]]`` (one inner list per
    chunk, each entry one frame jpeg).
    """
    # Shards laid out like the multivent uniform_5s set.
    _make_tar(tmp_path / "shard_000000.tar", {
        "chunk_001.kf_uni5s.t000000.jpg": b"\x89PNG\x00C1F0",
        "chunk_001.kf_uni5s.t000005.jpg": b"\x89PNG\x00C1F1",
    })
    _make_tar(tmp_path / "shard_000001.tar", {
        "chunk_002.kf_uni5s.t000000.jpg": b"\x89PNG\x00C2F0",
    })
    manifest = tmp_path / "catalog.csv"
    manifest.write_text("chunk_id,shard_index\nchunk_001,0\nchunk_002,1\n")

    cfg = CollectionConfig(
        name="mvcoll",
        default_view="kf",
        views={
            "kf": ViewSpec(
                kind="bytes",
                source=TarSource(
                    source="tar",
                    tar_template=str(tmp_path / "shard_{shard:06d}.tar"),
                    shard_resolver=ShardManifest(
                        kind="manifest",
                        path=str(manifest),
                        id_column="chunk_id",
                        shard_column="shard_index",
                    ),
                    matcher=GlobMatcher(pattern="{id}.kf_uni5s.t*.jpg"),
                    mime="image/jpeg",
                ),
            ),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)

    bm25 = _StubSearch({"chunk_001": 1.0, "chunk_002": 0.5})
    rr = _RecordingBytesReranker()

    ProcessorRegistry.register("bm25", "search", bm25, view_kind="text")
    ProcessorRegistry.register("bytes-rr", "search", _StubSearch({"chunk_001": 1.0}), view_kind="text")
    ProcessorRegistry.register("bytes-rr", "score", rr, view_kind="bytes")
    ProcessorRegistry.register(
        "mvcoll", "content", cp,
        views={"kf": "bytes"}, default_view="kf",
    )
    try:
        pipeline = SearchPipeline.from_string(
            "bm25%100 >> bytes-rr@kf%10", collection="mvcoll",
        )
        result = await pipeline.run("query")
        passages = rr.received_passages
        assert passages is not None
        assert len(passages) == 2  # two chunks
        # Flatten and check we got the right bytes.
        flat = [b for inner in passages for b in inner]
        assert b"\x89PNG\x00C1F0" in flat
        assert b"\x89PNG\x00C1F1" in flat
        assert b"\x89PNG\x00C2F0" in flat
        assert set(result["scores"].keys()) == {"chunk_001", "chunk_002"}
    finally:
        for name in ("bm25", "bytes-rr", "mvcoll"):
            ProcessorRegistry.all_services.pop(name, None)
            ProcessorRegistry.slot_meta.pop(name, None)


# ----------------------------------------------------------------- ViewSpec validation


def test_view_spec_rejects_text_kind_with_tar_source(tmp_path):
    with pytest.raises(ValueError, match="TarSource is only valid for kind='bytes'"):
        ViewSpec(
            kind="text",
            source=TarSource(
                source="tar",
                tar_template=str(tmp_path / "x.tar"),
                matcher=GlobMatcher(pattern="{id}.jpg"),
            ),
        )
