"""PR7: shared sidecar fallback chain for ``.taridx`` + ``.offsetmap``.

Resolution priority (load and write):

  1. User-supplied ``cache_dir`` (per-view).
  2. Adjacent to the source file.
  3. ``${XDG_CACHE_HOME:-~/.cache}/routir/<suffix>/``.

These tests sandbox ``XDG_CACHE_HOME`` (and ``HOME``) to ``tmp_path`` so a
fallback never pollutes the developer's real cache.
"""

import io
import json
import os
import pickle
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from routir.collections.indexing import tar_index as _tarmod
from routir.collections.indexing.offset_file import OffsetFile
from routir.collections.indexing.sidecar import (
    _hash16,
    _path_as_folder,
    find_existing_sidecar,
    resolve_sidecar_candidates,
)
from routir.collections.indexing.tar_index import build_or_load_taridx
from routir.collections.views import tar as _tar_backend
from routir.collections.views import text_jsonl as _tj_mod


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


# ----------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _sandbox_xdg(monkeypatch, tmp_path):
    """Pin XDG_CACHE_HOME (and HOME) at tmp_path so a fallback can't leak."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture(autouse=True)
def _clear_module_caches():
    _tj_mod._READER_CACHE.clear()

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
    _tj_mod._READER_CACHE.clear()


# ----------------------------------------------------------------- helpers


def _make_tar(path: Path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _write_jsonl(path: Path, docs):
    with path.open("w") as fw:
        for d in docs:
            fw.write(json.dumps(d) + "\n")


def _read_only(d: Path):
    """Make *d* read-only (0o500: r-x for owner)."""
    os.chmod(d, 0o500)


def _writable(d: Path):
    os.chmod(d, 0o755)


# ----------------------------------------------------------------- candidate order


def test_resolve_candidates_with_cache_dir(tmp_path):
    source = tmp_path / "data" / "shard.tar"
    source.parent.mkdir()
    source.touch()
    cache_dir = tmp_path / "cache"

    cands = resolve_sidecar_candidates(source, ".taridx", str(cache_dir))
    assert len(cands) == 3
    h = _hash16(source)
    folder = _path_as_folder(source)
    assert cands[0] == cache_dir / folder / f"shard.tar.{h}.taridx"
    assert cands[1] == source.parent / "shard.tar.taridx"
    # XDG_CACHE_HOME is sandboxed by the autouse fixture.
    assert cands[2] == Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx" / f"shard.tar.{h}.taridx"


def test_resolve_candidates_no_cache_dir(tmp_path):
    source = tmp_path / "data" / "shard.tar"
    source.parent.mkdir()
    source.touch()
    cands = resolve_sidecar_candidates(source, ".taridx", None)
    assert len(cands) == 2
    h = _hash16(source)
    assert cands[0] == source.parent / "shard.tar.taridx"
    assert cands[1] == Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx" / f"shard.tar.{h}.taridx"


# ----------------------------------------------------------------- taridx adjacent


def test_taridx_adjacent_when_writable(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"foo.bin": b"FOO"})
    build_or_load_taridx(tar_path)
    assert (tmp_path / "x.tar.taridx").exists()
    # XDG fallback must NOT have been used.
    xdg = Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx"
    assert not xdg.exists() or list(xdg.glob("*")) == []


# ----------------------------------------------------------------- taridx user cache_dir


def test_taridx_user_cache_dir_wins(tmp_path):
    tar_path = tmp_path / "x.tar"
    _make_tar(tar_path, {"foo.bin": b"FOO"})
    cache_dir = tmp_path / "sidecars"
    build_or_load_taridx(tar_path, cache_dir=str(cache_dir))
    h = _hash16(tar_path)
    folder = _path_as_folder(tar_path)
    expected = cache_dir / folder / f"x.tar.{h}.taridx"
    assert expected.exists()
    # Adjacent must NOT exist.
    assert not (tmp_path / "x.tar.taridx").exists()


# ----------------------------------------------------------------- taridx read-only fallback


def test_taridx_read_only_falls_back_to_xdg(tmp_path):
    """Read-only dataset dir -> sidecar lands in XDG cache."""
    ds = tmp_path / "ds"
    ds.mkdir()
    tar_path = ds / "x.tar"
    _make_tar(tar_path, {"foo.bin": b"FOO"})

    _read_only(ds)
    try:
        build_or_load_taridx(tar_path)
    finally:
        # Restore so pytest tmp_path cleanup can rm-rf the tree.
        _writable(ds)

    h = _hash16(tar_path)
    xdg_sidecar = (
        Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx" / f"x.tar.{h}.taridx"
    )
    assert xdg_sidecar.exists()
    # Adjacent must not exist (the dir was read-only).
    assert not (ds / "x.tar.taridx").exists()


# ----------------------------------------------------------------- taridx existing lower-priority is found


def test_taridx_existing_in_xdg_is_loaded(tmp_path):
    """A valid sidecar at the lower-priority XDG location must be picked up
    without rebuilding."""
    ds = tmp_path / "ds"
    ds.mkdir()
    tar_path = ds / "x.tar"
    _make_tar(tar_path, {"foo.bin": b"FOO"})

    # First, build the sidecar via the read-only path so it goes to XDG.
    _read_only(ds)
    try:
        build_or_load_taridx(tar_path)
    finally:
        _writable(ds)

    h = _hash16(tar_path)
    xdg_sidecar = (
        Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx" / f"x.tar.{h}.taridx"
    )
    assert xdg_sidecar.exists()

    # Now call again -- _build_index must NOT be invoked.
    with patch.object(_tarmod, "_build_index", side_effect=AssertionError("must not rebuild")):
        idx = build_or_load_taridx(tar_path)
    assert "foo.bin" in idx


# ----------------------------------------------------------------- offsetmap parity


def test_offsetmap_adjacent_when_writable(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}])
    reader = OffsetFile(p, id_field="id")
    assert "a" in reader
    assert (tmp_path / "corpus.jsonl.offsetmap").exists()


def test_offsetmap_user_cache_dir_wins(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{"id": "a", "text": "A"}])
    cache_dir = tmp_path / "sidecars"
    reader = OffsetFile(p, cache_dir=str(cache_dir), id_field="id")
    assert "a" in reader
    h = _hash16(p)
    folder = _path_as_folder(p)
    assert (cache_dir / folder / f"corpus.jsonl.{h}.offsetmap").exists()
    assert not (tmp_path / "corpus.jsonl.offsetmap").exists()


def test_offsetmap_read_only_falls_back_to_xdg(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    p = ds / "corpus.jsonl"
    _write_jsonl(p, [{"id": "a", "text": "A"}])

    _read_only(ds)
    try:
        reader = OffsetFile(p, id_field="id")
        assert "a" in reader
    finally:
        _writable(ds)

    h = _hash16(p)
    xdg_sidecar = (
        Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "offsetmap" / f"corpus.jsonl.{h}.offsetmap"
    )
    assert xdg_sidecar.exists()
    assert not (ds / "corpus.jsonl.offsetmap").exists()


# ----------------------------------------------------------------- no tmp leftover under read-only fallback


def test_no_tmp_leak_under_readonly_fallback(tmp_path):
    """When the adjacent write fails and the XDG fallback succeeds, no
    ``*.tmp.*`` files should survive in either location."""
    ds = tmp_path / "ds"
    ds.mkdir()
    tar_path = ds / "x.tar"
    _make_tar(tar_path, {"foo.bin": b"FOO"})

    _read_only(ds)
    try:
        build_or_load_taridx(tar_path)
    finally:
        _writable(ds)

    # Adjacent dir: no taridx, no tmp.
    assert list(ds.glob("x.tar.taridx*")) == []
    # XDG location: real sidecar, no tmp.
    xdg = Path(os.environ["XDG_CACHE_HOME"]) / "routir" / "taridx"
    files = list(xdg.glob("*"))
    assert all(".tmp." not in f.name for f in files), files


# ----------------------------------------------------------------- warmup script


def test_warmup_builds_both_sidecars(tmp_path):
    """End-to-end: build a tiny config (1 TarSource + 1 TextJsonlSource) and
    run the warmup script via ``python -m`` against it.  Both sidecars must
    exist afterwards (and the subprocess must exit zero)."""
    # Tar shard
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    tar0 = shard_dir / "shard_000.tar"
    _make_tar(tar0, {"foo.bin": b"FOO"})

    # JSONL corpus
    jsonl_dir = tmp_path / "text"
    jsonl_dir.mkdir()
    corpus = jsonl_dir / "corpus.jsonl"
    _write_jsonl(corpus, [{"id": "x", "text": "X"}])

    cfg = {
        "services": [],
        "collections": [{
            "name": "wm",
            "default_view": "txt",
            "views": {
                "txt": {
                    "kind": "text",
                    "source": {
                        "source": "text_jsonl",
                        "doc_path": str(corpus),
                        "id_field": "id",
                        "content_fields": "text",
                    },
                },
                "kf": {
                    "kind": "bytes",
                    "source": {
                        "source": "tar",
                        "tar_template": str(shard_dir / "shard_{shard:03d}.tar"),
                        "matcher": {"kind": "glob", "pattern": "{id}.bin"},
                    },
                },
            },
        }],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    # XDG_CACHE_HOME / HOME are already sandboxed by the autouse fixture.
    proc = subprocess.run(
        [sys.executable, "-m", "routir.collections.indexing.warmup",
         str(cfg_path), "--quiet"],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"warmup failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert (jsonl_dir / "corpus.jsonl.offsetmap").exists()
    assert (shard_dir / "shard_000.tar.taridx").exists()


def test_warmup_force_rebuilds(tmp_path):
    """``--force`` should rebuild even when sidecars already exist."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    tar0 = shard_dir / "shard_000.tar"
    _make_tar(tar0, {"foo.bin": b"FOO"})

    # Pre-build.
    build_or_load_taridx(tar0)
    sidecar = shard_dir / "shard_000.tar.taridx"
    assert sidecar.exists()
    # Move mtime back so we can detect the rebuild.
    old = sidecar.stat().st_mtime - 100
    os.utime(sidecar, (old, old))

    cfg = {
        "services": [],
        "collections": [{
            "name": "wm",
            "default_view": "kf",
            "views": {
                "kf": {
                    "kind": "bytes",
                    "source": {
                        "source": "tar",
                        "tar_template": str(shard_dir / "shard_{shard:03d}.tar"),
                        "matcher": {"kind": "glob", "pattern": "{id}.bin"},
                    },
                },
            },
        }],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [sys.executable, "-m", "routir.collections.indexing.warmup",
         str(cfg_path), "--quiet", "--force"],
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"warmup failed:\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    # Mtime advanced -> rebuild happened.
    assert sidecar.stat().st_mtime > old


# ----------------------------------------------------------------- legacy cache_path back-compat


def test_legacy_cache_path_maps_to_cache_dir(tmp_path):
    """``CollectionConfig(cache_path=...)`` should populate the new
    ``cache_dir`` on the synthesised TextJsonlSource."""
    from routir.config import CollectionConfig

    p = tmp_path / "corpus.jsonl"
    _write_jsonl(p, [{"id": "a", "text": "A"}])
    cache_dir = tmp_path / "legacy-cache"

    cfg = CollectionConfig(
        name="legacy",
        doc_path=str(p),
        id_field="id",
        content_field="text",
        cache_path=str(cache_dir),
    )
    assert cfg.views["text"].source.cache_dir == str(cache_dir)
