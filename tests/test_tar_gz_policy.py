"""Tests for the PR6 ``.tar.gz`` policy and the ``tar_index`` CLI.

Covers:

* Config-load rejects ``.tar.gz`` tar_template references when
  ``--allow-tar-gz-decompress-cache`` is not set.
* Config-load with the flag decompresses each shard once and rewrites the
  view's ``tar_template`` to the cached ``.tar`` path.
* Decompression is idempotent across calls.
* ``python -m routir.collections.indexing.tar_index <dir>`` writes ``.taridx``
  sidecars; ``--force`` rebuilds in place.
"""

import gzip
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from routir.config import (
    CollectionConfig,
    Config,
    GlobMatcher,
    TarSource,
    ViewSpec,
)
from routir.config.load import _resolve_tar_gz_in_views


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _make_tar_bytes(members):
    """Return the bytes of an in-memory plain tar with the given members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_tar_file(path: Path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_tar_gz_file(path: Path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _make_tar_bytes(members)
    with gzip.open(path, "wb") as fw:
        fw.write(raw)


def _cfg_with_tar_gz_view(tar_template: str) -> Config:
    """Build a Config containing one collection with a single .tar.gz tar view."""
    return Config(
        collections=[
            CollectionConfig(
                name="gzcoll",
                views={
                    "kf": ViewSpec(
                        kind="bytes",
                        source=TarSource(
                            source="tar",
                            tar_template=tar_template,
                            matcher=GlobMatcher(pattern="{id}.jpg"),
                        ),
                    ),
                },
            )
        ]
    )


# ----------------------------------------------------------------- rejection


def test_resolve_rejects_tar_gz_without_flag(tmp_path):
    """Without ``decompress_cache_dir``, .tar.gz refs raise with a clear redirect."""
    gz_path = tmp_path / "x.tar.gz"
    # File need not exist on disk — the policy is detected from the template.
    cfg = _cfg_with_tar_gz_view(str(gz_path))
    with pytest.raises(RuntimeError, match="allow-tar-gz-decompress-cache"):
        _resolve_tar_gz_in_views(cfg, None)


def test_resolve_rejects_tgz_without_flag(tmp_path):
    """``.tgz`` suffix follows the same policy as ``.tar.gz``."""
    cfg = _cfg_with_tar_gz_view(str(tmp_path / "x.tgz"))
    with pytest.raises(RuntimeError, match="allow-tar-gz-decompress-cache"):
        _resolve_tar_gz_in_views(cfg, None)


def test_resolve_noop_for_plain_tar(tmp_path):
    """Configs without any .tar.gz views are a no-op (no error, no rewrite)."""
    cfg = _cfg_with_tar_gz_view(str(tmp_path / "x.tar"))
    _resolve_tar_gz_in_views(cfg, None)
    assert cfg.collections[0].views["kf"].source.tar_template == str(tmp_path / "x.tar")


# ----------------------------------------------------------------- accept + decompress


def test_resolve_decompresses_tar_gz_with_flag(tmp_path):
    """With the flag set, .tar.gz is decompressed and the template rewritten."""
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    gz_path = src_dir / "x.tar.gz"
    _make_tar_gz_file(gz_path, {"foo.jpg": b"\x89PNG\x00FOO"})

    cfg = _cfg_with_tar_gz_view(str(gz_path))
    _resolve_tar_gz_in_views(cfg, str(cache_dir))

    expected_tar = cache_dir / "x.tar"
    assert expected_tar.exists()
    new_tmpl = cfg.collections[0].views["kf"].source.tar_template
    assert new_tmpl == str(expected_tar)
    # Verify the decompressed file is a valid tar with the right member.
    with tarfile.open(expected_tar, mode="r:") as tf:
        names = tf.getnames()
        assert "foo.jpg" in names


def test_resolve_decompresses_sharded_tar_gz(tmp_path):
    """Sharded {shard} templates expand and decompress every concrete shard."""
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    _make_tar_gz_file(src_dir / "shard_000000.tar.gz", {"a.jpg": b"AAA"})
    _make_tar_gz_file(src_dir / "shard_000001.tar.gz", {"b.jpg": b"BBB"})

    cfg = _cfg_with_tar_gz_view(str(src_dir / "shard_{shard:06d}.tar.gz"))
    _resolve_tar_gz_in_views(cfg, str(cache_dir))

    assert (cache_dir / "shard_000000.tar").exists()
    assert (cache_dir / "shard_000001.tar").exists()
    new_tmpl = cfg.collections[0].views["kf"].source.tar_template
    assert new_tmpl == str(cache_dir / "shard_{shard:06d}.tar")


def test_resolve_decompresses_tgz_suffix(tmp_path):
    """``.tgz`` rewrites to ``.tar`` (suffix is replaced rather than stripped)."""
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    tgz = src_dir / "x.tgz"
    _make_tar_gz_file(tgz, {"foo.jpg": b"FOO"})

    cfg = _cfg_with_tar_gz_view(str(tgz))
    _resolve_tar_gz_in_views(cfg, str(cache_dir))

    expected_tar = cache_dir / "x.tar"
    assert expected_tar.exists()
    assert cfg.collections[0].views["kf"].source.tar_template == str(expected_tar)


# ----------------------------------------------------------------- idempotence


def test_resolve_decompress_is_idempotent(tmp_path):
    """A second call reuses the cached .tar — no re-decompression."""
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    gz_path = src_dir / "x.tar.gz"
    _make_tar_gz_file(gz_path, {"foo.jpg": b"FOO"})

    cfg1 = _cfg_with_tar_gz_view(str(gz_path))
    _resolve_tar_gz_in_views(cfg1, str(cache_dir))
    cached_tar = cache_dir / "x.tar"
    assert cached_tar.exists()
    first_mtime = cached_tar.stat().st_mtime

    # Wait a bit so a re-decompress would produce a measurably newer mtime.
    time.sleep(0.05)

    cfg2 = _cfg_with_tar_gz_view(str(gz_path))
    _resolve_tar_gz_in_views(cfg2, str(cache_dir))
    second_mtime = cached_tar.stat().st_mtime

    assert second_mtime == first_mtime, "Decompressed cache should not be rewritten"


# ----------------------------------------------------------------- CLI: tar_index


def _run_tar_index_cli(args, env=None):
    """Run ``python -m routir.collections.indexing.tar_index`` and return CompletedProcess."""
    proc_env = os.environ.copy()
    proc_env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{proc_env.get('PYTHONPATH', '')}"
    if env:
        proc_env.update(env)
    cmd = [sys.executable, "-m", "routir.collections.indexing.tar_index", *args]
    return subprocess.run(cmd, capture_output=True, text=True, env=proc_env, timeout=60)


def test_tar_index_cli_help():
    """``--help`` exits 0 and lists the documented args."""
    result = _run_tar_index_cli(["--help"])
    assert result.returncode == 0
    assert "directory" in result.stdout.lower()
    assert "--force" in result.stdout
    assert "--quiet" in result.stdout


def test_tar_index_cli_builds_taridx(tmp_path):
    """The CLI walks the directory and writes a ``.taridx`` for every .tar."""
    _make_tar_file(tmp_path / "a.tar", {"x.bin": b"X"})
    _make_tar_file(tmp_path / "sub" / "b.tar", {"y.bin": b"Y"})
    result = _run_tar_index_cli([str(tmp_path), "--quiet"])
    assert result.returncode == 0, result.stderr or result.stdout
    assert (tmp_path / "a.tar.taridx").exists()
    assert (tmp_path / "sub" / "b.tar.taridx").exists()


def test_tar_index_cli_force_rebuilds(tmp_path):
    """``--force`` deletes existing sidecars and rebuilds them."""
    tar_path = tmp_path / "a.tar"
    _make_tar_file(tar_path, {"x.bin": b"X"})
    # Initial build.
    result1 = _run_tar_index_cli([str(tmp_path), "--quiet"])
    assert result1.returncode == 0
    sidecar = tmp_path / "a.tar.taridx"
    assert sidecar.exists()

    # Move sidecar mtime back so the unconditional rebuild is observable.
    old_mtime = sidecar.stat().st_mtime - 100
    os.utime(sidecar, (old_mtime, old_mtime))

    result2 = _run_tar_index_cli([str(tmp_path), "--force", "--quiet"])
    assert result2.returncode == 0, result2.stderr or result2.stdout
    assert sidecar.exists()
    assert sidecar.stat().st_mtime > old_mtime
