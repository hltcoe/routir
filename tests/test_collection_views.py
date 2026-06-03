"""Tests for the multi-view collection schema introduced in PR1.

Covers:
* legacy single-view configs (only ``doc_path`` + ``content_field`` + ``id_field``)
  still load and serve ``/content`` correctly through the back-compat shim.
* explicit two-view configs, verifying per-view dispatch and shared
  underlying reader for views over the same ``(doc_path, id_field)``.
* ``default_view`` auto-election, required-error for multi-view configs
  without a default, and ``ValueError`` for an unknown default name.
* cache key correctness: same id under two different views must produce
  two independent cache entries.
* preservation of the legacy ``"title"`` field in the response payload.
* race-safe sidecar build via atomic rename.
"""

import asyncio
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from routir.collections.processor import ContentProcessor
from routir.collections.views import text_jsonl as _tj_mod
from routir.config import CollectionConfig, TextJsonlSource, ViewSpec


async def _drain_cache_writes():
    """Yield to the event loop so fire-and-forget cache writes complete."""
    # ``Processor.submit`` schedules the cache write via ``asyncio.create_task``
    # without awaiting it.  A single ``sleep(0)`` is enough to let the scheduler
    # run pending tasks; do a few to be safe.
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Per-test isolation so different jsonl fixtures don't share readers."""
    _tj_mod._READER_CACHE.clear()
    yield
    _tj_mod._READER_CACHE.clear()


def _write_jsonl(path: Path, docs):
    with path.open("w") as fw:
        for d in docs:
            fw.write(json.dumps(d) + "\n")


@pytest.fixture
def simple_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "doc1", "text": "hello world", "title": "T1"},
            {"id": "doc2", "text": "second doc", "title": "T2"},
            {"id": "doc3", "text": "third doc"},
        ],
    )
    return p


@pytest.fixture
def multifield_jsonl(tmp_path):
    p = tmp_path / "multi.jsonl"
    _write_jsonl(
        p,
        [
            {"docid": "a", "ocr": "OCR-A", "asr": "ASR-A"},
            {"docid": "b", "ocr": "OCR-B", "asr": "ASR-B"},
        ],
    )
    return p


# ---------------------------------------------------------------- back-compat


def test_legacy_single_view_synthesizes_text_view(simple_jsonl):
    cfg = CollectionConfig(
        name="legacy",
        doc_path=str(simple_jsonl),
        id_field="id",
        content_field="text",
    )
    assert set(cfg.views.keys()) == {"text"}
    assert cfg.default_view == "text"
    spec = cfg.views["text"]
    assert spec.kind == "text"
    assert isinstance(spec.source, TextJsonlSource)
    assert spec.source.content_fields == ["text"]
    assert spec.source.id_field == "id"


async def test_legacy_submit_returns_text_view_kind_decorations(simple_jsonl):
    cfg = CollectionConfig(
        name="legacy",
        doc_path=str(simple_jsonl),
        content_field="text",
    )
    cp = ContentProcessor(cfg, cache_size=-1)
    result = await cp._submit({"id": "doc1"})
    assert result["text"] == "hello world"
    assert result["title"] == "T1"
    assert result["view"] == "text"
    assert result["kind"] == "text"


async def test_legacy_submit_missing_id_returns_error(simple_jsonl):
    cfg = CollectionConfig(name="legacy", doc_path=str(simple_jsonl))
    cp = ContentProcessor(cfg, cache_size=-1)
    result = await cp._submit({"id": "no-such-doc"})
    assert "error" in result
    assert "no-such-doc" in result["error"]


async def test_title_preserved_when_present_omitted_when_absent(simple_jsonl):
    cfg = CollectionConfig(name="legacy", doc_path=str(simple_jsonl))
    cp = ContentProcessor(cfg, cache_size=-1)
    r1 = await cp._submit({"id": "doc1"})
    r3 = await cp._submit({"id": "doc3"})
    assert r1["title"] == "T1"
    assert "title" not in r3


# ---------------------------------------------------------------- multi-view


def _two_views_same_file_cfg(jsonl_path):
    return CollectionConfig(
        name="multi",
        default_view="ocr",
        views={
            "ocr": ViewSpec(
                kind="text",
                source=TextJsonlSource(
                    source="text_jsonl",
                    doc_path=str(jsonl_path),
                    id_field="docid",
                    content_fields="ocr",
                ),
            ),
            "asr": ViewSpec(
                kind="text",
                source=TextJsonlSource(
                    source="text_jsonl",
                    doc_path=str(jsonl_path),
                    id_field="docid",
                    content_fields="asr",
                ),
            ),
        },
    )


async def test_two_views_dispatch_returns_right_field(multifield_jsonl):
    cfg = _two_views_same_file_cfg(multifield_jsonl)
    cp = ContentProcessor(cfg, cache_size=-1)
    ocr = await cp._submit({"id": "a", "view": "ocr"})
    asr = await cp._submit({"id": "a", "view": "asr"})
    assert ocr["text"] == "OCR-A"
    assert ocr["view"] == "ocr"
    assert asr["text"] == "ASR-A"
    assert asr["view"] == "asr"


async def test_two_views_default_view_used_when_unspecified(multifield_jsonl):
    cfg = _two_views_same_file_cfg(multifield_jsonl)
    cp = ContentProcessor(cfg, cache_size=-1)
    result = await cp._submit({"id": "b"})
    assert result["view"] == "ocr"
    assert result["text"] == "OCR-B"


def test_two_views_share_one_reader(multifield_jsonl):
    cfg = _two_views_same_file_cfg(multifield_jsonl)
    ContentProcessor(cfg, cache_size=-1)
    # The cache key is (doc_path, id_field).  Both views share these, so
    # exactly one reader should have been built.
    assert len(_tj_mod._READER_CACHE) == 1


def test_two_views_different_files_dont_share_reader(tmp_path):
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    _write_jsonl(p1, [{"id": "x", "text": "X"}])
    _write_jsonl(p2, [{"id": "y", "text": "Y"}])
    cfg = CollectionConfig(
        name="sep",
        default_view="va",
        views={
            "va": ViewSpec(source=TextJsonlSource(source="text_jsonl", doc_path=str(p1))),
            "vb": ViewSpec(source=TextJsonlSource(source="text_jsonl", doc_path=str(p2))),
        },
    )
    ContentProcessor(cfg, cache_size=-1)
    assert len(_tj_mod._READER_CACHE) == 2


# ---------------------------------------------------------------- default_view


def test_default_view_auto_elected_for_single_view(simple_jsonl):
    cfg = CollectionConfig(
        name="single",
        views={
            "v1": ViewSpec(source=TextJsonlSource(source="text_jsonl", doc_path=str(simple_jsonl))),
        },
    )
    assert cfg.default_view == "v1"


def test_default_view_required_for_multi_view(multifield_jsonl):
    with pytest.raises(ValueError, match="multiple views but no default_view"):
        CollectionConfig(
            name="multi",
            views={
                "ocr": ViewSpec(source=TextJsonlSource(
                    source="text_jsonl",
                    doc_path=str(multifield_jsonl),
                    id_field="docid",
                    content_fields="ocr",
                )),
                "asr": ViewSpec(source=TextJsonlSource(
                    source="text_jsonl",
                    doc_path=str(multifield_jsonl),
                    id_field="docid",
                    content_fields="asr",
                )),
            },
        )


def test_default_view_unknown_name_raises(simple_jsonl):
    with pytest.raises(ValueError, match="default_view 'nope' not in views"):
        CollectionConfig(
            name="bad",
            default_view="nope",
            views={
                "v1": ViewSpec(source=TextJsonlSource(source="text_jsonl", doc_path=str(simple_jsonl))),
            },
        )


async def test_submit_unknown_view_returns_error(multifield_jsonl):
    cfg = _two_views_same_file_cfg(multifield_jsonl)
    cp = ContentProcessor(cfg, cache_size=-1)
    result = await cp._submit({"id": "a", "view": "no-view"})
    assert "error" in result
    assert "no-view" in result["error"]


# ---------------------------------------------------------------- cache keying


async def test_cache_key_distinguishes_views(multifield_jsonl):
    """Same id under two different views must be cached independently."""
    cfg = _two_views_same_file_cfg(multifield_jsonl)
    cp = ContentProcessor(cfg, cache_size=4)

    r1 = await cp.submit({"id": "a", "view": "ocr"})
    r2 = await cp.submit({"id": "a", "view": "asr"})
    assert r1["text"] == "OCR-A"
    assert r2["text"] == "ASR-A"
    assert r1["cached"] is False
    assert r2["cached"] is False

    await _drain_cache_writes()

    # Hits: each view should return its own cached entry, not the other view's.
    r1b = await cp.submit({"id": "a", "view": "ocr"})
    r2b = await cp.submit({"id": "a", "view": "asr"})
    assert r1b["text"] == "OCR-A"
    assert r2b["text"] == "ASR-A"
    assert r1b["cached"] is True
    assert r2b["cached"] is True


async def test_cache_key_default_view_fallback(simple_jsonl):
    """Legacy ``{"id": ...}`` requests should still hit the cache."""
    cfg = CollectionConfig(name="legacy", doc_path=str(simple_jsonl))
    cp = ContentProcessor(cfg, cache_size=4)

    miss = await cp.submit({"id": "doc1"})
    await _drain_cache_writes()
    hit = await cp.submit({"id": "doc1"})
    assert miss["cached"] is False
    assert hit["cached"] is True


# ---------------------------------------------------------------- roundtrip


def test_legacy_json_roundtrip(simple_jsonl):
    js = json.dumps({
        "name": "legacy",
        "doc_path": str(simple_jsonl),
        "id_field": "id",
        "content_field": "text",
    })
    cfg = CollectionConfig.model_validate_json(js)
    assert set(cfg.views) == {"text"}
    assert cfg.default_view == "text"
    assert cfg.views["text"].source.doc_path == str(simple_jsonl)


def test_modern_json_roundtrip(simple_jsonl):
    js = json.dumps({
        "name": "modern",
        "views": {
            "text": {
                "kind": "text",
                "source": {
                    "source": "text_jsonl",
                    "doc_path": str(simple_jsonl),
                    "id_field": "id",
                    "content_fields": ["text"],
                },
            },
        },
    })
    cfg = CollectionConfig.model_validate_json(js)
    assert cfg.default_view == "text"
    assert cfg.views["text"].source.content_fields == ["text"]


# ---------------------------------------------------------------- race-safety


def test_offsetmap_temp_file_cleaned_on_failure(simple_jsonl, tmp_path, monkeypatch):
    """If pickling fails mid-write, the .tmp.* sidecar must not survive.

    PR7 rename: ``offset_fn`` (single explicit path) -> ``cache_dir`` (directory
    inside the fallback chain).  The cache_dir is empty here, so the only place
    the sidecar can land is the user-provided directory.  We also point
    XDG_CACHE_HOME at the same tmp_path so we can detect any cross-leak.
    """
    from routir.collections.indexing.offset_file import OffsetFile

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Pin XDG so a fallback to ~/.cache doesn't pollute the user's real cache
    # (and so leftover detection below can see if anything leaked there).
    xdg_root = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_root))

    def boom_dump(*a, **kw):
        raise RuntimeError("simulated pickle failure")

    with patch("routir.collections.indexing.offset_file.pickle.dump", boom_dump):
        with pytest.raises(RuntimeError, match="simulated pickle failure"):
            OffsetFile(simple_jsonl, cache_dir=str(cache_dir), id_field="id")

    # No leftover *.tmp.* in any candidate location, and no completed sidecar
    # either (pickle.dump raised before os.replace).
    def _walk_for_leftovers(root: Path):
        if not root.exists():
            return []
        return [p for p in root.rglob("*") if p.is_file() and ".offsetmap" in p.name]

    assert _walk_for_leftovers(cache_dir) == []
    assert _walk_for_leftovers(simple_jsonl.parent) == []
    assert _walk_for_leftovers(xdg_root) == []

    # The next (real) build under the same cache_dir must succeed.
    real = OffsetFile(simple_jsonl, cache_dir=str(cache_dir), id_field="id")
    assert "doc1" in real
