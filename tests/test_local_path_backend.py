"""Tests for the PR5a bytes view: ``LocalPathSource`` + ``LocalPathBackend``.

Covers:

* Single-file ``path_template`` mode happy path / missing-id KeyError.
* ``path_glob`` mode with multi-match (sorted) and zero-match (empty list).
* ``base_dir`` path-traversal guard (block ``../`` ids; allow normal paths).
* ``mime`` hint forwarded into the payload.
* :class:`ContentProcessor` dispatch into the bytes view (``data`` list,
  ``kind='bytes'``, ``view='img'``).
* End-to-end :class:`SearchPipeline` with a bytes-modality reranker: the
  rerank engine receives ``List[List[bytes]]`` as ``passages``.
* Per-pipeline bytes-content cache FIFO eviction once over the cap.
* :class:`~routir.config.ViewSpec` rejects ``kind``/source mismatches.
"""

import asyncio

import pytest

from routir.collections.processor import ContentProcessor
from routir.collections.views import LocalPathBackend
from routir.collections.views import text_jsonl as _tj_mod
from routir.config import (
    CollectionConfig,
    LocalPathSource,
    TextJsonlSource,
    ViewSpec,
)
from routir.pipeline import SearchPipeline
from routir.processors.abstract import Processor
from routir.processors.registry import ProcessorRegistry


# ----------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Per-test isolation."""
    _tj_mod._READER_CACHE.clear()
    yield
    _tj_mod._READER_CACHE.clear()


def _make_backend(spec, name="img"):
    # CollectionConfig.views is required to be non-empty for back-compat
    # synthesis to be skipped; build a real config with this view in it.
    cfg = CollectionConfig(
        name="dummy",
        views={name: ViewSpec(kind="bytes", source=spec)},
    )
    return LocalPathBackend(name=name, spec=spec, collection_config=cfg)


# ----------------------------------------------------------------- path_template


def test_path_template_single_file_happy(tmp_path):
    p = tmp_path / "foo.jpg"
    p.write_bytes(b"\x89PNG\x00synthetic-1")
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}.jpg"),
        mime="image/jpeg",
    )
    backend = _make_backend(spec)
    out = backend["foo"]
    assert out["data"] == [b"\x89PNG\x00synthetic-1"]
    assert out["mime"] == "image/jpeg"


def test_path_template_missing_id_raises(tmp_path):
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}.jpg"),
    )
    backend = _make_backend(spec)
    assert "missing" not in backend
    with pytest.raises(KeyError):
        backend["missing"]


def test_path_template_contains_present_only_when_file_exists(tmp_path):
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}.bin"),
    )
    backend = _make_backend(spec)
    assert "anything" not in backend
    (tmp_path / "anything.bin").write_bytes(b"data")
    assert "anything" in backend


# ----------------------------------------------------------------- path_glob


def test_path_glob_multi_match_sorted(tmp_path):
    # Mixed creation order; the backend must return them lexically sorted.
    (tmp_path / "foo_010.jpg").write_bytes(b"ten")
    (tmp_path / "foo_001.jpg").write_bytes(b"one")
    (tmp_path / "foo_002.jpg").write_bytes(b"two")
    spec = LocalPathSource(
        source="local_path",
        path_glob=str(tmp_path / "{id}_*.jpg"),
    )
    backend = _make_backend(spec)
    out = backend["foo"]
    # foo_001 < foo_002 < foo_010 lexically.
    assert out["data"] == [b"one", b"two", b"ten"]


def test_path_glob_zero_match_returns_empty_data(tmp_path):
    """Pin Scott's audio-only edge case: zero keyframes is legal."""
    spec = LocalPathSource(
        source="local_path",
        path_glob=str(tmp_path / "{id}_*.jpg"),
    )
    backend = _make_backend(spec)
    # __contains__ stays True for path_glob even on zero match.
    assert "no-such-id" in backend
    out = backend["no-such-id"]
    assert out["data"] == []


# ----------------------------------------------------------------- base_dir guard


def test_base_dir_blocks_traversal(tmp_path):
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}"),
        base_dir=str(tmp_path),
    )
    backend = _make_backend(spec)
    # An id that escapes the base_dir via ``../`` must raise ValueError when
    # accessed directly and report not-present via __contains__.
    with pytest.raises(ValueError, match="escapes base_dir"):
        backend["../etc/passwd"]
    assert "../etc/passwd" not in backend


def test_base_dir_allows_normal_paths(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"ok-bytes")
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}.jpg"),
        base_dir=str(tmp_path),
    )
    backend = _make_backend(spec)
    out = backend["ok"]
    assert out["data"] == [b"ok-bytes"]


def test_base_dir_block_applies_to_glob(tmp_path):
    """Path-traversal guard also applies in path_glob mode."""
    other = tmp_path.parent / "elsewhere"
    other.mkdir(exist_ok=True)
    (other / "leak.jpg").write_bytes(b"leak")
    spec = LocalPathSource(
        source="local_path",
        path_glob=str(other / "{id}.jpg"),
        base_dir=str(tmp_path),
    )
    backend = _make_backend(spec)
    with pytest.raises(ValueError, match="escapes base_dir"):
        backend["leak"]


def test_mime_omitted_when_not_set(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"raw")
    spec = LocalPathSource(
        source="local_path",
        path_template=str(tmp_path / "{id}.bin"),
    )
    backend = _make_backend(spec)
    out = backend["x"]
    assert "mime" not in out


# ----------------------------------------------------------------- ContentProcessor dispatch


async def test_content_processor_dispatches_bytes_view(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"jpeg-bytes-for-a")
    cfg = CollectionConfig(
        name="bytes-coll",
        default_view="img",
        views={
            "img": ViewSpec(
                kind="bytes",
                source=LocalPathSource(
                    source="local_path",
                    path_template=str(tmp_path / "{id}.jpg"),
                    mime="image/jpeg",
                ),
            ),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)
    payload = await cp._submit({"id": "a", "view": "img"})
    assert payload["data"] == [b"jpeg-bytes-for-a"]
    assert payload["view"] == "img"
    assert payload["kind"] == "bytes"
    assert payload["mime"] == "image/jpeg"


# ----------------------------------------------------------------- ViewSpec validation


def test_view_spec_rejects_text_kind_with_local_path(tmp_path):
    with pytest.raises(ValueError, match="LocalPathSource is only valid for kind='bytes'"):
        ViewSpec(
            kind="text",
            source=LocalPathSource(
                source="local_path",
                path_template=str(tmp_path / "{id}.jpg"),
            ),
        )


def test_view_spec_rejects_bytes_kind_with_text_jsonl(tmp_path):
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text('{"id": "a", "text": "t"}\n')
    with pytest.raises(ValueError, match="TextJsonlSource is only valid for kind='text'"):
        ViewSpec(
            kind="bytes",
            source=TextJsonlSource(
                source="text_jsonl", doc_path=str(jsonl), id_field="id",
            ),
        )


def test_local_path_source_requires_exactly_one_mode():
    with pytest.raises(ValueError, match="exactly one of path_template or path_glob"):
        LocalPathSource(source="local_path")
    with pytest.raises(ValueError, match="exactly one of path_template or path_glob"):
        LocalPathSource(
            source="local_path",
            path_template="/foo/{id}",
            path_glob="/foo/{id}_*",
        )


# ----------------------------------------------------------------- end-to-end pipeline


class _StubSearch(Processor):
    def __init__(self, scores):
        super().__init__(cache_size=-1)
        self._scores = scores

    async def _submit(self, item):
        return {"scores": dict(self._scores)}


class _RecordingBytesReranker(Processor):
    """Reranker stub that records the ``passages`` payload it received."""

    def __init__(self):
        super().__init__(cache_size=-1)
        self.received_passages = None

    async def _submit(self, item):
        self.received_passages = item.get("passages")
        n = len(self.received_passages or [])
        return {"scores": [float(i) for i in range(n)]}


@pytest.fixture
def bytes_pipeline_stubs(tmp_path):
    """Register a text search 'bm25' and a bytes-modality rerank 'bytes-rr'.

    Also creates a bytes collection ``imgcoll`` backed by ``LocalPathBackend``
    so the pipeline can fetch doc bytes via the real /content path.
    """
    # Create JPG-like files for two docs.
    (tmp_path / "doc1.jpg").write_bytes(b"\x89PNG\x00DOC1")
    (tmp_path / "doc2.jpg").write_bytes(b"\x89PNG\x00DOC2-longer")

    cfg = CollectionConfig(
        name="imgcoll",
        default_view="img",
        views={
            "img": ViewSpec(
                kind="bytes",
                source=LocalPathSource(
                    source="local_path",
                    path_template=str(tmp_path / "{id}.jpg"),
                    mime="image/jpeg",
                ),
            ),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)

    bm25 = _StubSearch({"doc1": 1.0, "doc2": 0.5})
    rr = _RecordingBytesReranker()

    ProcessorRegistry.register("bm25", "search", bm25, view_kind="text")
    ProcessorRegistry.register("bytes-rr", "search", _StubSearch({"doc1": 1.0}), view_kind="text")
    ProcessorRegistry.register("bytes-rr", "score", rr, view_kind="bytes")
    ProcessorRegistry.register(
        "imgcoll", "content", cp,
        views={"img": "bytes"}, default_view="img",
    )
    yield {"rerank": rr, "search": bm25, "tmp": tmp_path}
    for name in ("bm25", "bytes-rr", "imgcoll"):
        ProcessorRegistry.all_services.pop(name, None)
        ProcessorRegistry.slot_meta.pop(name, None)


async def test_end_to_end_bytes_pipeline(bytes_pipeline_stubs):
    """`bm25%100 >> bytes-rr@img%10` end-to-end: the engine receives
    ``List[List[bytes]]`` for ``passages``, one inner list per doc."""
    pipeline = SearchPipeline.from_string(
        "bm25%100 >> bytes-rr@img%10", collection="imgcoll",
    )
    result = await pipeline.run("query")
    # The rerank stub got both docs' bytes payloads.
    passages = bytes_pipeline_stubs["rerank"].received_passages
    assert passages is not None
    # Outer list: one entry per doc.  Inner list: bytes blobs for that doc.
    assert len(passages) == 2
    assert all(isinstance(p, list) for p in passages)
    assert all(isinstance(b, (bytes, bytearray)) for inner in passages for b in inner)
    # Payloads match the on-disk bytes (doc1 / doc2 sorted alphabetically).
    flat = [b for inner in passages for b in inner]
    assert b"\x89PNG\x00DOC1" in flat
    assert b"\x89PNG\x00DOC2-longer" in flat
    # Final result shape: a dict of {docid: float} as for any rerank.
    assert set(result["scores"].keys()) == {"doc1", "doc2"}


# ----------------------------------------------------------------- bytes cache cap


async def test_bytes_cache_cap_evicts_fifo(bytes_pipeline_stubs):
    """With a 10-byte cap, inserting two 6-byte payloads evicts the first."""
    pipeline = SearchPipeline.from_string(
        "bm25%100 >> bytes-rr@img%10",
        collection="imgcoll",
        bytes_content_cache_max_bytes=10,
    )
    # Pre-load two distinct ids via get_doc_content directly.  Use the
    # backend so we control what's in the cache (avoid relying on the
    # registered fixture files' sizes).
    tmp = bytes_pipeline_stubs["tmp"]
    (tmp / "small1.jpg").write_bytes(b"AAAAAA")  # 6 bytes
    (tmp / "small2.jpg").write_bytes(b"BBBBBB")  # 6 bytes; total 12 > 10
    v1 = await pipeline.get_doc_content("small1", "img")
    assert v1 == [b"AAAAAA"]
    assert ("img", "small1") in pipeline.doc_content_cache
    v2 = await pipeline.get_doc_content("small2", "img")
    assert v2 == [b"BBBBBB"]
    # 'small1' should have been evicted; only 'small2' remains.
    assert ("img", "small1") not in pipeline.doc_content_cache
    assert ("img", "small2") in pipeline.doc_content_cache


async def test_bytes_cache_no_cap_keeps_everything(bytes_pipeline_stubs):
    """With no cap (default), nothing gets evicted."""
    pipeline = SearchPipeline.from_string(
        "bm25%100 >> bytes-rr@img%10", collection="imgcoll",
    )
    tmp = bytes_pipeline_stubs["tmp"]
    (tmp / "k1.jpg").write_bytes(b"AAAAAA")
    (tmp / "k2.jpg").write_bytes(b"BBBBBB")
    await pipeline.get_doc_content("k1", "img")
    await pipeline.get_doc_content("k2", "img")
    assert ("img", "k1") in pipeline.doc_content_cache
    assert ("img", "k2") in pipeline.doc_content_cache


async def test_bytes_cache_oversize_single_entry_kept(bytes_pipeline_stubs):
    """When a single entry exceeds the cap, it still ends up cached
    (eviction stops at the just-inserted entry).  Practical: predictable
    behaviour even when the cap is misconfigured smaller than the largest
    individual blob."""
    pipeline = SearchPipeline.from_string(
        "bm25%100 >> bytes-rr@img%10",
        collection="imgcoll",
        bytes_content_cache_max_bytes=4,
    )
    tmp = bytes_pipeline_stubs["tmp"]
    (tmp / "big.jpg").write_bytes(b"X" * 100)
    v = await pipeline.get_doc_content("big", "img")
    assert v == [b"X" * 100]
    assert ("img", "big") in pipeline.doc_content_cache
