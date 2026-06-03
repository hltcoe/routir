"""PR4 modality-aware tests.

Covers:

* ``ProcessorRegistry.register(..., view_kind=...)`` stores the metadata and
  ``get_meta`` / ``get_slot_meta`` read it back.
* ``SearchPipeline.verify()`` slot-kind mismatch error.
* Reranker.search_batch refuses bytes-modality engines.
* REST ``/score`` returns 400 for a bytes-modality score service.
* REST ``/content`` returns 400 for a bytes view; text view returns 200.
* ``/avail`` REST + gRPC carry the new structured shape.
* ``auto_add_relay_services`` consumes the new shape and registers slots with
  the right ``view_kind`` / ``views`` metadata (using a fake remote response).
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from routir.collections.processor import ContentProcessor
from routir.collections.relay import RelayContentProcessor
from routir.collections.views import text_jsonl as _tj_mod
from routir.config import CollectionConfig, TextJsonlSource, ViewSpec
from routir.models.abstract import Engine, Reranker
from routir.pipeline import SearchPipeline
from routir.processors.abstract import Processor
from routir.processors.registry import ProcessorRegistry


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TRIVIAL_ENGINE_PATH = REPO_ROOT / "tests" / "_trivial_engine.py"
BYTES_ENGINE_PATH = REPO_ROOT / "tests" / "_bytes_engine.py"
PYTHON = os.environ.get("ROUTIR_TEST_PYTHON") or sys.executable
BOOT_TIMEOUT = 30.0


# ----------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Per-test isolation: don't share jsonl readers across tests."""
    _tj_mod._READER_CACHE.clear()
    yield
    _tj_mod._READER_CACHE.clear()


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _wait_for_ping(base_url: str, timeout: float = BOOT_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    url = base_url.rstrip("/") + "/ping"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.25)
    return False


def _spawn_server(config_path: Path, args: list):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.Popen(
        [PYTHON, "-m", "routir.serve", str(config_path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def _terminate(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _http_post_json(url: str, payload: dict):
    """POST JSON and return (status, body). Doesn't raise on non-2xx."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _http_get_json(url: str):
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


@pytest.fixture(scope="module")
def multiview_corpus(tmp_path_factory):
    p = tmp_path_factory.mktemp("pr4-corpus") / "corpus.jsonl"
    with p.open("w") as fw:
        fw.write(json.dumps({"id": "d1", "ocr": "OCR-1", "asr": "ASR-1"}) + "\n")
        fw.write(json.dumps({"id": "d2", "ocr": "OCR-2", "asr": "ASR-2"}) + "\n")
    return p


@pytest.fixture(scope="module")
def pr4_config_path(tmp_path_factory, multiview_corpus):
    """Config with a text search service, a bytes-modality score service,
    and a multi-view text collection.  The bytes service isn't actually
    bytes-capable end-to-end (no bytes view backends ship yet — PR5a/5b),
    but the *slot* gets registered with ``view_kind="bytes"`` so we can
    test the validation paths.
    """
    p = tmp_path_factory.mktemp("pr4-cfg") / "pr4.json"
    cfg = {
        "file_imports": [str(TRIVIAL_ENGINE_PATH), str(BYTES_ENGINE_PATH)],
        "services": [
            {
                "name": "trivial",
                "engine": "TrivialSearchEngine",
                "config": {},
                "batch_size": 4,
                "max_wait_time": 0.01,
                "cache": -1,
            },
            {
                "name": "bytes-rr",
                "engine": "TrivialBytesScoreEngine",
                "config": {},
                "batch_size": 4,
                "max_wait_time": 0.01,
                "cache": -1,
            },
        ],
        "collections": [
            {
                "name": "vcoll",
                "default_view": "ocr",
                "views": {
                    "ocr": {
                        "kind": "text",
                        "source": {
                            "source": "text_jsonl",
                            "doc_path": str(multiview_corpus),
                            "id_field": "id",
                            "content_fields": "ocr",
                        },
                    },
                    "asr": {
                        "kind": "text",
                        "source": {
                            "source": "text_jsonl",
                            "doc_path": str(multiview_corpus),
                            "id_field": "id",
                            "content_fields": "asr",
                        },
                    },
                },
            }
        ],
        "server_imports": [],
        "pipeline_aliases": {},
    }
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def pr4_server_both(pr4_config_path):
    rest_port = _pick_free_port()
    grpc_port = _pick_free_port()
    proc = _spawn_server(
        pr4_config_path,
        ["--port", str(rest_port), "--host", "127.0.0.1",
         "--grpc", "--grpc-port", str(grpc_port)],
    )
    rest_url = f"http://127.0.0.1:{rest_port}"
    grpc_target = f"127.0.0.1:{grpc_port}"
    try:
        if not _wait_for_ping(rest_url):
            _terminate(proc)
            pytest.skip("PR4 server boot timed out")
        yield rest_url, grpc_target
    finally:
        _terminate(proc)


# ------------------------------------------------------- registry unit


def test_register_stores_view_kind_meta():
    class _P(Processor):
        def __init__(self):
            super().__init__(cache_size=-1)

        async def _submit(self, item):
            return {}

    p = _P()
    name = "pr4-reg-test"
    ProcessorRegistry.register(name, "score", p, view_kind="bytes")
    try:
        assert ProcessorRegistry.get_meta(name, "score") == {"view_kind": "bytes"}
        assert ProcessorRegistry.get_slot_meta(name, "score", "view_kind") == "bytes"
        # Default for absent keys.
        assert ProcessorRegistry.get_slot_meta(name, "score", "nonsense", "fallback") == "fallback"
        # Empty dict for unknown slot.
        assert ProcessorRegistry.get_meta(name, "search") == {}
    finally:
        ProcessorRegistry.all_services.pop(name, None)
        ProcessorRegistry.slot_meta.pop(name, None)


def test_register_meta_isolated_from_other_slots():
    class _P(Processor):
        def __init__(self):
            super().__init__(cache_size=-1)

        async def _submit(self, item):
            return {}

    name = "pr4-reg-iso"
    ProcessorRegistry.register(name, "score", _P(), view_kind="bytes")
    ProcessorRegistry.register(name, "search", _P(), view_kind="text")
    try:
        assert ProcessorRegistry.get_meta(name, "score")["view_kind"] == "bytes"
        assert ProcessorRegistry.get_meta(name, "search")["view_kind"] == "text"
    finally:
        ProcessorRegistry.all_services.pop(name, None)
        ProcessorRegistry.slot_meta.pop(name, None)


# ------------------------------------------------------- pipeline verify slot-kind


class _StubProcessor(Processor):
    def __init__(self, kind="search"):
        super().__init__(cache_size=-1)
        self.kind = kind

    async def _submit(self, item):
        if self.kind == "search":
            return {"scores": {"d1": 1.0}}
        if self.kind == "rerank":
            return {"scores": [0.0 for _ in item.get("passages", [])]}
        return {}


@pytest.fixture
def kf_view_collection(multiview_corpus):
    """Collection with a 'text' view and a 'kf' view we'll mark as bytes via
    registry slot metadata.  We don't actually have a bytes backend yet
    (PR5a/5b), so the view itself is wired as text on disk and the *metadata*
    declares it bytes for the test.
    """
    cfg = CollectionConfig(
        name="vcoll-kf",
        default_view="ocr",
        views={
            "ocr": ViewSpec(source=TextJsonlSource(
                source="text_jsonl", doc_path=str(multiview_corpus),
                id_field="id", content_fields="ocr",
            )),
            "kf": ViewSpec(source=TextJsonlSource(
                source="text_jsonl", doc_path=str(multiview_corpus),
                id_field="id", content_fields="asr",
            )),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)
    # Register with a synthetic kind map that marks 'kf' as bytes so we can
    # exercise the slot-kind mismatch in verify().
    ProcessorRegistry.register(
        "vcoll-kf", "content", cp,
        views={"ocr": "text", "kf": "bytes"},
        default_view="ocr",
    )
    yield cp
    ProcessorRegistry.all_services.pop("vcoll-kf", None)
    ProcessorRegistry.slot_meta.pop("vcoll-kf", None)


@pytest.fixture
def text_and_bytes_scorers():
    """Register one text reranker 'text-rr' and one bytes reranker 'bytes-rr'."""
    text_rr = _StubProcessor("rerank")
    bytes_rr = _StubProcessor("rerank")
    ProcessorRegistry.register("text-rr", "search", _StubProcessor("search"), view_kind="text")
    ProcessorRegistry.register("text-rr", "score", text_rr, view_kind="text")
    ProcessorRegistry.register("bytes-rr", "search", _StubProcessor("search"), view_kind="text")
    ProcessorRegistry.register("bytes-rr", "score", bytes_rr, view_kind="bytes")
    ProcessorRegistry.register("retr", "search", _StubProcessor("search"), view_kind="text")
    yield
    for n in ("text-rr", "bytes-rr", "retr"):
        ProcessorRegistry.all_services.pop(n, None)
        ProcessorRegistry.slot_meta.pop(n, None)


def test_verify_text_view_routed_to_text_service(kf_view_collection, text_and_bytes_scorers):
    """Text view through a text-modality score service — no error."""
    SearchPipeline.from_string("retr >> text-rr@ocr", collection="vcoll-kf")


def test_verify_bytes_view_routed_to_text_service_raises(kf_view_collection, text_and_bytes_scorers):
    """Bytes view routed to a text scorer must raise with a kind-mismatch
    message."""
    with pytest.raises(ValueError, match="accepts 'text'"):
        SearchPipeline.from_string("retr >> text-rr@kf", collection="vcoll-kf")


def test_verify_bytes_view_routed_to_bytes_service_ok(kf_view_collection, text_and_bytes_scorers):
    """Bytes view through a bytes-modality scorer is allowed."""
    SearchPipeline.from_string("retr >> bytes-rr@kf", collection="vcoll-kf")


def test_verify_text_view_routed_to_bytes_service_raises(kf_view_collection, text_and_bytes_scorers):
    """Mismatch in the opposite direction also raises."""
    with pytest.raises(ValueError, match="accepts 'bytes'"):
        SearchPipeline.from_string("retr >> bytes-rr@ocr", collection="vcoll-kf")


# ------------------------------------------------------- Reranker.search_batch guard


def test_reranker_search_batch_refuses_bytes_engine():
    """A Reranker subclass whose ``accepts_view_kind`` is "bytes" must refuse
    to enter the text-fetching helper path.
    """
    class _BytesReranker(Reranker):
        accepts_view_kind = "bytes"

        async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
            return [[0.0] * len(passages)]

    r = _BytesReranker(name="bytes-rr-helper", config={})
    # The check fires before upstream/text_service resolution; just calling
    # search_batch with a query is enough.
    with pytest.raises(RuntimeError, match="bytes"):
        asyncio.run(r.search_batch(["q"]))


# ------------------------------------------------------- REST 400 paths (subprocess fixture)


def test_avail_rest_shape(pr4_server_both):
    rest_url, _grpc = pr4_server_both
    status, body = _http_get_json(rest_url + "/avail")
    assert status == 200
    # Collections are exposed as a dict-of-collection-info.
    assert isinstance(body["collection"], dict)
    assert "vcoll" in body["collection"]
    assert body["collection"]["vcoll"]["default"] == "ocr"
    assert body["collection"]["vcoll"]["views"] == {"ocr": "text", "asr": "text"}
    # Score view kinds for both services.
    assert body["score_view_kinds"]["bytes-rr"] == "bytes"
    # ``trivial`` is search-only, so it shouldn't appear in score_view_kinds.
    assert "trivial" not in body["score_view_kinds"]
    # Callable roles list still includes both services where applicable.
    assert "trivial" in body["search"]
    assert "bytes-rr" in body["score"]


async def test_avail_grpc_shape(pr4_server_both):
    from routir.client import AsyncClient

    rest_url, grpc_target = pr4_server_both
    async with AsyncClient(endpoint=rest_url, grpc_endpoint=grpc_target) as c:
        assert c.transport == "grpc"
        avail = await c.avail()
    # gRPC: the ``services`` map omits collections entirely; they live in their
    # own dedicated keys.
    assert "collection" in avail and isinstance(avail["collection"], dict)
    assert "vcoll" in avail["collection"]
    assert avail["collection"]["vcoll"]["default"] == "ocr"
    assert avail["collection"]["vcoll"]["views"] == {"ocr": "text", "asr": "text"}
    assert avail["score_view_kinds"]["bytes-rr"] == "bytes"


def test_rest_score_400_for_bytes_service(pr4_server_both):
    rest_url, _grpc = pr4_server_both
    status, body = _http_post_json(
        rest_url + "/score",
        {"service": "bytes-rr", "query": "q", "passages": ["p1"]},
    )
    assert status == 400
    assert "bytes" in body["error"].lower()
    assert "grpc" in body["error"].lower() or "pipeline" in body["error"].lower()


def test_rest_content_400_for_bytes_view(tmp_path):
    """Spin up a server whose collection declares one text view and one
    bytes view, then check that REST /content rejects the bytes view with
    a 400 and serves the text view normally.
    """
    corpus = tmp_path / "c.jsonl"
    with corpus.open("w") as fw:
        fw.write(json.dumps({"id": "d1", "ocr": "OCR-1", "asr": "ASR-1"}) + "\n")
    # PR5a: bytes views must use a bytes source (LocalPathSource).  Create a
    # minimal local-path bytes view alongside the OCR text view.
    (tmp_path / "d1.jpg").write_bytes(b"\x89PNG\x00d1")
    cfg_obj = {
        "file_imports": [str(TRIVIAL_ENGINE_PATH)],
        "services": [
            {"name": "trivial", "engine": "TrivialSearchEngine", "config": {}}
        ],
        "collections": [{
            "name": "bvc",
            "default_view": "ocr",
            "views": {
                "ocr": {"kind": "text", "source": {
                    "source": "text_jsonl", "doc_path": str(corpus),
                    "id_field": "id", "content_fields": "ocr"}},
                "kf":  {"kind": "bytes", "source": {
                    "source": "local_path",
                    "path_template": str(tmp_path / "{id}.jpg"),
                    "mime": "image/jpeg"}},
            },
        }],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg_obj))
    port = _pick_free_port()
    proc = _spawn_server(cfg_path, ["--port", str(port), "--host", "127.0.0.1"])
    rest_url = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_ping(rest_url):
            _terminate(proc)
            pytest.skip("bytes-view server boot timed out")
        # OCR (text) view succeeds.
        status, body = _http_post_json(
            rest_url + "/content",
            {"collection": "bvc", "id": "d1", "view": "ocr"},
        )
        assert status == 200, body
        assert body["text"] == "OCR-1"
        # KF (bytes) view returns 400.
        status, body = _http_post_json(
            rest_url + "/content",
            {"collection": "bvc", "id": "d1", "view": "kf"},
        )
        assert status == 400
        assert "bytes" in body["error"].lower()
    finally:
        _terminate(proc)


# ------------------------------------------------------- relay round-trip


async def test_relay_threads_view_kind_and_views():
    """``auto_add_relay_services`` must read the new /avail shape and register
    relay slots with the correct ``view_kind`` (score) and ``views`` / ``default``
    (content) metadata.

    We patch :class:`AsyncClient.avail` to return a synthesized response so
    the test stays in-process — no remote server needed.
    """
    from routir.config import load as _load

    fake_avail = {
        "search":         ["s-search"],
        "score":          ["s-bytes", "s-text"],
        "fuse":           [],
        "decompose_query": [],
        "collection": {
            "remote-coll": {
                "views": {"ocr": "text", "kf": "bytes"},
                "default": "ocr",
            },
        },
        "score_view_kinds": {"s-bytes": "bytes", "s-text": "text"},
        "pipeline_aliases": {},
    }

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def avail(self): return fake_avail

    # Patch the AsyncClient symbol the way ``auto_add_relay_services`` looks
    # it up (module-local import).
    with patch.object(_load, "AsyncClient", _FakeClient):
        # ``RelayContentProcessor.start`` is a no-op in tests but real code
        # opens a channel; patch out the underlying client so it doesn't try
        # to talk to the network when ``start`` is called.
        with patch.object(RelayContentProcessor, "start", new=lambda self: asyncio.sleep(0)):
            with patch("routir.processors.score_processors.AsyncPairwiseScoreProcessor.start",
                       new=lambda self: asyncio.sleep(0)):
                with patch("routir.processors.query_processors.AsyncQueryProcessor.start",
                           new=lambda self: asyncio.sleep(0)):
                    await _load.auto_add_relay_services([{"endpoint": "http://fake"}])

    try:
        # Score relays should be registered with the right ``view_kind``.
        assert ProcessorRegistry.has_service("s-bytes", "score")
        assert ProcessorRegistry.has_service("s-text", "score")
        assert ProcessorRegistry.get_meta("s-bytes", "score")["view_kind"] == "bytes"
        assert ProcessorRegistry.get_meta("s-text", "score")["view_kind"] == "text"
        # Content relay should have the views map and default.
        assert ProcessorRegistry.has_service("remote-coll", "content")
        meta = ProcessorRegistry.get_meta("remote-coll", "content")
        assert meta["views"] == {"ocr": "text", "kf": "bytes"}
        assert meta["default_view"] == "ocr"
        # Processor instance also has the views hydrated for verify().
        cp = ProcessorRegistry["remote-coll"]["content"]
        assert isinstance(cp, RelayContentProcessor)
        assert cp.views == {"ocr": "text", "kf": "bytes"}
        assert cp.default_view == "ocr"
    finally:
        for n in ("s-bytes", "s-text", "s-search", "remote-coll"):
            ProcessorRegistry.all_services.pop(n, None)
            ProcessorRegistry.slot_meta.pop(n, None)
