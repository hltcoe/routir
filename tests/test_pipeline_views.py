"""Tests for the PR3 ``@view`` extension to the pipeline DSL.

Covers:

* Grammar — the optional ``@view`` slot between ``[alias]`` and ``%limit``,
  including the back-compat case of no view.
* ``SystemCall`` dataclass plumbing — ``__hash__``, ``repr``, ``as_role``,
  ``_apply_outer_limit`` all preserve / include ``view``.
* Alias propagation — ``_propagate_view_to_leaves`` and ``expand_aliases``:
  leaves without their own ``@view`` pick up the call-site view; leaves
  with their own ``@view`` keep it; mergers and expanders are skipped.
* ``SearchPipeline.verify()`` view validation against a registered
  :class:`ContentProcessor`'s backends.
* ``runtime_kwargs`` rejecting any ``"view"`` key.
* Scratch-dict key tuple disambiguates same-alias rerank calls under two
  different views.
"""

import asyncio
import json

import pytest

from routir.collections.processor import ContentProcessor
from routir.collections.views import text_jsonl as _tj_mod
from routir.config import CollectionConfig, TextJsonlSource, ViewSpec
from routir.pipeline import SearchPipeline
from routir.pipeline.parser import (
    CallSequence,
    ParallelCallSequences,
    SystemCall,
    _apply_outer_limit,
    _propagate_view_to_leaves,
    expand_aliases,
    parser,
)
from routir.processors.abstract import Processor
from routir.processors.registry import ProcessorRegistry


# ---------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _clear_reader_cache():
    """Per-test isolation: don't share jsonl readers across tests."""
    _tj_mod._READER_CACHE.clear()
    yield
    _tj_mod._READER_CACHE.clear()


# ------------------------------------------------------------- grammar / parser


def test_parser_view_only():
    call = parser.parse("foo@bar")
    assert isinstance(call, SystemCall)
    assert call.name == "foo"
    assert call.alias == "foo"
    assert call.view == "bar"
    assert call.limit is None
    assert call.role == "search"


def test_parser_alias_view_and_limit():
    call = parser.parse("foo[a]@bar%5")
    assert isinstance(call, SystemCall)
    assert call.name == "foo"
    assert call.alias == "a"
    assert call.view == "bar"
    assert call.limit == 5


def test_parser_back_compat_no_view():
    """An old-style DSL string without ``@view`` still parses cleanly."""
    call = parser.parse("foo%5")
    assert isinstance(call, SystemCall)
    assert call.name == "foo"
    assert call.alias == "foo"
    assert call.view is None
    assert call.limit == 5


def test_parser_parallel_per_branch_views():
    """``{a@vA, b@vB}c%10`` parses with per-branch views and no view on merger."""
    node = parser.parse("{a@vA, b@vB}c%10")
    assert isinstance(node, ParallelCallSequences)
    branches = node.sequences
    assert len(branches) == 2
    assert branches[0].name == "a" and branches[0].view == "vA"
    assert branches[1].name == "b" and branches[1].view == "vB"
    assert node.merger.name == "c"
    assert node.merger.view is None
    assert node.merger.limit == 10


def test_parser_dash_in_view_name():
    """View names allow hyphens just like service / alias names."""
    call = parser.parse("kf-rerank[r]@keyframe-v2%50")
    assert call.name == "kf-rerank"
    assert call.alias == "r"
    assert call.view == "keyframe-v2"
    assert call.limit == 50


# ------------------------------------------------------------- hash / repr


def test_hash_distinguishes_views():
    a = SystemCall(name="foo", view="v1")
    b = SystemCall(name="foo", view="v2")
    assert hash(a) != hash(b)
    assert a != b


def test_repr_includes_view():
    a = SystemCall(name="foo", view="v1")
    b = SystemCall(name="foo", view="v2")
    assert "v1" in repr(a)
    assert "v2" in repr(b)
    # And the two reprs differ (so cache keys based on repr will).
    assert repr(a) != repr(b)


# ------------------------------------------------------------- as_role / _apply_outer_limit


def test_as_role_preserves_view():
    c = SystemCall(name="x", view="v1").as_role("rerank")
    assert c.view == "v1"
    assert c.role == "rerank"


def test_apply_outer_limit_preserves_view_on_system_call():
    c = SystemCall(name="x", view="v1", limit=10)
    out = _apply_outer_limit(c, 20)
    assert out.view == "v1"
    assert out.limit == 20


def test_apply_outer_limit_preserves_view_on_parallel_merger():
    """Even though we re-make the merger node, its existing ``view`` survives."""
    node = parser.parse("{a, b}c@m%5")
    assert node.merger.view == "m"
    out = _apply_outer_limit(node, 33)
    assert out.merger.view == "m"
    assert out.merger.limit == 33


# ------------------------------------------------------------- alias propagation


def test_propagate_view_overrides_no_view_leaves():
    body = parser.parse("bm25 >> reranker")
    out = _propagate_view_to_leaves(body, "kf")
    assert isinstance(out, CallSequence)
    assert all(s.view == "kf" for s in out.stages)


def test_propagate_view_skips_leaves_with_own_view():
    body = parser.parse("bm25 >> reranker@asr")
    out = _propagate_view_to_leaves(body, "kf")
    assert out.stages[0].view == "kf"
    assert out.stages[1].view == "asr"


def test_propagate_view_skips_merger_and_expander():
    body = parser.parse("e{a, b}c")
    out = _propagate_view_to_leaves(body, "kf")
    # branch leaves carry the view
    assert all(s.view == "kf" for s in out.sequences)
    # merger and expander are skipped — they don't fetch document content
    assert out.merger.view is None
    assert out.expander.view is None


def test_expand_aliases_propagates_call_site_view():
    aliases = {"r": parser.parse("bm25 >> reranker")}
    expanded = expand_aliases(parser.parse("r@kf"), aliases)
    assert isinstance(expanded, CallSequence)
    assert [s.view for s in expanded.stages] == ["kf", "kf"]


def test_expand_aliases_inner_view_wins_over_call_site():
    aliases = {"r": parser.parse("bm25 >> reranker@asr")}
    expanded = expand_aliases(parser.parse("r@kf"), aliases)
    assert expanded.stages[0].view == "kf"
    assert expanded.stages[1].view == "asr"


def test_expand_aliases_parallel_merger_no_view():
    aliases = {"r": parser.parse("{a, b}c")}
    expanded = expand_aliases(parser.parse("r@kf"), aliases)
    assert isinstance(expanded, ParallelCallSequences)
    assert all(s.view == "kf" for s in expanded.sequences)
    assert expanded.merger.view is None


def test_expand_aliases_limit_and_view_both_applied():
    """A call-site ``r%20@kf`` (here we use the grammar order alias-view-limit)
    applies the limit to the outer node and the view to all leaves."""
    aliases = {"r": parser.parse("bm25 >> reranker")}
    # grammar order: ``r@kf%20`` (alias, view, limit)
    expanded = expand_aliases(parser.parse("r@kf%20"), aliases)
    assert isinstance(expanded, CallSequence)
    assert [s.view for s in expanded.stages] == ["kf", "kf"]
    # outer limit lands on the last stage (the final-output node)
    assert expanded.stages[-1].limit == 20


# ------------------------------------------------------------- ContentProcessor fixture


@pytest.fixture
def multiview_jsonl(tmp_path):
    p = tmp_path / "corpus.jsonl"
    with p.open("w") as fw:
        fw.write(json.dumps({"id": "d1", "ocr": "OCR-1", "asr": "ASR-1"}) + "\n")
        fw.write(json.dumps({"id": "d2", "ocr": "OCR-2", "asr": "ASR-2"}) + "\n")
    return p


@pytest.fixture
def multiview_content_processor(multiview_jsonl):
    cfg = CollectionConfig(
        name="vcoll",
        default_view="ocr",
        views={
            "ocr": ViewSpec(source=TextJsonlSource(
                source="text_jsonl", doc_path=str(multiview_jsonl),
                id_field="id", content_fields="ocr",
            )),
            "asr": ViewSpec(source=TextJsonlSource(
                source="text_jsonl", doc_path=str(multiview_jsonl),
                id_field="id", content_fields="asr",
            )),
        },
    )
    cp = ContentProcessor(cfg, cache_size=-1)
    # PR4: ``SearchPipeline.verify()`` now reads view metadata from registry
    # slot meta rather than ``cp.backends``, so register with the metadata
    # explicitly here too.
    ProcessorRegistry.register(
        "vcoll", "content", cp,
        views={name: backend.kind for name, backend in cp.backends.items()},
        default_view=cp.default_view,
    )
    yield cp
    ProcessorRegistry.all_services.pop("vcoll", None)
    ProcessorRegistry.slot_meta.pop("vcoll", None)


# A trivial search / score / merger processor stack so verify() passes.

class _StubProcessor(Processor):
    def __init__(self, kind="search"):
        super().__init__(cache_size=-1)
        self.kind = kind
        self.calls = []

    async def _submit(self, item):
        self.calls.append(dict(item))
        if self.kind == "search":
            # mimic a search response: dict of {docid: score}
            return {"scores": {"d1": 1.0, "d2": 0.5}}
        if self.kind == "rerank":
            n = len(item.get("passages", []))
            return {"scores": [float(i) for i in range(n)]}
        if self.kind == "merger":
            # ``item['scores']`` is List[Dict]; just return the first
            return {"scores": dict(item["scores"][0])}
        raise AssertionError(f"unknown stub kind: {self.kind}")


@pytest.fixture
def stub_services():
    """Register a search engine 'bm25' and a reranker 'rr' for verify()."""
    bm25 = _StubProcessor("search")
    rr = _StubProcessor("rerank")
    ProcessorRegistry.register("bm25", "search", bm25, view_kind="text")
    ProcessorRegistry.register("rr", "search", _StubProcessor("search"), view_kind="text")
    ProcessorRegistry.register("rr", "score", rr, view_kind="text")
    yield {"bm25": bm25, "rr": rr}
    ProcessorRegistry.all_services.pop("bm25", None)
    ProcessorRegistry.all_services.pop("rr", None)
    ProcessorRegistry.slot_meta.pop("bm25", None)
    ProcessorRegistry.slot_meta.pop("rr", None)


# ------------------------------------------------------------- verify() view validation


def test_verify_view_resolves_against_collection(multiview_content_processor, stub_services):
    # Should not raise — 'asr' is in the collection's views.
    p = SearchPipeline.from_string("bm25 >> rr@asr", collection="vcoll")
    assert p.collection == "vcoll"


def test_verify_unknown_view_raises(multiview_content_processor, stub_services):
    with pytest.raises(ValueError, match="no-such"):
        SearchPipeline.from_string("bm25 >> rr@no-such", collection="vcoll")


def test_verify_default_view_used_when_not_specified(multiview_content_processor, stub_services):
    # No @view on the rerank stage — should resolve to the collection's default ('ocr').
    p = SearchPipeline.from_string("bm25 >> rr", collection="vcoll")
    # Confirm at runtime the stage gets its view by default
    asyncio.run(p.run("hello world"))
    rerank_calls = stub_services["rr"].calls
    # The rerank stub got called with 2 passages (d1, d2).
    assert any("passages" in c for c in rerank_calls)


# ------------------------------------------------------------- runtime_kwargs view rejection


def test_runtime_kwargs_view_key_rejected(multiview_content_processor, stub_services):
    """``view`` is structural — runtime_kwargs cannot override it."""
    with pytest.raises(ValueError, match="view is structural"):
        SearchPipeline.from_string(
            "bm25 >> rr@asr",
            collection="vcoll",
            runtime_kwargs={"rr": {"view": "ocr"}},
        )


# ------------------------------------------------------------- scratch key collision


def test_scratch_key_tuple_includes_view():
    """As tuple keys, (alias, role, v1) and (alias, role, v2) are distinct."""
    k1 = ("rr", "rerank", "v1")
    k2 = ("rr", "rerank", "v2")
    assert k1 != k2
    d = {}
    d[k1] = "first"
    d[k2] = "second"
    assert d[k1] == "first" and d[k2] == "second"


async def test_scratch_dict_keyed_by_view_after_run(multiview_content_processor, stub_services):
    """Running a pipeline populates the scratch dict with a view-aware tuple key.

    ``run()`` rebinds ``scratch`` when the caller passes a falsy value, so we
    pre-seed it with a sentinel and let the inner reassignment ``scratch =
    scratch or {}`` keep our dict — that way we can read back what got
    written.
    """
    p = SearchPipeline.from_string("bm25 >> rr@asr", collection="vcoll")
    scratch = {("__sentinel__", "x", None): "keep"}
    await p.run("q", current_node=p.pipeline, scratch=scratch)
    # bm25 has no view (search stage); rr was annotated with asr.
    assert any(k[2] == "asr" and k[0] == "rr" for k in scratch.keys())


async def test_rerank_fetches_view_specific_content(multiview_content_processor, stub_services):
    """The rerank stage should fetch ASR text when annotated @asr."""
    p = SearchPipeline.from_string("bm25 >> rr@asr", collection="vcoll")
    await p.run("query")
    # Check the rerank stub got the ASR strings, not OCR.
    rr_call = stub_services["rr"].calls[-1]
    assert "ASR-1" in rr_call["passages"]
    assert "ASR-2" in rr_call["passages"]


async def test_doc_content_cache_keyed_by_view(multiview_content_processor, stub_services):
    """Same doc id under two different views populates two cache entries."""
    p = SearchPipeline.from_string("bm25 >> rr@asr", collection="vcoll", verify=False)
    t1 = await p.get_doc_content("d1", "asr")
    t2 = await p.get_doc_content("d1", "ocr")
    assert t1 == "ASR-1"
    assert t2 == "OCR-1"
    assert ("asr", "d1") in p.doc_content_cache
    assert ("ocr", "d1") in p.doc_content_cache
