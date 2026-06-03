#!/usr/bin/env python3
"""
A thin web UI for querying a RoutIR endpoint and browsing video results as
keyframe strips.

This is the interactive sibling of scripts/query.py: instead of writing a JSONL
run file, it serves a small web page where you type a query + a RoutIR pipeline
string, hit search, and see the ranked video IDs alongside their keyframes.

The server keeps NO per-request state. Every search re-queries the endpoint
(RoutIR's server-side query cache makes repeated identical queries cheap), and
page navigation / "show all" toggles happen entirely in the browser, loading at
most one page (20 results) of keyframes at a time.

Result content is pulled from RoutIR itself, not from local files. Collections
carry typed *views* (multivent exposes keyframes, the video, and ASR/OCR
transcripts), and each result card shows one tab per view of the selected
collection. The keyframe view is forced first and is the default tab; the rest
follow in /avail order. A tab's content loads lazily from RoutIR's /content the
first time it's opened (so a result still costs just one keyframe fetch up
front), and what it renders depends on what the bytes actually are — /avail
labels every view "bytes" regardless, so we sniff each payload:
  * JPEG/PNG  -> image strip (N-frame preview + "show all" + lightbox)
  * MP4/WebM  -> an HTML5 <video> player
  * UTF-8 JSON/JSONL (ASR/OCR) -> readable transcript/OCR text, with a toggle to
                                  the raw JSON; ASR shows per-segment timestamps,
                                  OCR shows per-frame lines
  * anything else -> a download link
There is no --keyframes flag or local tar indexing. The keyframe tab is
auto-detected per collection (the default view if it is bytes, else a
"keyframe"/"frame"/"image"-named bytes view).

Bytes views are only served over gRPC (REST /content is text-only), so the
client must be able to reach the endpoint's gRPC port. With the default
transport="auto" the client auto-discovers gRPC via /avail, so install the grpc
extra (routir[grpc]) when running. RoutIR caches /content server-side, so the
UI keeps no media cache of its own; each per-frame image request just re-asks
for the (cheap, cached) part list.

Result doc_ids returned by the endpoint are exactly the collection's doc ids
(e.g. "XM5xOIzL_vSkGAKR_0000"); the keyframe parts come back as an ordered list
with no timestamps, so frames are labelled by ordinal (#1, #2, ...).

Run with uvx so the routir client + quart/hypercorn deps are available without
touching a conda env (routir already depends on quart + hypercorn). Use
--with-editable, not --with 'routir @ .': uv wheel-caches a non-editable path
dep and will silently keep serving a STALE build (e.g. a gRPC avail() from
before collections were added, leaving the collection dropdown empty), whereas
an editable install always reflects the current checkout:

    uvx --with-editable '.[grpc]' python scripts/webui.py \
        --endpoint grpc://compute01:50051 \
        --port 8080

Then open http://<this-host>:8080 in a browser. --endpoint is only the default;
each browser carries its own endpoint in the URL (editable via the header's
"edit" link), so concurrent users can target different endpoints at once. The
server pools one AsyncClient per distinct endpoint. Available engines,
collections (with their views) and pipeline aliases are shown in a panel
(fetched from /avail); refresh the page to update it. The collection dropdown is
populated from /avail's collection map. Pass --descriptions <json> (a
{service-name: description} map) to add a collapsible "Descriptions" panel
listing every service this endpoint exposes with its blurb beside it (services
absent from the file are still listed, with no text).
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from quart import Quart, Response, abort, g, request
from routir.client.client import AsyncClient

log = logging.getLogger("webui")

# Reuse RoutIR's real pipeline grammar so live validation / autocomplete never
# drift from what the server actually parses. A transformer-free LALR parser lets
# us drive it interactively (accepts() = the terminals valid at the cursor).
try:
    from lark import Lark
    from lark.exceptions import UnexpectedInput
    from lark.lexer import PatternStr
    from routir.pipeline.parser import PIPELINE_GRAMMAR

    GRAMMAR_PARSER = Lark(PIPELINE_GRAMMAR, parser="lalr")
    # accepts() yields terminal *names* (COMMA, __ANON_0, ...); map the literal
    # ones back to the text they match so we can suggest the symbol itself
    # without hardcoding lark's auto-generated names (e.g. ">>" -> __ANON_0).
    TERM_LITERAL = {t.name: t.pattern.value for t in GRAMMAR_PARSER.terminals if isinstance(t.pattern, PatternStr)}
except Exception:  # noqa: BLE001 - if anything is off, the feature just disables
    GRAMMAR_PARSER = None
    UnexpectedInput = Exception
    TERM_LITERAL = {}

# Human-readable hint shown next to each suggested operator/punctuation symbol.
# LBRACE is described contextually (fusion vs. query-expansion) in _symbol_suggestions.
_SYMBOL_HELP = {
    "%": "cut: keep top N",
    ">>": "rerank: pipe into the next stage",
    ",": "add a parallel branch",
    "}": "close fusion group (a merger follows)",
    "[": "name this stage with an alias",
    "]": "close the alias",
}
# Display order for symbols when several are valid at once.
_SYMBOL_ORDER = ["%", ">>", "{", "}", ",", "[", "]"]
# The service name of the system_call immediately left of the cursor (the engine
# that a following "{" would use as a query expander), allowing for [alias]/%N.
_PRECEDING_NAME = re.compile(r"([A-Za-z][A-Za-z0-9_\-]*)\s*(?:\[[A-Za-z0-9_\-]*\])?\s*(?:%\d+)?\s*$")

# Trailing identifier-ish run (a NAME or NUMBER being typed) just before the cursor.
_NAME_TAIL = re.compile(r"[A-Za-z0-9_\-]*$")

# A leading pure-letter prefix ("full-", "lite-", ...) stripped when looking up a
# service's description, so a single line keyed by the engine (e.g. "qwen3-vl-8b")
# matches every prefixed variant ("full-qwen3-vl-8b"). The prefix must be ALL
# letters: "qwen3asr-emb8b" has a digit before its first "-", so it isn't treated
# as a prefix and is matched whole (it can't collapse to "emb8b").
_DESC_PREFIX = re.compile(r"^[A-Za-z]+-(.+)$")

# Set in main(); read by the request handlers.
ARGS = None
CLIENTS = {}  # endpoint -> routir AsyncClient (per-endpoint pool; see _client_for)
DESCRIPTIONS = {}  # service name -> short description, loaded from --descriptions

# Keyframe-ish bytes-view name fragments, in preference order, used to pick which
# of a collection's bytes views holds displayable keyframes (see _media_view).
_MEDIA_VIEW_PREFS = ("keyframe", "frame", "thumb", "image", "img")


def _media_view(avail, collection):
    """Pick the bytes view of ``collection`` that holds keyframes, or None.

    /avail's collection map is ``{collection: {"default": view, "views": {name: kind}}}``.
    We render only ``bytes`` views as images; among them prefer a keyframe-ish
    name (keyframe/frame/thumb/image), then the default view if it's bytes, then
    the first bytes view. None means the collection has nothing to show.
    """
    info = (avail.get("collection") or {}).get(collection)
    if not isinstance(info, dict):
        return None
    views = info.get("views") or {}
    default = info.get("default")
    bytes_views = [v for v, kind in views.items() if kind == "bytes"]
    if not bytes_views:
        return None

    def rank(v):
        lv = v.lower()
        for i, frag in enumerate(_MEDIA_VIEW_PREFS):
            if frag in lv:
                return i
        return len(_MEDIA_VIEW_PREFS)

    # Lowest preference rank wins; the default view breaks ties.
    return min(bytes_views, key=lambda v: (rank(v), 0 if v == default else 1))


async def _fetch_media(endpoint, collection, view, vid):
    """Return the list[bytes] of keyframe parts for one doc id from RoutIR.

    RoutIR caches /content server-side, so we don't cache here — each per-frame
    image request just re-asks for the (cheap, cached) part list. Text views (no
    "data") come back as an empty list, i.e. nothing to show.
    """
    client = _client_for(endpoint)
    result = await client.content(collection=collection, id=vid, view=view)
    return result.get("data") or []


def _accepts(prefix):
    """Terminal names the grammar will accept right after ``prefix`` (or empty)."""
    try:
        ip = GRAMMAR_PARSER.parse_interactive(prefix)
        ip.exhaust_lexer()
        return ip.accepts()
    except Exception:  # noqa: BLE001 - prefix not a parseable pipeline prefix
        return set()


def _innermost_brace_branches(s):
    """Comma count of the innermost still-open ``{`` in ``s`` (None if not inside one).

    Used so we only suggest ``}`` once a fusion group holds more than one branch:
    ``{a`` has 0 (one branch -> fusing nothing, no ``}`` yet), ``{a, b`` has 1.
    Commas only ever appear inside braces in this grammar, so a plain scan suffices.
    """
    stack = []
    for ch in s:
        if ch == "{":
            stack.append(0)
        elif ch == "}":
            if stack:
                stack.pop()
        elif ch == "," and stack:
            stack[-1] += 1
    return stack[-1] if stack else None


def _symbol_suggestions(accepts, left):
    """Map the operator/punctuation terminals valid at the cursor to suggestions.

    ``accepts`` is the terminal set for the full left context (so a just-finished
    NAME is treated as a complete token and we surface what may follow it). Each
    suggestion is ``{value, type}`` where value is the literal symbol to insert.
    """
    syms = []
    for term in accepts:
        lit = TERM_LITERAL.get(term)
        if lit is None or lit == "{" or term in ("NAME", "NUMBER"):
            continue  # "{" handled below; names/numbers aren't symbols
        if lit == "}":
            # Only worth closing once the fusion group has >1 branch.
            if (_innermost_brace_branches(left) or 0) < 1:
                continue
        syms.append({"value": lit, "type": _SYMBOL_HELP.get(lit, "")})
    if "LBRACE" in accepts:
        # "{" after a complete service is query *expansion* (that engine makes
        # sub-queries) and is only meaningful if the engine is a decompose_query
        # service; elsewhere "{" just opens a parallel *fusion* group, always
        # valid. We tag the expander case with its engine so the browser can drop
        # it when the engine can't actually expand (it only has the avail lists).
        prev = left.rstrip()
        after_call = bool(prev) and (prev[-1].isalnum() or prev[-1] in "_-]")
        if after_call:
            mo = _PRECEDING_NAME.search(prev)
            syms.append(
                {
                    "value": "{",
                    "type": "query expansion: this engine makes sub-queries",
                    "expander": mo.group(1) if mo else "",
                }
            )
        else:
            # A bare "{" opens a fusion group; right after "{" or "," it's a nested
            # fusion (a branch that is itself a fusion), which the grammar allows.
            syms.append({"value": "{", "type": "fusion: open parallel branches"})
    syms.sort(key=lambda s: _SYMBOL_ORDER.index(s["value"]) if s["value"] in _SYMBOL_ORDER else 99)
    return syms


def _grammar_complete(text, pos):
    """Grammar-driven validation + next-token classification at the cursor.

    Returns a dict with:
      valid       -- True/False/None(empty) for the whole string (the badge).
      error       -- first line of the parse error when invalid.
      error_col   -- 1-based column of the error, if known.
      slot        -- semantic category the next NAME would fill, one of
                     "stage" (search engine / expander / alias),
                     "rerank" (after ">>"), "merger" (after "}"),
                     "alias" (after "["; user-defined, no suggestions),
                     "number" (after "%"), or None.
      symbols     -- valid operator/punctuation completions at the cursor, each
                     {value, type}; inserted at the cursor, not replacing partial.
      partial     -- the in-progress token under the cursor.
      replace_from-- index where a name/number suggestion should be inserted.

    The grammar uses one NAME terminal for every service, so it tells us *a name
    is valid here*; the slot is derived from the operator immediately preceding
    the cursor, and the caller maps the slot to the right /avail category. Symbol
    suggestions come from the same grammar so they never drift from what parses.
    """
    out = {"valid": None, "error": None, "error_col": None, "slot": None, "symbols": [], "partial": "", "replace_from": pos}
    if GRAMMAR_PARSER is None:
        return out

    # Whole-string validity drives the badge; empty input is neutral.
    if text.strip():
        try:
            GRAMMAR_PARSER.parse(text)
            out["valid"] = True
        except UnexpectedInput as e:
            out["valid"] = False
            out["error"] = str(e).splitlines()[0]
            out["error_col"] = getattr(e, "column", None)
        except Exception as e:  # noqa: BLE001 - any lark error -> invalid
            out["valid"] = False
            out["error"] = str(e).splitlines()[0]

    # Split the cursor's left context into committed text + the token being typed.
    left = text[:pos]
    partial = _NAME_TAIL.search(left).group(0)
    committed = left[: len(left) - len(partial)]
    out["partial"] = partial
    out["replace_from"] = pos - len(partial)

    # Names/numbers complete the in-progress token, so classify from the committed
    # prefix (partial stripped); symbols follow a *finished* token, so classify
    # from the full left context (partial treated as a complete token).
    accepts_name = _accepts(committed)
    tail = committed.rstrip()
    last = tail[-1] if tail else ""
    if "NAME" in accepts_name:
        if tail.endswith(">>"):
            out["slot"] = "rerank"
        elif last == "}":
            out["slot"] = "merger"
        elif last == "[":
            out["slot"] = "alias"
        else:
            out["slot"] = "stage"
    elif "NUMBER" in accepts_name:
        out["slot"] = "number"

    out["symbols"] = _symbol_suggestions(_accepts(left), left)
    return out


def _evenly_spaced(values, n):
    """Pick n evenly-spaced items (by position) from a sorted list, in order."""
    L = len(values)
    if n >= L or n <= 0:
        return list(values)
    if n == 1:
        return [values[L // 2]]
    idxs = sorted({round(i * (L - 1) / (n - 1)) for i in range(n)})
    return [values[i] for i in idxs]


# --------------------------------------------------------------------------- #
# HTTP handlers
# --------------------------------------------------------------------------- #
app = Quart(__name__)


@app.before_request
async def _log_request_start():
    # Stash a start time so the after-request hook can report how long each
    # request took (most of which is the upstream RoutIR round-trip on / and /view).
    g.req_start = time.perf_counter()


@app.after_request
async def _log_request_end(response):
    """Log every request served by the UI: method, path+query, status, elapsed.

    This surfaces what the UI is doing in the console — page loads (/), the
    autocomplete polls (/complete), and the lazy media fetches each result card
    makes (/view, /part) — so you can watch its activity and spot slow upstream
    calls. The query string is included since it carries the query/pipeline.
    """
    qs = request.query_string.decode("utf-8", "replace")
    path = f"{request.path}?{qs}" if qs else request.path
    started = g.get("req_start")
    took = f" {(time.perf_counter() - started) * 1000:.0f}ms" if started is not None else ""
    log.info("%s %s -> %s%s", request.method, path, response.status_code, took)
    return response


def _media_params():
    """Pull (endpoint, collection, view) out of the request's query string.

    The page bakes these in (the per-request endpoint, the searched collection,
    and its auto-picked keyframe view) and passes them on every frame request, so
    the handlers stay stateless and concurrent users/endpoints don't collide.
    """
    endpoint = (request.args.get("endpoint") or "").strip() or ARGS.endpoint
    collection = (request.args.get("collection") or "").strip()
    view = (request.args.get("view") or "").strip()
    return endpoint, collection, view


def _sniff_type(parts):
    """Classify a /content bytes payload by its first part, returning (type, mime).

    /avail reports every multivent view as ``bytes`` regardless of what it holds
    (JPEG keyframes, an MP4, or UTF-8 JSON/JSONL transcripts), so we detect the
    real kind from magic bytes: image / video / text / binary.
    """
    head = parts[0] if parts else b""
    if head[:3] == b"\xff\xd8\xff":
        return "image", "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image", "image/png"
    if head[4:8] == b"ftyp":
        return "video", "video/mp4"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video", "video/webm"
    try:
        head.decode("utf-8")
        return "text", "text/plain; charset=utf-8"
    except UnicodeDecodeError:
        return "binary", "application/octet-stream"


def _extract_readable(raw):
    """Turn an ASR/OCR JSON(L) payload into human-readable text (raw kept for toggle).

    ASR views are a single JSON object with ``transcript.segments`` ([{start, end,
    text}]); OCR views are JSONL with one ``{"frame","txt"}`` per sampled frame.
    Anything else falls back to pretty-printed JSON, or the raw string verbatim.
    """
    raw = raw.strip()
    if not raw:
        return ""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    try:
        objs = [json.loads(ln) for ln in lines] if len(lines) > 1 else [json.loads(raw)]
    except (ValueError, TypeError):
        return raw  # not JSON -> the decoded text is already the readable form

    # ASR: one object carrying transcript.segments.
    if len(objs) == 1 and isinstance(objs[0], dict) and "transcript" in objs[0]:
        o = objs[0]
        lang = (o.get("language") or {}).get("detected") or "?"
        segs = (o.get("transcript") or {}).get("segments") or []
        body = [f"[{s.get('start', 0):.1f}–{s.get('end', 0):.1f}s] {t}" for s in segs if (t := (s.get("text") or "").strip())]
        return f"language: {lang}\n\n" + ("\n".join(body) if body else "(no speech detected)")

    # OCR: JSONL of per-frame text.
    if objs and all(isinstance(o, dict) and "frame" in o for o in objs):
        body = [f"{o['frame']}: {t}" for o in objs if (t := (o.get("txt") or o.get("cleaned") or o.get("raw") or "").strip())]
        return "\n".join(body) if body else "(no text detected)"

    # Some other JSON shape -> just pretty-print it.
    if len(objs) == 1:
        return json.dumps(objs[0], indent=2, ensure_ascii=False)
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in objs)


@app.route("/view/<vid>")
async def view_meta(vid):
    """Describe one view of one doc so the browser can render its tab.

    Returns ``{type, n, mime}`` plus, for images, the 0-based part indices to show
    (``all`` + an evenly-spaced ``preview``), or for text the readable rendering
    (``text``) alongside the raw payload (``raw``) for the show-raw toggle. Video
    and binary carry only type/n/mime; their bytes stream from /part.
    """
    endpoint, collection, view = _media_params()
    if not collection or not view:
        return {"type": "empty", "n": 0}
    try:
        parts = await _fetch_media(endpoint, collection, view, vid)
    except Exception as e:  # noqa: BLE001 - surface fetch errors in the tab, don't 500
        return {"type": "error", "error": str(e), "n": 0}
    if not parts:
        return {"type": "empty", "n": 0}
    typ, mime = _sniff_type(parts)
    out = {"type": typ, "n": len(parts), "mime": mime}
    if typ == "image":
        idxs = list(range(len(parts)))
        out["all"] = idxs
        out["preview"] = _evenly_spaced(idxs, ARGS.frames)
    elif typ == "text":
        out["raw"] = b"".join(parts).decode("utf-8", "replace")
        out["text"] = _extract_readable(out["raw"])
    return out


def _ranged_response(body, mime):
    """Serve ``body`` honoring a single HTTP Range request.

    Video seeking only works if we advertise ``Accept-Ranges: bytes`` and answer a
    ``Range:`` header with ``206 Partial Content`` + ``Content-Range`` for the
    requested slice; a plain 200 of the whole file leaves the timeline unseekable.
    """
    total = len(body)
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "max-age=3600"}
    rng = request.headers.get("Range", "")
    start, end = 0, total - 1
    partial = False
    if rng.startswith("bytes="):
        spec = rng[len("bytes=") :].split(",", 1)[0].strip()  # only the first range
        lo, _, hi = spec.partition("-")
        try:
            if lo == "":  # suffix range: last N bytes
                n = int(hi)
                if n > 0:
                    start, end, partial = max(0, total - n), total - 1, True
            else:
                start = int(lo)
                end = min(int(hi), total - 1) if hi else total - 1
                if start > end or start >= total:  # unsatisfiable
                    return Response(status=416, headers={**headers, "Content-Range": f"bytes */{total}"})
                partial = True
        except ValueError:
            partial = False
    if partial:
        chunk = body[start : end + 1]
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        return Response(chunk, status=206, mimetype=mime, headers=headers)
    return Response(body, mimetype=mime, headers=headers)


@app.route("/part/<vid>/<int:idx>")
async def part(vid, idx):
    """Serve the idx-th raw part of a view (image frame, video file, ...) with its
    sniffed mime, so an <img>/<video> src can point straight at it. Video parts are
    served with Range support (see _ranged_response) so the player can seek."""
    endpoint, collection, view = _media_params()
    if not collection or not view:
        abort(404)
    try:
        parts = await _fetch_media(endpoint, collection, view, vid)
    except Exception:  # noqa: BLE001
        abort(404)
    if idx < 0 or idx >= len(parts):
        abort(404)
    _, mime = _sniff_type([parts[idx]])
    return _ranged_response(parts[idx], mime)


@app.route("/complete")
async def complete():
    """Validate a (partial) pipeline and classify the token at the cursor.

    Query args: pipeline=<text>, pos=<cursor index>. The browser maps the
    returned slot to the right /avail category for the suggestion list.
    """
    text = request.args.get("pipeline", "")
    try:
        pos = int(request.args.get("pos", len(text)))
    except ValueError:
        pos = len(text)
    pos = max(0, min(pos, len(text)))
    return _grammar_complete(text, pos)


def _make_client(endpoint):
    """Build an AsyncClient for ``endpoint`` using the rest of ARGS for config."""
    # An explicit --grpc-endpoint override only applies to the endpoint it was
    # given for (the startup default); any other endpoint auto-derives its target.
    grpc_endpoint = ARGS.grpc_endpoint if endpoint == ARGS.endpoint else None
    return AsyncClient(
        endpoint=endpoint,
        grpc_endpoint=grpc_endpoint,
        api_key=ARGS.api_key,
        transport=ARGS.transport,
        timeout=ARGS.timeout,
    )


def _client_for(endpoint):
    """Return a shared AsyncClient for ``endpoint``, creating it on first use.

    The endpoint is per-request (carried in the URL), so two users can hit the
    UI against different endpoints at the same time. Clients are pooled by
    endpoint and reused — the server still keeps no per-*user* state, just a
    small shared cache of connections. AsyncClient construction is synchronous
    and Quart runs on a single event loop, so the plain-dict insert is race-free.
    """
    client = CLIENTS.get(endpoint)
    if client is None:
        client = _make_client(endpoint)
        CLIENTS[endpoint] = client
    return client


@app.route("/")
async def index_page():
    # The endpoint is per-request: it rides in the URL (the header's edit form
    # sets it, the search form preserves it) and defaults to --endpoint. This is
    # what lets concurrent users each target a different endpoint.
    endpoint = (request.args.get("endpoint") or "").strip() or ARGS.endpoint
    client = _client_for(endpoint)

    q = (request.args.get("q") or "").strip()
    # The pipeline runs verbatim. We don't force a trailing %N, so a pipeline with
    # no cut falls back to RoutIR's own default result size.
    custom_pipeline = (request.args.get("pipeline") or ARGS.pipeline or "").strip()
    # None means the collection wasn't in the URL (first load); "" means the user
    # explicitly picked "(none)". Resolve the default after /avail below so we can
    # fall back to the first available collection (e.g. multivent) when present.
    collection = request.args.get("collection")

    # /avail powers the engines panel, the collection dropdown, and the engine
    # dropdown. Its "search" list is exactly the first-stage retrievers (rerankers
    # live under "score", fusion under "fuse"), so the dropdown shows search
    # engines only. Refresh the page to refresh all of this.
    avail, avail_err = {}, None
    try:
        avail = await client.avail()
    except Exception as e:  # noqa: BLE001
        avail_err = str(e)

    if collection is None:
        # Default (mirrors the engine dropdown defaulting to the first search
        # engine): --collection if set, else the first collection /avail lists.
        collection = ARGS.collection or next(iter(avail.get("collection") or {}), "")

    search_engines = list(avail.get("search", []))
    # Engine selection drives the pipeline: a named first-stage engine runs as a
    # bare "<engine>" pipeline (custom box disabled), so RoutIR's default result
    # size applies; the sentinel __custom__ uses whatever is typed in the box.
    engine = request.args.get("engine")
    if engine is None:
        engine = search_engines[0] if search_engines else "__custom__"

    if engine == "__custom__":
        effective_pipeline = custom_pipeline
    else:
        effective_pipeline = engine

    ranked, search_err = [], None
    if q and effective_pipeline:
        try:
            payload = await client.pipeline(effective_pipeline, q, collection=collection or None)
            scores = payload.get("scores") or {}
            ranked = [
                {"rank": i, "id": did, "score": float(sc)}
                for i, (did, sc) in enumerate(sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1)
            ]
        except Exception as e:  # noqa: BLE001
            search_err = str(e)
    elif q and not effective_pipeline:
        search_err = "No pipeline to run: choose an engine or enter a custom pipeline."

    return render_page(
        q=q,
        endpoint=endpoint,
        custom_pipeline=custom_pipeline,
        effective_pipeline=effective_pipeline,
        collection=collection,
        media_view=_media_view(avail, collection),
        active_tab=(request.args.get("tab") or ""),
        avail=avail,
        avail_err=avail_err,
        search_engines=search_engines,
        selected_engine=engine,
        ranked=ranked,
        search_err=search_err,
    )


def _syntax_help(avail):
    """Build the pipeline-syntax cheatsheet, using real engines where available."""
    search = list(avail.get("search") or [])
    score = list(avail.get("score") or [])
    fuse = list(avail.get("fuse") or [])
    expander = list(avail.get("decompose_query") or [])

    e1 = search[0] if search else "EXAMPLE-ENGINE"
    e2 = search[1] if len(search) > 1 else "EXAMPLE-ENGINE-2"
    rr = score[0] if score else "EXAMPLE-RERANKER"
    fz = fuse[0] if fuse else "RRF"
    ex = expander[0] if expander else "EXAMPLE-EXPANDER"

    # (symbol, description, example) — examples are escaped at render time.
    items = [
        ("%N", "Each stage's own cut: keep its top N results.", f"{e1}%1000"),
        ("A &gt;&gt; B", "Rerank: run A, then re-score A's hits with B.", f"{e1}%1000 >> {rr}%20"),
        ("{A, B}M", "Fusion: run A and B in parallel, combine with merger M.", f"{{{e1}%1000, {e2}%1000}}{fz}%100"),
        (
            "E{A, B}M",
            "Query expansion: E makes sub-queries, each run on A and B, then fused by M.",
            f"{ex}{{{e1}%1000, {e2}%1000}}{fz}%100",
        ),
    ]
    rows = []
    for sym, desc, ex_str in items:
        rows.append(
            f"<div class='syn-item'><code>{sym}</code> &mdash; {desc}"
            f"<div class='syn-ex'>e.g. <code>{_esc(ex_str)}</code></div></div>"
        )
    note = (
        "<div class='syn-note'><b>Choosing %N:</b> every stage cuts independently, and inner "
        "stages feed the next. A reranker or merger can only work with the candidates it's given, "
        f"so go deep inside (e.g. <code>%1000</code> per branch before <code>{_esc(fz)}</code>, or "
        "before a reranker) and cut to your final size only at the end (e.g. <code>%100</code>). "
        "Even if you ultimately want just the top 100, a larger upstream pool usually gives a "
        "better top 100. Braces are only for fusion and <b>require</b> a merger after "
        "<code>}</code>; a single engine takes no braces.</div>"
    )
    return "".join(rows) + note


def _describe(name):
    """Short description for a service: exact match wins, else strip a leading
    pure-letter prefix and match the engine alone (full-qwen3-vl-8b -> qwen3-vl-8b).

    The fallback lets one description line cover every prefixed variant of an
    engine. Returns "" when neither the full name nor the stripped engine is in
    the descriptions file.
    """
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    mo = _DESC_PREFIX.match(name)
    if mo and mo.group(1) in DESCRIPTIONS:
        return DESCRIPTIONS[mo.group(1)]
    return ""


def render_page(
    q,
    endpoint,
    custom_pipeline,
    effective_pipeline,
    collection,
    media_view,
    active_tab,
    avail,
    avail_err,
    search_engines,
    selected_engine,
    ranked,
    search_err,
):
    collections = list(avail.get("collection", []))
    options = ['<option value="">(none)</option>']
    for c in collections:
        sel = " selected" if c == collection else ""
        options.append(f'<option value="{_esc(c)}"{sel}>{_esc(c)}</option>')

    # Tabs = every view of the selected collection, ordered keyframe (the
    # auto-picked media view, also the default) first, then "video", then the rest
    # sorted by name. One global tab bar drives all cards; each card fetches the
    # active view's /view lazily.
    coll_views = ((avail.get("collection") or {}).get(collection) or {}).get("views") or {}
    head = []
    for v in (media_view, "video"):
        if v in coll_views and v not in head:
            head.append(v)
    view_order = head + sorted(v for v in coll_views if v not in head)

    # Engine dropdown: first-stage search engines from /avail + a custom option.
    eng_opts = []
    for e in search_engines:
        sel = " selected" if e == selected_engine else ""
        eng_opts.append(f'<option value="{_esc(e)}"{sel}>{_esc(e)}</option>')
    custom_sel = " selected" if selected_engine == "__custom__" else ""
    eng_opts.append(f'<option value="__custom__"{custom_sel}>Custom pipeline…</option>')
    custom_active = selected_engine == "__custom__"

    # Service categories for grammar-aware autocomplete (mapped from /avail roles).
    avail_cats = {
        "search": list(avail.get("search", [])),
        "score": list(avail.get("score", [])),
        "fuse": list(avail.get("fuse", [])),
        "expander": list(avail.get("decompose_query", [])),
        "aliases": list((avail.get("pipeline_aliases") or {}).keys()),
    }

    def _section(title, items):
        if not items:
            return ""
        body = " ".join(f"<code>{_esc(x)}</code>" for x in items)
        return f"<div class='av-row'><span class='av-label'>{_esc(title)}</span> {body}</div>"

    avail_html = ""
    if avail_err:
        avail_html = f"<div class='err'>/avail failed: {_esc(avail_err)}</div>"
    elif avail:
        # routir has no dedicated "reranker" type: "score" just means the service
        # implements the rerank capability. Split it by whether the service can
        # also retrieve first-stage (it's also in "search") or only rerank.
        search_set = set(avail.get("search") or [])
        score = list(avail.get("score") or [])
        rerank_dual = [s for s in score if s in search_set]
        rerank_only = [s for s in score if s not in search_set]
        avail_html += _section("search", avail.get("search"))
        avail_html += _section("rerank (dual-use)", rerank_dual)
        avail_html += _section("rerank-only", rerank_only)
        avail_html += _section("fuse", avail.get("fuse"))
        avail_html += _section("decompose_query", avail.get("decompose_query"))
        # Collections now carry typed views; show each as "name [view*:kind, ...]"
        # ('*' marks the default view) so the keyframe view in use is visible.
        coll_map = avail.get("collection") or {}
        if coll_map:
            cells = []
            for cname, info in coll_map.items():
                if isinstance(info, dict):
                    views = info.get("views") or {}
                    default = info.get("default")
                    vparts = ", ".join(f"{v}{'*' if v == default else ''}:{k}" for v, k in views.items())
                    label = f"{cname} [{vparts}]" if vparts else cname
                else:  # tolerate an older flat list shape
                    label = str(cname)
                cells.append(f"<code>{_esc(label)}</code>")
            avail_html += f"<div class='av-row'><span class='av-label'>collections</span> {' '.join(cells)}</div>"
        aliases = avail.get("pipeline_aliases") or {}
        if aliases:
            rows = "".join(
                f"<div class='alias'><code>{_esc(k)}</code> &rarr; <code>{_esc(v)}</code></div>" for k, v in aliases.items()
            )
            avail_html += f"<div class='av-row'><span class='av-label'>aliases</span></div>{rows}"
        if "grpc_port" in avail:
            avail_html += _section("grpc_port", [str(avail["grpc_port"])])

    search_html = ""
    if search_err:
        search_html = (
            f"<div class='err'><b>RoutIR error</b><br><code>{_esc(effective_pipeline)}</code><br>{_esc(search_err)}</div>"
        )
    elif q:
        search_html = (
            f"<div class='summary'>{len(ranked)} results for <b>{_esc(q)}</b> via <code>{_esc(effective_pipeline)}</code></div>"
        )

    syntax_html = _syntax_help(avail)
    pipelines_html = (
        "<div class='syn-note' style='margin-top:0'>A pipeline composes one or more <i>engines</i> "
        "into a single ranked result. Each engine plays one of a few roles, and the syntax below "
        "wires them together:</div>"
        "<div class='syn-item'><b>Search engines</b> (<code>/search</code>) &mdash; first-stage "
        "retrieval: take a query and return an initial ranked list of documents (e.g. dense or "
        "sparse retrievers).</div>"
        "<div class='syn-item'><b>Reranking engines</b> (engines providing <code>/score</code>) "
        "&mdash; re-score a candidate list from an earlier stage to reorder its top results more "
        "precisely.</div>"
        "<div class='syn-item'><b>Fusion engines</b> (<code>/fuse</code>) &mdash; merge several "
        "ranked lists (run in parallel) into one combined ranking, e.g. RRF.</div>"
        "<div class='syn-item'><b>Expansion engines</b> (<code>/decompose_query</code>) &mdash; "
        "turn one query into several sub-queries, each run through the pipeline and then fused "
        "back together.</div>"
        "<div class='syn-note'>A pipeline can be as simple as a single search engine, or chain "
        "these together &mdash; retrieve, rerank, and fuse &mdash; using the syntax below.</div>"
    )

    # Per-service descriptions (from --descriptions), limited to the services this
    # endpoint actually exposes so the panel never lists engines that aren't here.
    # Every present service gets a bullet "NAME: <desc>" (the description is looked
    # up by exact name, then by the prefix-stripped engine; see _describe); one
    # with no match in the file still appears as "NAME:" with nothing after it,
    # flagging the gap.
    present = []
    seen_svc = set()
    for role in ("search", "score", "fuse", "decompose_query"):
        for name in avail.get(role) or []:
            if name not in seen_svc:
                seen_svc.add(name)
                present.append(name)
    desc_rows = "".join(f"<li><code>{_esc(name)}</code>: {_esc(_describe(name))}</li>" for name in sorted(present))
    descriptions_html = (
        f"<ul class='desc-list'>{desc_rows}</ul>" if desc_rows else "<div class='nokf'>no services available</div>"
    )

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8"><title>Query RoutIR</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; color: #1a1a1a; }}
  header {{ background: #1f2933; color: #fff; padding: 12px 20px; }}
  header h1 {{ font-size: 16px; margin: 0 0 8px; font-weight: 600; }}
  header h1 a {{ color: inherit; text-decoration: underline; text-underline-offset: 2px; }}
  .endpoint {{ font-size: 12px; color: #9aa5b1; }}
  .endpoint a {{ color: #7ee2a8; text-decoration: underline; cursor: pointer; margin-left: 4px; }}
  /* Override the global flex `form` rule: the endpoint editor is a compact
     inline row that stays hidden until you click "edit". */
  #ep-form {{ display: none; gap: 6px; align-items: center; margin-top: 6px; }}
  #ep-form.open {{ display: flex; }}
  #ep-form input {{ width: 320px; font-size: 12px; padding: 4px 6px; font-family: ui-monospace, monospace; }}
  #ep-form button {{ font-size: 12px; padding: 4px 10px; }}
  form {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: flex-start; margin-top: 8px; }}
  /* The pipeline field carries a multi-line error-message slot (#pipe-msg)
     under its input; with the form top-aligned (align-items: flex-start) it
     simply grows downward without nudging the other fields, so no reserved
     spacer is needed under them. */
  form label {{ font-size: 11px; color: #cbd2d9; display: block; }}
  form input, form select, form textarea {{ font-size: 13px; padding: 6px 8px; border: 1px solid #52606d; border-radius: 4px; }}
  textarea[name=q] {{ width: 560px; font-family: inherit; resize: vertical; }}
  #pipeline {{ width: 690px; font-family: ui-monospace, monospace; resize: vertical; }}
  .pipe-wrap {{ position: relative; }}
  #pipe-badge {{ font-weight: 600; }}
  #pipe-badge.ok {{ color: #7ee2a8; }}
  #pipe-badge.bad {{ color: #ffb3b3; }}
  #pipeline.invalid {{ border-color: #cf1124; box-shadow: 0 0 0 2px rgba(207,17,36,.35); background: #fff6f6; }}
  #pipe-msg {{ font-size: 11px; color: #ffb3b3; margin-top: 3px; max-width: 690px; min-height: 48px; line-height: 1.4; font-family: ui-monospace, monospace; }}
  .suggest {{ position: absolute; top: 100%; left: 0; z-index: 30; background: #fff; color: #1a1a1a;
              border: 1px solid #cbd2d9; border-radius: 4px; min-width: 300px; max-height: 240px;
              overflow: auto; box-shadow: 0 4px 14px rgba(0,0,0,.2); display: none; margin-top: 2px; }}
  .suggest div {{ padding: 5px 10px; font-size: 12px; font-family: ui-monospace, monospace; cursor: pointer;
                  display: flex; justify-content: space-between; gap: 16px; }}
  .suggest div .t {{ color: #9aa5b1; font-family: system-ui, sans-serif; font-size: 11px; }}
  .suggest div.sel, .suggest div:hover {{ background: #e6f4fb; }}
  button {{ font-size: 13px; padding: 7px 16px; border: 0; border-radius: 4px; background: #2bb0ed; color: #fff; cursor: pointer; }}
  main {{ display: flex; gap: 16px; padding: 16px 20px; align-items: flex-start; }}
  #results {{ flex: 1; min-width: 0; }}
  aside {{ width: 320px; flex: none; background: #f5f7fa; border: 1px solid #e4e7eb; border-radius: 6px; padding: 12px; font-size: 12px; }}
  aside h2 {{ font-size: 13px; margin: 0 0 8px; }}
  aside details > summary {{ font-size: 13px; font-weight: 700; margin: 0 0 8px; cursor: pointer; list-style: none; }}
  aside details > summary::-webkit-details-marker {{ display: none; }}
  aside details > summary::before {{ content: "\\25B8"; display: inline-block; width: 1em; color: #616e7c; transition: transform 0.15s; }}
  aside details[open] > summary::before {{ transform: rotate(90deg); }}
  .av-row {{ margin: 4px 0; line-height: 1.6; }}
  .av-label {{ display: inline-block; min-width: 84px; color: #616e7c; font-weight: 600; }}
  code {{ background: #e4e7eb; padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
  .alias {{ margin: 2px 0 2px 84px; }}
  .desc-list {{ margin: 0; padding-left: 18px; line-height: 1.6; }}
  .desc-list li {{ margin: 2px 0; }}
  .card {{ border: 1px solid #e4e7eb; border-radius: 6px; padding: 10px; margin-bottom: 12px; }}
  .card-head {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
  .vid {{ font-family: ui-monospace, monospace; font-weight: 600; }}
  .score {{ color: #616e7c; font-size: 12px; }}
  .strip {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start; }}
  .strip figure {{ margin: 0; text-align: center; }}
  .strip img {{ height: 150px; width: auto; border-radius: 3px; background: #eee; display: block; }}
  .strip figcaption {{ font-size: 10px; color: #9aa5b1; }}
  .toggle {{ background: #f0f4f8; color: #334e68; font-size: 11px; padding: 4px 10px; margin-top: 6px; }}
  /* View tabs: deliberately styled unlike the blue prev/next pager buttons —
     larger gray "pill" tabs with a dark active state — and stuck to the top of
     the viewport so they stay clickable while you scroll the results. */
  .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; position: sticky; top: 0; z-index: 20;
           background: #fff; padding: 10px 0 12px; margin-bottom: 10px;
           border-bottom: 2px solid #cbd2d9; box-shadow: 0 4px 8px -4px rgba(0,0,0,.15); }}
  .tab {{ background: #e4e7eb; color: #1f2933; font-size: 14px; font-weight: 600;
          font-family: ui-monospace, monospace; padding: 8px 16px; border-radius: 999px; border: 1px solid #cbd2d9; }}
  .tab:hover {{ background: #d3dce6; }}
  .tab.active {{ background: #1f2933; color: #fff; border-color: #1f2933; }}
  .tabpanel {{ min-height: 40px; }}
  .textview {{ margin: 0; max-height: 360px; overflow: auto; background: #f5f7fa; border: 1px solid #e4e7eb;
               border-radius: 4px; padding: 8px 10px; font-size: 12px; line-height: 1.45; white-space: pre-wrap;
               word-break: break-word; }}
  .videoview {{ max-height: 360px; max-width: 100%; border-radius: 4px; background: #000; display: block; }}
  .pager {{ display: flex; gap: 12px; align-items: center; margin: 8px 0 16px; }}
  .summary {{ color: #616e7c; font-size: 13px; margin-bottom: 8px; }}
  .err {{ color: #cf1124; background: #ffeeee; padding: 8px; border-radius: 4px; }}
  .nokf {{ color: #9aa5b1; font-size: 12px; font-style: italic; }}
  .av-sep {{ border: 0; border-top: 1px solid #e4e7eb; margin: 14px 0; }}
  .syn-item {{ margin: 0 0 9px; line-height: 1.5; }}
  .syn-ex {{ margin-top: 2px; color: #616e7c; }}
  .syn-note {{ margin-top: 6px; color: #616e7c; font-size: 11px; line-height: 1.5; }}
  /* Python snippet panel: a dark monospace block (preserving whitespace so the
     indentation reads as Python) with a copy button above it. */
  .code-head {{ display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }}
  .code-label {{ font-size: 11px; font-weight: 600; color: #616e7c; text-transform: uppercase; letter-spacing: .04em; }}
  .code-snippet {{ margin: 4px 0 0; max-height: 360px; overflow: auto; background: #1f2933; color: #e4e7eb;
                   border-radius: 4px; padding: 8px 10px; font-size: 11px; line-height: 1.45; white-space: pre;
                   word-break: normal; font-family: ui-monospace, monospace; }}
  .strip img {{ cursor: zoom-in; }}
  .lb {{ display: none; position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,.78);
         align-items: center; justify-content: center; padding: 24px; }}
  .lb.open {{ display: flex; }}
  .lb-box {{ background: #fff; border-radius: 6px; max-width: 95vw; max-height: 95vh; display: flex;
             flex-direction: column; overflow: hidden; box-shadow: 0 10px 40px rgba(0,0,0,.5); }}
  .lb-head {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 8px 12px; }}
  .lb-title {{ font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; }}
  .lb-close {{ background: #e4e7eb; color: #1a1a1a; font-size: 18px; line-height: 1; padding: 2px 10px; }}
  .lb-close:hover {{ background: #cbd2d9; }}
  #lb-img {{ display: block; max-width: 95vw; max-height: calc(95vh - 44px); width: auto; height: auto; background: #eee; }}
</style>
</head><body>
<header>
  <h1>Query <a href="https://github.com/hltcoe/routir" target="_blank" rel="noopener">RoutIR</a></h1>
  <div class="endpoint">
    endpoint: <span id="ep-text">{_esc(endpoint)}</span>
    <a href="#" id="ep-edit" onclick="toggleEpEdit(event)">edit</a>
    <form method="get" action="/" id="ep-form">
      <input name="endpoint" id="ep-input" value="{_esc(endpoint)}" placeholder="http://compute01:5000">
      <input type="hidden" name="q" value="{_esc(q)}">
      <input type="hidden" name="pipeline" value="{_esc(custom_pipeline)}">
      <input type="hidden" name="collection" value="{_esc(collection)}">
      <input type="hidden" name="engine" value="{_esc(selected_engine)}">
      <button type="submit">apply &amp; reload</button>
      <a href="#" onclick="toggleEpEdit(event)">cancel</a>
    </form>
  </div>
  <form method="get" action="/">
    <input type="hidden" name="endpoint" value="{_esc(endpoint)}">
    <input type="hidden" name="tab" id="tab-field" value="{_esc(active_tab)}">
    <div><label>Query</label><textarea name="q" rows="5" placeholder="a person cooking pasta" autofocus>{_esc(q)}</textarea></div>
    <div><label>Engine</label><select name="engine" id="engine" onchange="onEngineChange()">{"".join(eng_opts)}</select></div>
    <div class="pipe-wrap">
      <label>Custom pipeline <span id="pipe-badge"></span></label>
      <textarea name="pipeline" id="pipeline" rows="2" autocomplete="off" placeholder="full-qwen3-vl-8b%1000  or  {{a%1000, b%1000}}RRF%100"{"" if custom_active else " disabled"}>{_esc(custom_pipeline)}</textarea>
      <div id="pipe-suggest" class="suggest"></div>
      <div id="pipe-msg"></div>
    </div>
    <div><label>Collection</label><select name="collection">{"".join(options)}</select></div>
    <div><label>&nbsp;</label><button type="submit">Search</button></div>
  </form>
</header>
<main>
  <div id="results">
    {search_html}
    <div class="pager" id="pager"></div>
    <div class="tabs" id="tabbar"></div>
    <div id="cards"></div>
    <div class="pager" id="pager2"></div>
  </div>
  <aside>
    <h2>/avail <span style="font-weight:400;color:#9aa5b1">(refresh page to update)</span></h2>
    {avail_html or "<div class='nokf'>no data</div>"}
    <hr class="av-sep">
    <details style="margin-top:10px">
      <summary>Descriptions</summary>
      {descriptions_html}
    </details>
    <hr class="av-sep">
    <details open>
      <summary>Python</summary>
      <div class='syn-note' style='margin-top:0'>Run the current query and pipeline from Python with the
      RoutIR client (it reflects the query, engine/pipeline and collection above as you edit them):</div>
      <div class="code-head"><span class="code-label">sync</span><button type="button" id="py-copy-sync" class="toggle">copy</button></div>
      <pre class="code-snippet" id="py-sync"></pre>
      <div class="code-head"><span class="code-label">async</span><button type="button" id="py-copy-async" class="toggle">copy</button></div>
      <pre class="code-snippet" id="py-async"></pre>
    </details>
    <hr class="av-sep">
    <details open>
      <summary>Pipelines</summary>
      {pipelines_html}
    </details>
    <hr class="av-sep">
    <details open>
      <summary>Pipeline syntax</summary>
      {syntax_html}
    </details>
  </aside>
</main>
<div id="lightbox" class="lb">
  <div class="lb-box" onclick="event.stopPropagation()">
    <div class="lb-head">
      <span class="lb-title" id="lb-title"></span>
      <button type="button" class="lb-close" id="lb-close" aria-label="Close">&times;</button>
    </div>
    <img id="lb-img" alt="">
  </div>
</div>
<script>
// ----- endpoint editor: reveal an inline form that reloads the page against a
// new endpoint (submitting ?endpoint=... retargets the server-side client) -----
function toggleEpEdit(ev) {{
  ev.preventDefault();
  const f = document.getElementById('ep-form');
  const opening = !f.classList.contains('open');
  f.classList.toggle('open', opening);
  if (opening) {{ const i = document.getElementById('ep-input'); i.focus(); i.select(); }}
}}

// ----- grammar-aware autocomplete + live validation for the pipeline box -----
const AVAIL = {json.dumps(avail_cats)};
// slot (from /complete) -> ordered [category-label, names[]] groups to suggest.
function slotSources(slot) {{
  if (slot === 'stage')  return [['search', AVAIL.search], ['expander', AVAIL.expander], ['alias', AVAIL.aliases]];
  if (slot === 'rerank') return [['reranker', AVAIL.score], ['alias', AVAIL.aliases]];
  if (slot === 'merger') return [['fusion', AVAIL.fuse]];
  if (slot === 'number') return [['cut', ['100', '500', '1000']]];
  return [];  // 'alias' (free label) and null -> nothing
}}

// Every service name known to this endpoint, across all /avail roles plus the
// pipeline-alias shortcuts. The grammar accepts any NAME, so a token that isn't
// in here is a typo for a service that doesn't exist (e.g. "asdf").
const KNOWN = new Set([].concat(
  AVAIL.search || [], AVAIL.score || [], AVAIL.fuse || [], AVAIL.expander || [], AVAIL.aliases || []
));
// Names referenced in the pipeline that aren't known services. Bracketed
// [alias] labels are user-defined, so they're skipped, and so is the token
// under the cursor while it's still being typed (so each prefix of a real name
// isn't flagged mid-type); pass cursor=null to check the whole string.
function unknownNames(text, cursor) {{
  const unknown = [], seen = new Set();
  let inBracket = false, m;
  const re = /\[|\]|[A-Za-z][A-Za-z0-9_\-]*/g;
  while ((m = re.exec(text)) !== null) {{
    const tok = m[0];
    if (tok === '[') {{ inBracket = true; continue; }}
    if (tok === ']') {{ inBracket = false; continue; }}
    if (inBracket) continue;  // user-defined alias label, not a service
    if (cursor != null && m.index <= cursor && cursor <= m.index + tok.length) continue;
    if (KNOWN.has(tok) || seen.has(tok)) continue;
    seen.add(tok); unknown.push(tok);
  }}
  return unknown;
}}

const box = document.getElementById('pipeline');
const badge = document.getElementById('pipe-badge');
const msg = document.getElementById('pipe-msg');
const suggest = document.getElementById('pipe-suggest');
let sugg = [];          // current suggestions
let suggSel = -1;       // highlighted index
let debounceTimer = null;

function closeSuggest() {{ suggest.style.display = 'none'; sugg = []; suggSel = -1; }}

function renderSuggest() {{
  if (!sugg.length) {{ closeSuggest(); return; }}
  suggest.innerHTML = '';
  sugg.forEach((s, i) => {{
    const d = document.createElement('div');
    if (i === suggSel) d.className = 'sel';
    d.innerHTML = '<span>' + s.value + '</span><span class="t">' + s.type + '</span>';
    d.onmousedown = (ev) => {{ ev.preventDefault(); applySuggest(i); }};
    suggest.appendChild(d);
  }});
  suggest.style.display = 'block';
}}

function applySuggest(i) {{
  const s = sugg[i]; if (!s) return;
  const v = box.value;
  // Names/numbers replace the token being typed; symbols insert at the cursor.
  const before = v.slice(0, s.from);
  const after = v.slice(box.selectionStart);
  box.value = before + s.value + after;
  const caret = before.length + s.value.length;
  box.focus();
  box.setSelectionRange(caret, caret);
  closeSuggest();
  scheduleComplete();  // re-validate + offer the next slot
}}

function setBadge(valid, m, label) {{
  box.classList.toggle('invalid', valid === false);
  if (valid === true)  {{ badge.textContent = '✓ valid'; badge.className = 'ok'; msg.textContent = ''; }}
  else if (valid === false) {{
    badge.textContent = label || '✗ invalid syntax';
    badge.className = 'bad';
    // Non-blocking: the box turns red and we explain why, but Search still works
    // (the server-side RoutIR error is still shown if you submit anyway).
    msg.textContent = '⚠ ' + (m || 'pipeline is incomplete or malformed') + ' — you can still search, but it will likely error.';
  }} else {{ badge.textContent = ''; badge.className = ''; msg.textContent = ''; }}
}}

async function runComplete() {{
  if (box.disabled) {{ setBadge(null); closeSuggest(); return; }}
  const pos = box.selectionStart;
  let data;
  try {{
    const r = await fetch('/complete?' + new URLSearchParams({{pipeline: box.value, pos}}));
    data = await r.json();
  }} catch (e) {{ return; }}
  // The grammar only checks syntax; a NAME it accepts may still be a service
  // that doesn't exist on this endpoint (a typo like "asdf"). Flag those too.
  let valid = data.valid, errMsg = data.error, badgeLabel = null;
  if (valid === true) {{
    const cursor = document.activeElement === box ? box.selectionStart : null;
    const unknown = unknownNames(box.value, cursor);
    if (unknown.length) {{
      valid = false;
      badgeLabel = unknown.length > 1 ? '✗ unknown services' : '✗ unknown service';
      errMsg = (unknown.length > 1 ? 'unknown services: ' : 'unknown service: ') + unknown.join(', ');
    }}
  }}
  setBadge(valid, errMsg, badgeLabel);
  // The badge validates on load / programmatic calls, but the dropdown should
  // only appear while you're actually typing in the box — not after a refresh
  // or while typing in the query box.
  if (document.activeElement !== box) {{ closeSuggest(); return; }}
  const partial = (data.partial || '').toLowerCase();
  // Name/number completions (replace the in-progress token, prefix-filtered).
  const seen = new Set();
  const names = [];
  for (const [type, vals] of slotSources(data.slot)) {{
    for (const name of (vals || [])) {{
      if (seen.has(name)) continue;
      if (partial && !name.toLowerCase().startsWith(partial)) continue;
      if (name.toLowerCase() === partial) continue;  // don't offer to complete a token to itself
      seen.add(name);
      names.push({{value: name, type, from: data.replace_from}});
    }}
  }}
  // Operator/punctuation completions (insert at the cursor) from the grammar.
  // The grammar allows "engine{{...}}" (query expansion) after any engine, but it
  // only works if that engine is a decompose_query service, so drop the expander
  // open-brace unless its engine is in AVAIL.expander.
  const expanders = AVAIL.expander || [];
  const syms = (data.symbols || [])
    .filter(s => !('expander' in s) || expanders.includes(s.expander))
    .map(s => ({{value: s.value, type: s.type, from: pos}}));
  // At a boundary (no token being typed) both are relevant; while typing a token
  // show its name/number matches, falling back to symbols once the token is whole.
  if (data.partial === '') sugg = names.concat(syms);
  else if (names.length)   sugg = names;
  else                     sugg = syms;
  suggSel = sugg.length ? 0 : -1;
  renderSuggest();
}}

function scheduleComplete() {{
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(runComplete, 120);
}}

box.addEventListener('input', scheduleComplete);
box.addEventListener('click', scheduleComplete);
box.addEventListener('keydown', (e) => {{
  if (suggest.style.display === 'block' && sugg.length) {{
    if (e.key === 'ArrowDown') {{ e.preventDefault(); suggSel = (suggSel + 1) % sugg.length; renderSuggest(); return; }}
    if (e.key === 'ArrowUp')   {{ e.preventDefault(); suggSel = (suggSel - 1 + sugg.length) % sugg.length; renderSuggest(); return; }}
    if (e.key === 'Enter' || e.key === 'Tab') {{ e.preventDefault(); applySuggest(suggSel); return; }}
    if (e.key === 'Escape')    {{ e.preventDefault(); closeSuggest(); return; }}
  }} else if (e.key === 'Enter' && !e.shiftKey) {{
    // The box is a 2-row textarea (so long pipelines wrap), but a pipeline is a
    // single line: plain Enter submits the form like the old <input> did; use
    // Shift+Enter if you ever need a literal newline.
    e.preventDefault();
    box.form.requestSubmit();
  }}
}});
// On blur, re-check with cursor=null so a finished-but-unknown name (skipped
// while it was the token under the cursor) gets flagged once you leave the box.
box.addEventListener('blur', () => setTimeout(() => {{ closeSuggest(); runComplete(); }}, 150));

// The custom-pipeline box is only editable when "Custom pipeline…" is chosen.
function onEngineChange() {{
  box.disabled = document.getElementById('engine').value !== '__custom__';
  if (box.disabled) {{ setBadge(null); closeSuggest(); }}
  else {{ box.focus(); runComplete(); }}
}}
if (!box.disabled) runComplete();  // validate the prefilled default on load

const RANKED = {json.dumps(ranked)};
// Each result shows one tab per collection view (VIEWS), keyframe-first/default.
// A tab's content is pulled lazily from RoutIR's /content (via /view + /part) the
// first time it's opened; the page bakes in endpoint/collection so those routes
// stay stateless. VIEWS is empty when no collection is selected.
const ENDPOINT = {json.dumps(endpoint)};
const COLLECTION = {json.dumps(collection)};
const MEDIA_VIEW = {json.dumps(media_view)};
const VIEWS = {json.dumps(view_order)};
const PAGE_SIZE = 20;
let page = 0;
const nPages = Math.max(1, Math.ceil(RANKED.length / PAGE_SIZE));

// One global active view drives every card: clicking a tab switches all results
// at once. /view payloads are cached per (doc, view) so switching back is free.
// The active tab lives in the URL (?tab=) so a copy-pasted link reopens on it.
const URL_TAB = {json.dumps(active_tab)} || new URLSearchParams(location.search).get('tab');
let activeView = (URL_TAB && VIEWS.includes(URL_TAB)) ? URL_TAB
               : (MEDIA_VIEW && VIEWS.includes(MEDIA_VIEW)) ? MEDIA_VIEW
               : (VIEWS[0] || null);
const VIEW_DATA = {{}};

// Set the active view and mirror it into the URL (replaceState: copy-pasteable,
// no history spam) and the search form's hidden field (so a new search keeps it).
function setActiveView(view) {{
  activeView = view;
  try {{
    const u = new URL(location);
    u.searchParams.set('tab', view);
    history.replaceState(null, '', u);
  }} catch (e) {{ /* file:// or odd URLs: just skip the URL sync */ }}
  const f = document.getElementById('tab-field');
  if (f) f.value = view;
}}

// Query string pinning a request to one collection view.
function viewQS(view) {{
  return new URLSearchParams({{endpoint: ENDPOINT, collection: COLLECTION || '', view: view || ''}}).toString();
}}
function partSrc(vid, view, idx) {{
  return '/part/' + encodeURIComponent(vid) + '/' + idx + '?' + viewQS(view);
}}

// ----- lightbox (used by the image/keyframe tab) -----
const lightbox = document.getElementById('lightbox');
const lbImg = document.getElementById('lb-img');
const lbTitle = document.getElementById('lb-title');
function openLightbox(vid, view, idx) {{
  lbImg.src = partSrc(vid, view, idx);
  lbImg.alt = vid + ' #' + (idx + 1);
  lbTitle.textContent = vid + '  #' + (idx + 1);
  lightbox.classList.add('open');
}}
function closeLightbox() {{ lightbox.classList.remove('open'); lbImg.src = ''; }}
document.getElementById('lb-close').onclick = closeLightbox;
lightbox.onclick = closeLightbox;  // click the backdrop (the box stops propagation)
document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape' && lightbox.classList.contains('open')) closeLightbox(); }});

// ----- per-type renderers -----
function imgEl(vid, view, idx) {{
  const fig = document.createElement('figure');
  const img = document.createElement('img');
  img.loading = 'lazy';
  img.src = partSrc(vid, view, idx);
  img.alt = vid + ' #' + (idx + 1);
  img.onclick = () => openLightbox(vid, view, idx);
  const cap = document.createElement('figcaption');
  cap.textContent = '#' + (idx + 1);
  fig.appendChild(img); fig.appendChild(cap);
  return fig;
}}

function renderImage(vid, view, data) {{
  const wrap = document.createElement('div');
  const strip = document.createElement('div');
  strip.className = 'strip';
  wrap.appendChild(strip);
  let showingAll = false;
  const draw = () => {{
    strip.innerHTML = '';
    for (const t of (showingAll ? data.all : data.preview)) strip.appendChild(imgEl(vid, view, t));
  }};
  draw();
  if (data.all.length > data.preview.length) {{
    const btn = document.createElement('button');
    btn.className = 'toggle';
    const label = () => btn.textContent = showingAll ? 'show fewer' : ('show all ' + data.all.length + ' frames');
    label();
    btn.onclick = () => {{ showingAll = !showingAll; draw(); label(); }};
    wrap.appendChild(btn);
  }}
  return wrap;
}}

function renderVideo(vid, view) {{
  const v = document.createElement('video');
  v.controls = true;
  v.className = 'videoview';
  v.src = partSrc(vid, view, 0);
  return v;
}}

function renderText(data) {{
  const wrap = document.createElement('div');
  const pre = document.createElement('pre');
  pre.className = 'textview';
  pre.textContent = data.text || '(empty)';
  wrap.appendChild(pre);
  // Offer the raw JSON/JSONL when it differs from the readable rendering.
  if (data.raw && data.raw !== data.text) {{
    let showingRaw = false;
    const btn = document.createElement('button');
    btn.className = 'toggle';
    btn.textContent = 'show raw';
    btn.onclick = () => {{
      showingRaw = !showingRaw;
      pre.textContent = showingRaw ? data.raw : (data.text || '(empty)');
      btn.textContent = showingRaw ? 'show readable' : 'show raw';
    }};
    wrap.appendChild(btn);
  }}
  return wrap;
}}

// Fetch one view of one doc (cached per doc+view, so a tab switch never refetches).
async function getView(vid, view) {{
  const key = vid + '|' + view;
  if (!(key in VIEW_DATA)) {{
    try {{
      const r = await fetch('/view/' + encodeURIComponent(vid) + '?' + viewQS(view));
      VIEW_DATA[key] = await r.json();
    }} catch (e) {{
      VIEW_DATA[key] = {{type: 'error', error: String(e)}};
    }}
  }}
  return VIEW_DATA[key];
}}

// Build a fresh DOM node for an already-fetched /view payload.
function buildViewNode(vid, view, data) {{
  if (data.type === 'image') return renderImage(vid, view, data);
  if (data.type === 'video') return renderVideo(vid, view);
  if (data.type === 'text')  return renderText(data);
  const msg = document.createElement('div');
  if (data.type === 'empty')      msg.innerHTML = '<span class="nokf">no ' + view + ' for this doc</span>';
  else if (data.type === 'error') msg.innerHTML = '<span class="nokf">unavailable: ' + (data.error || '?') + '</span>';
  else {{  // binary: offer the raw bytes
    const a = document.createElement('a');
    a.href = partSrc(vid, view, 0); a.textContent = 'download ' + view; a.download = vid + '.' + view;
    msg.appendChild(a);
  }}
  return msg;
}}

// The global tab bar: one set of tabs above all results; clicking re-renders the
// whole page for that view so every card switches at once.
function renderTabs() {{
  const bar = document.getElementById('tabbar');
  bar.innerHTML = '';
  if (!COLLECTION || !VIEWS.length || !RANKED.length) return;
  for (const view of VIEWS) {{
    const b = document.createElement('button');
    b.className = 'tab' + (view === activeView ? ' active' : '');
    b.textContent = view;
    b.onclick = () => {{
      if (view === activeView) return;
      setActiveView(view);
      renderTabs();
      renderPage();
    }};
    bar.appendChild(b);
  }}
}}

async function renderCard(res) {{
  const card = document.createElement('div');
  card.className = 'card';
  const head = document.createElement('div');
  head.className = 'card-head';
  head.innerHTML = '<span class="vid">#' + res.rank + ' ' + res.id + '</span>' +
                   '<span class="score">score ' + res.score.toFixed(4) + '</span>';
  card.appendChild(head);

  const panel = document.createElement('div');
  panel.className = 'tabpanel';
  card.appendChild(panel);
  if (!COLLECTION || !activeView) {{
    panel.innerHTML = '<span class="nokf">no collection selected</span>';
    return card;
  }}
  panel.innerHTML = '<span class="nokf">loading…</span>';
  const data = await getView(res.id, activeView);
  panel.innerHTML = '';
  panel.appendChild(buildViewNode(res.id, activeView, data));
  return card;
}}

function renderPager(el) {{
  el.innerHTML = '';
  if (!RANKED.length) return;
  const prev = document.createElement('button');
  prev.textContent = '‹ prev'; prev.disabled = page === 0;
  prev.onclick = () => {{ page--; renderPage(); }};
  const next = document.createElement('button');
  next.textContent = 'next ›'; next.disabled = page >= nPages - 1;
  next.onclick = () => {{ page++; renderPage(); }};
  const info = document.createElement('span');
  const start = page * PAGE_SIZE + 1;
  const end = Math.min((page + 1) * PAGE_SIZE, RANKED.length);
  info.textContent = 'results ' + start + '–' + end + ' of ' + RANKED.length +
                     ' (page ' + (page + 1) + '/' + nPages + ')';
  el.appendChild(prev); el.appendChild(info); el.appendChild(next);
}}

// Bumped on every (re)render so a rapid tab/page switch abandons the in-flight
// one instead of interleaving cards from two renders into #cards.
let renderEpoch = 0;
async function renderPage() {{
  const epoch = ++renderEpoch;
  renderPager(document.getElementById('pager'));
  renderPager(document.getElementById('pager2'));
  const cards = document.getElementById('cards');
  cards.innerHTML = '';
  window.scrollTo(0, 0);
  const slice = RANKED.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  for (const res of slice) {{
    const card = await renderCard(res);
    if (epoch !== renderEpoch) return;  // a newer render superseded this one
    cards.appendChild(card);
  }}
}}

if (activeView) setActiveView(activeView);  // reflect the initial tab in the URL/field
renderTabs();
renderPage();

// ----- Python snippet: copy-pasteable routir-client programs (sync + async)
// for the current query + pipeline + collection, kept in sync with the form as
// you edit it. JSON.stringify yields a valid Python string literal (double-
// quoted, with the usual escapes), so we use it to embed the query / pipeline /
// collection safely. The endpoint is fixed (the server's per-request endpoint);
// the snippet deliberately omits any API key — add one yourself (e.g.
// api_key=os.environ["ROUTIR_API_KEY"]) if this endpoint requires it. -----
const qBox = document.querySelector('textarea[name=q]');
const collSel = document.querySelector('select[name=collection]');
const engSel = document.getElementById('engine');
const pySync = document.getElementById('py-sync');
const pyAsync = document.getElementById('py-async');

// Mirror index_page's effective-pipeline rule: a chosen engine runs as a bare
// "<engine>" pipeline; "Custom pipeline…" uses whatever is in the box.
function effectivePipeline() {{
  return engSel.value === '__custom__' ? box.value.trim() : engSel.value;
}}

// The pipeline() call arguments shared by both snippets, each already indented by
// `indent` spaces (the call spans several lines so long pipelines stay readable).
function pipelineArgs(indent) {{
  const py = JSON.stringify;
  const pad = ' '.repeat(indent);
  const args = [py(effectivePipeline() || '<choose an engine or pipeline>'), py((qBox.value || '').trim() || '<your query>')];
  if (collSel.value) args.push('collection=' + py(collSel.value));
  return args.map(a => pad + a + ',').join('\\n');
}}

function buildSync() {{
  const py = JSON.stringify;
  return [
    'from routir.client.sync import Client',
    '',
    'with Client(endpoint=' + py(ENDPOINT) + ') as client:',
    '    result = client.pipeline(',
    pipelineArgs(8),
    '    )',
    '',
    '# {{doc_id: score}}, higher = more relevant',
    'scores = result["scores"]',
    'for rank, (doc_id, score) in enumerate(',
    '    sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1',
    '):',
    '    print(rank, doc_id, score)',
  ].join('\\n');
}}

function buildAsync() {{
  const py = JSON.stringify;
  return [
    'import asyncio',
    '',
    'from routir.client.client import AsyncClient',
    '',
    '',
    'async def main():',
    '    async with AsyncClient(endpoint=' + py(ENDPOINT) + ') as client:',
    '        result = await client.pipeline(',
    pipelineArgs(12),
    '        )',
    '',
    '    # {{doc_id: score}}, higher = more relevant',
    '    scores = result["scores"]',
    '    for rank, (doc_id, score) in enumerate(',
    '        sorted(scores.items(), key=lambda kv: kv[1], reverse=True), start=1',
    '    ):',
    '        print(rank, doc_id, score)',
    '',
    '',
    'asyncio.run(main())',
  ].join('\\n');
}}

function updateSnippet() {{
  pySync.textContent = buildSync();
  pyAsync.textContent = buildAsync();
}}

// Copy a <pre>'s text to the clipboard, falling back to selecting it (e.g. on a
// non-HTTPS origin where the Clipboard API is unavailable).
function wireCopy(btn, pre) {{
  btn.onclick = async () => {{
    try {{
      await navigator.clipboard.writeText(pre.textContent);
      btn.textContent = 'copied';
      setTimeout(() => {{ btn.textContent = 'copy'; }}, 1200);
    }} catch (e) {{
      const r = document.createRange();
      r.selectNodeContents(pre);
      const sel = window.getSelection();
      sel.removeAllRanges(); sel.addRange(r);
    }}
  }};
}}
wireCopy(document.getElementById('py-copy-sync'), pySync);
wireCopy(document.getElementById('py-copy-async'), pyAsync);

qBox.addEventListener('input', updateSnippet);
collSel.addEventListener('change', updateSnippet);
engSel.addEventListener('change', updateSnippet);
box.addEventListener('input', updateSnippet);
updateSnippet();
</script>
</body></html>"""


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    global ARGS, DESCRIPTIONS

    # Per-request log lines (the before/after_request hooks) go to stderr with a
    # timestamp so you can watch the UI's activity alongside hypercorn's output.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True, help="RoutIR endpoint URL (http(s):// or grpc(s)://).")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: %(default)s).")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind (default: %(default)s).")
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of evenly-spaced keyframes to preview per result before 'show all' (default: %(default)s).",
    )
    parser.add_argument(
        "--pipeline",
        default="full-qwen3-vl-8b",
        help="Default pipeline string to prefill the custom-pipeline box (default: %(default)s). "
        "The pipeline runs verbatim; add a trailing '%%N' to cap the result set, otherwise "
        "RoutIR's own default result size applies. Note: braces are only for parallel fusion "
        "and require a merger, e.g. '{a%%1000, b%%1000}RRF%%100'.",
    )
    parser.add_argument("--collection", default=None, help="Default collection to preselect.")
    # Default to the repo's default_descriptions.json when it's present (it sits at
    # the repo root, one level up from scripts/); fall back to no descriptions if
    # the script is run from outside a checkout.
    default_descriptions = Path(__file__).resolve().parent.parent / "default_descriptions.json"
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=default_descriptions if default_descriptions.is_file() else None,
        help="JSON file mapping service name -> short description, shown in a collapsible "
        "'Descriptions' panel (default: %(default)s). A key may be the full service name "
        "or just the engine with its leading 'letters-' prefix dropped (e.g. 'qwen3-vl-8b' "
        "matches 'full-qwen3-vl-8b'); exact names win over the stripped fallback. Services "
        "present in /avail but missing from the file are still listed (with no description).",
    )
    parser.add_argument("--api-key", default=os.environ.get("ROUTIR_API_KEY"), help="Bearer token (default: ROUTIR_API_KEY).")
    parser.add_argument("--grpc-endpoint", default=None, help="Explicit gRPC target (host:port) when --endpoint is REST.")
    parser.add_argument("--transport", default="auto", choices=["auto", "grpc", "rest"], help="Transport (default: %(default)s).")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds (default: %(default)s).")
    args = parser.parse_args()
    ARGS = args

    if args.descriptions is not None:
        try:
            with args.descriptions.open() as f:
                DESCRIPTIONS = json.load(f)
        except Exception as e:  # noqa: BLE001 - a bad descriptions file shouldn't be fatal
            parser.error(f"--descriptions could not be read: {e}")
        if not isinstance(DESCRIPTIONS, dict):
            parser.error("--descriptions must be a JSON object mapping service name -> description.")
        print(f"Loaded {len(DESCRIPTIONS)} service descriptions from {args.descriptions}")

    @app.before_serving
    async def _open_client():
        # Warm the pool with the default endpoint; others are created on demand.
        _client_for(args.endpoint)

    @app.after_serving
    async def _close_clients():
        for client in CLIENTS.values():
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - best-effort close on shutdown
                pass

    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"{args.host}:{args.port}"]
    print(f"Serving on http://{args.host}:{args.port}  (endpoint: {args.endpoint})")
    asyncio.run(serve(app, config))


if __name__ == "__main__":
    main()
