# RoutIR — `src/routir`

RoutIR is an async search/retrieval service. It hosts retrieval models (dense, sparse,
rerankers, fusion, query expanders) behind a uniform HTTP/gRPC API and lets you compose
them into multi-stage pipelines with a small DSL. This file orients an agent who needs to
**add a new engine** (a reranker, search engine, query expander, or fusion method).

For task-oriented walkthroughs — calling the client, standing up a local server that
imports a master server, wrapping a bi-encoder or reranker via `file_imports`, and
serving multi-view collections — see [`SKILL.md`](SKILL.md).

## The one concept that matters: `Engine`

Everything pluggable is a subclass of `Engine` (`src/routir/models/abstract.py`). An engine
declares its capabilities simply by which methods it overrides — there is no registration
boilerplate, no capability flag to set. The base class auto-detects capabilities by checking
whether you overrode the `*_batch` method:

| Override this method            | Capability (`can_*`)   | Pipeline role | Service type registered |
| ------------------------------- | ---------------------- | ------------- | ----------------------- |
| `search_batch`                  | `can_search`           | `search`      | `search`                |
| `score_batch`                   | `can_score`            | `rerank`      | `score`                 |
| `decompose_query_batch`         | `can_decompose_query`  | `expander`    | `decompose_query`       |
| `fuse_batch`                    | `can_fuse`             | `merger`      | `fuse`                  |

(The role→service-type map lives in `src/routir/pipeline/pipeline.py` as `_role_to_service`.)

A single engine may implement several of these. `SentenceTransformerEngine` implements both
`search_batch` and `score_batch`, for example. Implement only the methods that apply; the
rest raise `NotImplementedError` and the capability stays off.

**`*_batch` is always the method to override.** The singular convenience wrappers
(`search`, `score`, `decompose_query`, `fuse`) just call the batch form with one item — don't
override them. The exception is legacy code like `MT5Reranker`, which overrode `score`; prefer
`score_batch` in new engines.

### The two-layer pattern (used by every engine in the repo)

1. A plain **model wrapper** class holding tokenizer + synchronous inference (`self.model.score(pairs)`).
2. An **`Engine` subclass** that adapts the wrapper to RoutIR's async batch interface and reads `config`.

Synchronous model calls inside an `async def` method are fine for GPU work; RoutIR batches
*requests* before calling you, so a blocking `self.model.generate(...)` is the norm. (`LLMEngine`
wraps local vLLM calls in `loop.run_in_executor` to avoid blocking the event loop — do that only
if concurrency across services matters.)

## Method contracts (read these before writing one)

The docstrings in `src/routir/models/abstract.py` are authoritative and have worked examples.
Summary of the data shapes:

- **`search_batch(queries, limit, **kwargs) -> List[Dict[str, float]]`**
  One `{docid: score}` dict per query, higher = more relevant. `limit` may be an `int` (same
  for all) or `List[int]` (per query) — normalize it first: `if isinstance(limit, int): limit = [limit] * len(queries)`.

- **`score_batch(queries, passages, candidate_length, **kwargs) -> List[List[float]]`**
  `passages` is **flattened across all queries**; `candidate_length[i]` is how many consecutive
  passages belong to `queries[i]`. The canonical implementation: default `candidate_length` to
  `[len(passages)]`, expand queries to align with passages, score all pairs, then regroup by
  `candidate_length`. Return one score list per query in passage order.

- **`decompose_query_batch(queries, limit, **kwargs) -> List[List[str]]`**
  One list of sub-queries per query. Used by the `expander{...}merger` pipeline form.

- **`fuse_batch(queries, batch_scores, **kwargs) -> List[Dict[str, float]]`**
  `batch_scores[i]` is a *list* of `{docid: score}` ranked lists to merge for `queries[i]`.
  Return one fused `{docid: score}` per query. See `_rrf` / `_score_fusion` in `models/fusion.py`.

### `Reranker` base class — free candidate retrieval

If your reranker needs first-stage candidates fetched for it, subclass `Reranker` (also in
`abstract.py`) instead of `Engine`. It implements `search_batch` for you: it pulls
`limit * rerank_multiplier` candidates from a configured `upstream_service`, fetches their text
via a `text_service` (`/content` endpoint), calls **your** `score_batch`, and returns the top
`limit`. You only implement `score_batch`. Config keys: `upstream_service`, `text_service`
(`{endpoint, collection}`), `rerank_topk_max` (default 100), `rerank_multiplier` (default 5).
If candidates are instead supplied by the pipeline (the `A >> B` form), a plain `Engine` with
`score_batch` is enough — the pipeline fetches document text and hands it to you.

## How an engine gets loaded

`Engine.load(class_name, ...)` (via the `FactoryEnabled` mixin in `src/routir/utils/__init__.py`)
walks all `Engine` subclasses and matches by **exact class name**. So the `"engine"` field in a
service config must equal your class's `__name__`. There is no decorator to register a normal
engine — just make sure the class is *imported* before config load (see "Shipping" below).

`auto_register("fuse")` (in `src/routir/processors/registry.py`) is a separate, lightweight path
used only for **stateless built-ins** like `RRF` and `ScoreFusion`: it instantiates the engine at
import time and registers it directly under its class name with no batching/caching. Use it for
parameterless fusion/expander rules; use the config path for anything with a model or state.

## Config & wiring

Config is JSON, parsed by `Config` (`src/routir/config/config.py`) and loaded by `load_config`
(`src/routir/config/load.py`). Minimal shape:

```json
{
  "file_imports": ["./my_engine.py"],
  "collections": [{ "name": "my-corpus", "doc_path": "/data/corpus.jsonl" }],
  "services": [
    {
      "name": "my-retriever",
      "engine": "MyEngineClassName",
      "config": { "index_path": "/data/index", "...": "engine-specific" },
      "cache": 1024, "cache_ttl": 600, "batch_size": 32, "max_wait_time": 0.05
    }
  ]
}
```

- `services[].engine` → your class name. `services[].config` is passed verbatim as `config=` to
  your `__init__` (merged with any `**kwargs`). Read your params off `self.config.get(...)`.
- `services[].name` is the API/pipeline identifier (the `"service"` field in requests).
- `index_path` starting with `hfds:<repo>` is auto-downloaded from HuggingFace Datasets at load.
- For each service, `load_config` instantiates the engine once and registers a processor per
  capability: a `BatchQueryProcessor` (search), `BatchPairwiseScoreProcessor` (score, unless
  `scoring_disabled`), and/or `BatchDecomposeQueryProcessor` (decompose). Batching/caching params
  (`batch_size`, `max_wait_time`, `cache`, `cache_ttl`, `cache_key_fields`, Redis options) come
  from the `ServiceConfig`, not your engine.
- `collections[]` register `/content` services (document text by id). Rerankers in `A >> B`
  pipelines and `Reranker.get_text` read from these.

## Shipping a new engine — two options

1. **External file (fastest, no install):** write a `.py` defining your `Engine` subclass and list
   it in `file_imports`. `load_all_extensions` (`src/routir/utils/extensions.py`) execs it before
   services load, so the class becomes discoverable by `Engine.load`. This is how every
   `examples/*_extension.py` works.
2. **Installed package / built-in:** add the class under `src/routir/models/` and export it from
   `src/routir/models/__init__.py` (the import is what makes it discoverable). Or ship a third-party
   package named `routir_*` or exposing a `routir.extensions` entry point — both are auto-loaded.

## The pipeline DSL (why roles matter)

`src/routir/pipeline/parser.py` parses strings into an AST executed by `SearchPipeline`
(`pipeline/pipeline.py`). Operators:

- `service%N` — call `service`, keep top `N`.
- `A >> B` — sequential; `B` is assigned role `rerank` and receives `A`'s results + doc text.
- `{A, B}Merger` — run `A`, `B` in parallel, fuse with `Merger` (must implement `fuse_batch`).
- `Expander{A, B}Merger` — `Expander` (must implement `decompose_query_batch`) makes sub-queries,
  each fans out to the branches, all results fused by `Merger`.
- `service[alias]` — name a stage so per-stage `runtime_kwargs` can target it.

So the role a stage plays in the DSL determines which capability/service-type is invoked. A fusion
engine is only reachable in `merger` position; an expander only in `expander` position. `config.pipeline_aliases`
defines named DSL shortcuts.

## Reference implementations

Concrete, copyable examples, by type:

- **Dense search:** `src/routir/models/st.py` (`SentenceTransformerEngine`, FAISS + sentence-transformers,
  fully config-driven), `src/routir/models/qwen3.py`, `src/routir/models/plaidx.py`.
- **Sparse search:** `examples/pyserini_extension.py` (BM25), `examples/pyterrier_extension.py`,
  `src/routir/models/lsr.py`.
- **Reranker (score):** `src/routir/models/mt5.py` (mT5), `examples/rank1_extension.py` and
  `examples/vllm_qwen3reranker_extension.py` (vLLM yes/no logprob rerankers), `src/routir/models/qwen3reranker.py`.
- **Fusion:** `src/routir/models/fusion.py` (`Fusion` engine + `RRF`/`ScoreFusion` via `auto_register`).
- **Query expander + LLM reranker:** `src/routir/models/llm_engine.py` (`LLMEngine`, implements both
  `score_batch` and `decompose_query_batch`, OpenAI-compatible API or local vLLM backend).
- **Relay (proxy to another RoutIR server):** `src/routir/models/relay.py`.

`examples/CLAUDE.md` is a dedicated, longer how-to for wrapping models, with full annotated
`score_batch`/`search_batch` skeletons — read it when implementing.

## Serving & testing

**Always run RoutIR and any Python in this repo through `uvx`** — never bare `routir`,
`python`, `pytest`, or `ruff`. `uvx` builds a throwaway venv on the fly so you don't touch a
conda/system environment. Install the local checkout with `"routir[<extras>] @ ."` and add the
model's runtime deps with `--with`. Optional-dep extras (`pyproject.toml`): `dense`
(faiss/numpy), `gpu`, `plaidx`, `sparse`, `vllm`, `grpc`, `dev` (pytest).

```bash
# Serve (REST on :5000; add --grpc for gRPC on :50051). Add a --with per runtime dep.
uvx --with transformers --with torch --with "routir[dense,grpc] @ ." \
    routir <config.json> --port 5000 --grpc

# Tests / lint
uvx --with "routir[dev] @ ." pytest        # from repo root
uvx ruff check src/                          # line length 130
```

REST endpoints (`src/routir/serve.py`): `POST /search`, `POST /score`, `POST /content`,
`POST /pipeline`, `GET /avail` (lists services by type — your new engine should appear here),
`GET /ping`. gRPC mirrors these in `src/routir/servicer.py`. Set `ROUTIR_API_KEY` (or `--api_key`)
to require an `Authorization: Bearer <token>` header on every route except `/ping`.

Smoke-test a reranker (`curl` is not Python, so it runs directly):

```bash
curl -X POST http://localhost:5000/score -H 'Content-Type: application/json' \
  -d '{"service":"my-reranker","query":"q","passages":["p1","p2"]}'
# -> {"scores":[...], "query":"q", "service":"my-reranker", ...}
```

`tests/_trivial_engine.py` is a minimal engine for transport/registry tests.

## Search-results output format

When persisting retrieval runs against a RoutIR endpoint, the recommended on-disk
format is **JSONL, one line per query**. Each line is a JSON object with these fields:

```json
{
  "endpoint": "http://compute01:5000",
  "pipeline": "{qwen3asr-emb8b%1000, qwen3-vl-8b%1000}RRF%100",
  "collection": "microvent",
  "query_id": "1",
  "query": "full query text",
  "results": [
    {"rank": 1, "doc_id": "LUoCjPhSGLhy4ftu_0001", "score": 0.497211},
    {"rank": 2, "doc_id": "VLVKwM0-X_AmiZjA_0000", "score": 0.484272}
  ]
}
```

- `endpoint` — the RoutIR endpoint the run was issued against.
- `pipeline` — the full pipeline DSL string used (this, not `model`/`index`, identifies the
  retrieval setup, since results come from composed hosted engines rather than a single index).
- `collection` — the collection name passed to the pipeline (may be `null` for pipelines
  with no reranking stage).
- `query_id` / `query` — the query identifier and its text.
- `results` — a list of `{rank, doc_id, score}` objects, `rank` starting at 1, ordered by
  `score` descending (higher = more relevant). The result-set size is controlled by the `%N`
  cuts in the pipeline string, not a top-level limit.

This supersedes the older `model`/`index`/`topk` schema produced by a
local-FAISS script before retrieval moved behind the RoutIR API.

## Warming up sidecar caches

Bytes-view backends (`TarSource`, `LocalPathSource` with glob, `TextJsonlSource`)
build sidecar indexes on first access — `.taridx` for tar shards, `.offsetmap`
for plain JSONL. A cold first request can stall while ~~~50–500 K members per
tar are scanned. Pre-build the sidecars before serving.

**Sidecar resolution chain** (`src/routir/collections/indexing/sidecar.py`):

1. Per-view `cache_dir` on the source spec (set this in the config).
2. Adjacent to the source file (only works on writable mounts — shared
   dataset dirs are typically read-only).
3. `${XDG_CACHE_HOME:-~/.cache}/routir/{taridx,offsetmap}/...` fallback.

Always set `cache_dir` per view when the dataset mount is read-only.
By convention we point at the in-tree `./.cache/` so it's discoverable.

**Warm one config:**

```bash
uvx --with-editable . -- python -m routir.collections.indexing.warmup \
    <config.json> --workers 32
```

Use `--view <name>` to scope to one view; `--force` to rebuild from scratch.

**Off the login node, via SLURM.** Use `scripts/warmup_slurm.sh`:

```bash
# One job for the whole config (small datasets only):
sbatch scripts/warmup_slurm.sh default_baseline_config.json

# One job per view (the merged config includes both multivent-raw (~4669
# shards/view, keyframe ~16 min) and microvent (28 shards/view, rides
# along); per-view jobs sweep both collections at once):
for v in keyframe video ocr_ppocrvl15 ocr_ppocrv5icdar ocr_pagectc \
         asr_qwen3asr1p7b asr_whisperxlargev3; do
  sbatch scripts/warmup_slurm.sh default_baseline_config.json "$v"
done
```

> **Cluster citizenship — hard rule.** *Never* submit one slurm job (or
> array task) per shard. With thousands of fine-grained tasks the scheduler
> falls over for everyone. Always submit one job per **coarse logical unit**
> (per view, per dataset, per model) and use `--workers N` for internal
> multiprocess fan-out across the fine-grained units within. The sbatch
> script takes 32 cpus and exposes them via `$SLURM_CPUS_PER_TASK`.

## Layout

```
src/routir/
  models/        Engine subclasses (where new engines live) + abstract.py base classes
  processors/    BatchProcessor/caching layer + ProcessorRegistry (registry.py) + auto_register
  pipeline/      DSL parser (parser.py), executor (pipeline.py), aliases, cache
  config/        Pydantic Config schema (config.py) + load_config (load.py)
  client/        Async/sync REST + gRPC client
  utils/         FactoryEnabled, extension loading, FAISS indexing helpers
  serve.py       REST server + CLI (`routir` entry point)
  servicer.py    gRPC servicer
```
