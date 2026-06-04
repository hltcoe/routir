Changelog
=========

All notable changes to this project will be documented here.

Version 0.0.2
-------------

- Support for arbitrary sentence-transformers models using ``SentenceTransformerEngine``
- New ``LLMEngine`` for pointwise reranking and query decomposition via vLLM or any OpenAI-compatible endpoint, with optional runtime-supplied prompts
- gRPC server (opt-in via ``--grpc``) alongside the REST endpoints, sharing the same processor registry; new ``routir.client`` subpackage with a unified ``AsyncClient``/``Client`` that auto-discovers gRPC from a REST endpoint and falls back to REST when gRPC is unreachable
- Pipeline-level result cache: enable via top-level ``pipeline_cache`` / ``pipeline_cache_ttl`` / ``pipeline_cache_redis_url`` in the config; keyed on the canonical pipeline AST + query + collection + the subset of ``runtime_kwargs`` referenced by the pipeline, so whitespace and alias-vs-expansion collapse to one entry
- ``server_imports`` now also proxies remote ``content`` services (collections); a new local cache (``relay_content_cache``) avoids re-fetching document text on every rerank
- Pre-defined ``pipeline_aliases`` in the config: name common pipeline DSL fragments and reuse them at request time
- ``--api_key`` (or ``ROUTIR_API_KEY`` env var) enables Bearer-token authentication on all REST and gRPC endpoints except ``/ping``; CORS preflight requests are exempt
- ``collection`` field is optional on ``/pipeline``; required only when the pipeline contains a reranking stage
- ASCII banner with package version printed on startup

Version 0.0.1
-------------

- Initial release
- Support for PLAID-X (ColBERT) search
- Support for LSR (SPLADE) search
- Support for Qwen3 reranking
- Support for mT5 reranking
- Batch processing and caching
- REST API for search and scoring
- Pipeline system for multi-stage search
- PyTerrier and Pyserini integration examples

Version 0.0.1b9
---------------

- Download index from HFDS support
- Bug fixes and improvements

