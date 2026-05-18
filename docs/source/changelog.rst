Changelog
=========

All notable changes to this project will be documented here.

Version 0.0.2
-------------

- Support for arbitrary sentence-transformers models using SentenceTransformerEngine
- Pre-defined ``pipeline_aliases`` in the config: name common pipeline DSL fragments and reuse them at request time
- ``--api_key`` (or ``ROUTIR_API_KEY`` env var) enables Bearer-token authentication on all endpoints except ``/ping``
- ``collection`` field is optional on ``/pipeline``; required only when the pipeline contains a reranking stage

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

