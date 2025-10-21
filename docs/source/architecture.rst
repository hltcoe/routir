Architecture
============

This document describes the architecture of Routir.

Overview
--------

Routir is built with a modular architecture consisting of several key components:

- **Models**: Search and reranking models (PLAID-X, LSR, Qwen3, mT5, etc.)
- **Processors**: Request batching, caching, and content processing
- **Pipeline**: Composable search pipelines with multiple stages
- **Config**: JSON-based configuration system

Components
----------

Models
~~~~~~

Models implement the core search and scoring functionality. All models inherit from a base abstract class
and implement the required interface methods.

Processors
~~~~~~~~~~

Processors handle request batching, caching, and content transformation. They provide asynchronous
and batch processing capabilities for improved performance.

Pipeline
~~~~~~~~

The pipeline system allows composing multiple search stages together, enabling complex workflows
like hybrid search, reranking, and score fusion.

Configuration
~~~~~~~~~~~~~

Services are configured using JSON files that specify models, processors, caching settings,
and pipeline configurations.
