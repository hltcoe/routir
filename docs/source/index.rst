Welcome to Routir's Documentation!
====================================

``routir`` is a simple and fast search service for hosting state-of-the-art retrieval models.
It provides a flexible framework for building and deploying search services with support for
multiple search engines, processors, and pipeline configurations.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Key Features
------------

* **Multiple Search Engines**: Support for PLAID-X (ColBERT), LSR (SPLADE), Qwen3, and more
* **Flexible Pipeline System**: Create complex search pipelines with reranking and fusion
* **Batch Processing**: Efficient request batching with configurable parameters
* **Caching Support**: LRU and Redis caching for improved performance
* **Extensible Architecture**: Easy to add custom engines and processors
* **GPU Acceleration**: Optional GPU support for faster inference
* **REST API**: Simple HTTP API for search and scoring

Quick Start
-----------

Install ``routir``:

.. code-block:: bash

   pip install routir

Start a search service:

.. code-block:: bash

   routir config.json

Or using uvx:

.. code-block:: bash

   uvx routir config.json

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   getting_started
   configuration
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/models
   api/processors
   api/pipeline
   api/config
   api/utils
   api/serve

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   contributing
   architecture
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
