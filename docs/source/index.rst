RoutIR: Fast Server for Hosting Retrieval Models for RAG
=========================================================

**RoutIR** is a Python package that provides a simple and efficient wrapper around arbitrary retrieval models, including first stage retrieval, reranking, query expansion, and result fusion, with efficient asynchronous query batching and serving.

.. image:: https://img.shields.io/badge/Python->=3.9-blue?logo=python&logoColor=white
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Quick Start
-----------

Install:

.. code-block:: bash

   pip install routir
   pip install "routir[dense,gpu]"

Start the service:

.. code-block:: bash

   routir config.json --port 5000

Or use uvx:

.. code-block:: bash

   uvx --with transformers --with torch routir config.json

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
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
