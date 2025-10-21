Getting Started
===============

This guide will help you get started with Routir, from installation to running your first search service.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.9 or higher
* pip or uv package manager

Basic Installation
~~~~~~~~~~~~~~~~~~

Install Routir using pip:

.. code-block:: bash

   pip install routir

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Routir provides extras to install only the dependencies you need:

.. code-block:: bash

   pip install "routir[dense,gpu]"
   pip install "routir[plaidx,sparse]"

Available extras: ``dense``, ``gpu``, ``plaidx``, ``sparse``

Quick Start
-----------

1. Create a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``config.json`` with four main blocks:

* ``services``: List of search engines to serve
* ``collections``: Document collections
* ``server_imports``: External Routir endpoints to mirror
* ``file_imports``: Custom Python engines to load

.. code-block:: json

   {
     "server_imports": ["http://compute01:5000"],
     "file_imports": ["./custom_engine.py"],
     "services": [
       {
         "name": "qwen3-neuclir",
         "engine": "Qwen3",
         "cache": 1024,
         "batch_size": 32,
         "config": {
           "index_path": "hfds:routir/neuclir-qwen3-8b-faiss-PQ2048x4fs",
           "api_key": "YOUR_API_KEY",
           "embedding_base_url": "https://api.fireworks.ai/inference/v1/",
           "embedding_model_name": "accounts/fireworks/models/qwen3-embedding-8b"
         }
       }
     ],
     "collections": [
       {
         "name": "neuclir",
         "doc_path": "./neuclir-doc.jsonl"
       }
     ]
   }

For Redis caching, add ``cache_redis_url`` and ``cache_redis_kwargs`` to the service config.

2. Start the Service
~~~~~~~~~~~~~~~~~~~~

Start the search service (default port 8000):

.. code-block:: bash

   routir config.json --port 5000

Or use uvx to run without installing:

.. code-block:: bash

   uvx --with transformers --with torch routir config.json

Use ``--with`` to specify additional packages needed for your models.

3. Make Your First Query
~~~~~~~~~~~~~~~~~~~~~~~~~

Using Python:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:5000/search",
       json={
           "service": "my-search",
           "query": "what is information retrieval?",
           "limit": 10
       }
   )

   results = response.json()
   print(results["scores"])  # Dict of {doc_id: score}

Using cURL:

.. code-block:: bash

   curl -X POST http://localhost:5000/search \
     -H "Content-Type: application/json" \
     -d '{
       "service": "my-search",
       "query": "what is information retrieval?",
       "limit": 10
     }'

Understanding the Response
~~~~~~~~~~~~~~~~~~~~~~~~~~

The search endpoint returns a JSON response:

.. code-block:: json

   {
     "query": "what is information retrieval?",
     "scores": {
       "doc_1": 0.95,
       "doc_2": 0.87,
       "doc_3": 0.82
     },
     "service": "my-search",
     "processed": true,
     "cached": false,
     "timestamp": 1234567890.123
   }

Fields:
* ``query``: The original query
* ``scores``: Dictionary mapping document IDs to relevance scores
* ``service``: The service that processed the query
* ``processed``: Whether the request was processed successfully
* ``cached``: Whether the result came from cache
* ``timestamp``: Unix timestamp of when the request was processed

Available Endpoints
-------------------

Routir provides several REST API endpoints:

Search Endpoint
~~~~~~~~~~~~~~~

``POST /search``

Search for documents using a configured search service.

**Request:**

.. code-block:: json

   {
     "service": "service-name",
     "query": "search query",
     "limit": 10
   }

**Response:**

.. code-block:: json

   {
     "query": "search query",
     "scores": {"doc_id": score, ...},
     "service": "service-name",
     "processed": true,
     "cached": false,
     "timestamp": 1234567890.123
   }

Score Endpoint
~~~~~~~~~~~~~~

``POST /score``

Score query-passage pairs using a reranker.

**Request:**

.. code-block:: json

   {
     "service": "reranker-name",
     "query": "search query",
     "passages": ["passage 1", "passage 2", "passage 3"]
   }

**Response:**

.. code-block:: json

   {
     "query": "search query",
     "scores": [0.95, 0.82, 0.67],
     "service": "reranker-name",
     "processed": true,
     "timestamp": 1234567890.123
   }

Content Endpoint
~~~~~~~~~~~~~~~~

``POST /content``

Retrieve document content by ID.

**Request:**

.. code-block:: json

   {
     "collection": "collection-name",
     "id": "doc_123"
   }

**Response:**

.. code-block:: json

   {
     "id": "doc_123",
     "text": "Document content...",
     "title": "Document Title"
   }

Pipeline Endpoint
~~~~~~~~~~~~~~~~~

``POST /pipeline``

Execute a custom search pipeline with dynamic composition.

**Request:**

.. code-block:: json

   {
     "pipeline": "{qwen3-neuclir, plaidx-neuclir}RRF%50 >> rank1",
     "collection": "neuclir",
     "query": "which team is the world series champion in 2020?"
   }

Pipeline syntax:
* ``{s1, s2}RRF%50``: Fuse results from s1 and s2 using RRF, keep top 50
* ``>>`` : Pass results to next stage
* ``service_name``: Rerank with service

**Response:**

.. code-block:: json

   {
     "query": "which team is the world series champion in 2020?",
     "scores": {"doc_id": score, ...},
     "pipeline": "{qwen3-neuclir, plaidx-neuclir}RRF%50 >> rank1"
   }

Health Check
~~~~~~~~~~~~

``GET /ping``

Check if the service is running.

**Response:**

.. code-block:: json

   {
     "status": "pong"
   }

Available Services
~~~~~~~~~~~~~~~~~~

``GET /avail``

List all available services.

**Response:**

.. code-block:: json

   {
     "search": ["service1", "service2"],
     "score": ["reranker1"],
     "content": ["collection1"],
     "decompose_query": [],
     "fuse": []
   }

Extension Examples
------------------

Integrate other IR toolkits by implementing custom engines. Import them via ``file_imports`` in config.

PyTerrier
~~~~~~~~~

.. code-block:: bash

   python ./examples/pyterrier_extension.py  # build index
   uvx --with python-terrier routir ./examples/pyterrier_example_config.json

Pyserini
~~~~~~~~

.. code-block:: bash

   uvx --with pyserini routir ./examples/pyserini_example_config.json

Rank1
~~~~~

.. code-block:: bash

   uvx --with mteb==1.39.0 --with vllm routir ./examples/rank1_example_config.json

FAISS Indexing
--------------

Generate FAISS indexes from encoded vectors:

.. code-block:: bash

   python -m routir.utils.faiss_indexing \
     ./encoded_vectors/ ./faiss_index.PQ2048x4fs.IP/ \
     --index_string "PQ2048x4fs" --use_gpu --sampling_rate 0.25

Next Steps
----------

* Learn about :doc:`configuration` options
* Check out :doc:`examples/index` for real-world examples
* Read the :doc:`api/models` documentation for available search engines
