Getting Started
===============

This guide will help you get started with Routir, from installation to running your first search service.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.8 or higher
* pip or uv package manager

Basic Installation
~~~~~~~~~~~~~~~~~~

Install Routir using pip:

.. code-block:: bash

   pip install routir

Or using uv:

.. code-block:: bash

   uv pip install routir

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU support (PLAID-X, neural rerankers):

.. code-block:: bash

   pip install routir[gpu]

For all features:

.. code-block:: bash

   pip install routir[all]

Quick Start
-----------

1. Create a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``config.json`` file that defines your search services and document collections:

.. code-block:: json

   {
     "collections": [
       {
         "name": "my-collection",
         "doc_path": "/path/to/documents.jsonl",
         "id_field": "id",
         "content_field": "text"
       }
     ],
     "services": [
       {
         "name": "my-search",
         "engine": "PLAIDX",
         "config": {
           "index_path": "/path/to/plaidx-index",
           "checkpoint": "colbert-ir/colbertv2.0"
         }
       }
     ]
   }

2. Start the Service
~~~~~~~~~~~~~~~~~~~~

Start the search service:

.. code-block:: bash

   routir config.json

By default, the service runs on ``http://localhost:5000``. You can specify a different port:

.. code-block:: bash

   routir config.json --port 8000 --host 0.0.0.0

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

From the command line using cURL:

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

Execute a custom search pipeline.

**Request:**

.. code-block:: json

   {
     "pipeline": "search[my-search] | rerank[my-reranker]@10",
     "collection": "my-collection",
     "query": "search query"
   }

**Response:**

.. code-block:: json

   {
     "query": "search query",
     "scores": {"doc_id": score, ...}
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

Next Steps
----------

* Learn about :doc:`configuration` options
* Check out :doc:`examples/index` for real-world examples
* Read the :doc:`api/models` documentation for available search engines
