Configuration Guide
===================

This guide explains how to configure ``routir`` services, collections, and pipelines.

Configuration File Structure
-----------------------------

A ``routir`` configuration file is a JSON document with the following structure:

.. code-block:: json

   {
     "collections": [...],
     "services": [...],
     "file_imports": [...],
     "server_imports": [...],
     "dynamic_pipeline": true
   }

Collections
-----------

Collections define document sets that can be searched and retrieved.

Basic Collection Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "name": "my-docs",
     "doc_path": "/path/to/documents.jsonl",
     "id_field": "id",
     "content_field": "text"
   }

Collection Parameters
~~~~~~~~~~~~~~~~~~~~~

* ``name`` (required): Unique identifier for the collection
* ``doc_path`` (required): Path to the JSONL document file
* ``id_field``: JSON field containing document IDs (default: ``"id"``)
* ``content_field``: JSON field(s) containing document text (default: ``"text"``)

  - Can be a string: ``"text"``
  - Can be a list: ``["title", "body"]`` (will be concatenated)

* ``offset_source``: Method for document indexing (default: ``"offsetfile"``)

  - ``"offsetfile"``: Use pre-built offset map
  - ``"msmarco_seg"``: Use MS MARCO segment format

* ``cache_path``: Custom path for offset map cache
* ``id_to_lang_mapping``: Path to pickle file mapping doc IDs to language codes

Example with Multiple Content Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "name": "papers",
     "doc_path": "/data/papers.jsonl",
     "id_field": "paper_id",
     "content_field": ["title", "abstract", "body"],
     "cache_path": "/cache/papers.offsetmap"
   }

Services
--------

Services define search engines, rerankers, and other processing components.

Basic Service Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "name": "my-search",
     "engine": "PLAIDX",
     "config": {
       "index_path": "/path/to/index",
       "checkpoint": "colbert-ir/colbertv2.0"
     }
   }

Service Parameters
~~~~~~~~~~~~~~~~~~

Common Parameters
^^^^^^^^^^^^^^^^^

* ``name`` (required): Unique service identifier
* ``engine`` (required): Engine class name (e.g., ``"PLAIDX"``, ``"LSR"``, ``"Qwen3"``)
* ``config`` (required): Engine-specific configuration dictionary
* ``processor``: Processor type (default: ``"BatchQueryProcessor"``)

  - ``"BatchQueryProcessor"``: Batch requests for efficiency
  - ``"AsyncQueryProcessor"``: Process requests independently

* ``cache``: Cache size in entries (default: ``-1`` = disabled)
* ``cache_ttl``: Cache time-to-live in seconds (default: ``600``)
* ``batch_size``: Maximum batch size (default: ``32``)
* ``max_wait_time``: Maximum wait time for batching in seconds (default: ``0.05``)
* ``cache_key_fields``: Fields to use for cache keys (default: ``["query", "limit"]``)
* ``scoring_disabled``: Disable scoring functionality (default: ``false``)

Redis Caching
^^^^^^^^^^^^^

For distributed caching:

.. code-block:: json

   {
     "name": "my-search",
     "engine": "PLAIDX",
     "config": {...},
     "cache": 10000,
     "cache_redis_url": "redis://localhost:6379/0",
     "cache_redis_kwargs": {
       "encoding": "utf-8",
       "decode_responses": true
     }
   }

Available Engines
-----------------

PLAIDX (ColBERT)
~~~~~~~~~~~~~~~~

Dense retrieval using ColBERT's late interaction.

.. code-block:: json

   {
     "name": "colbert-search",
     "engine": "PLAIDX",
     "config": {
       "index_path": "/path/to/plaidx-index",
       "checkpoint": "colbert-ir/colbertv2.0",
       "use_gpu": true,
       "gpu_assignment": 0
     }
   }

Configuration options:

* ``index_path``: Path to the PLAID-X index directory
* ``checkpoint``: HuggingFace model ID or local path
* ``use_gpu``: Enable GPU inference (default: ``false``)
* ``gpu_assignment``: GPU device ID (default: ``0``)
* ``passage_mapping``: Path to passage-to-document mapping file
* ``id_to_subset_mapping``: Path to subset mapping pickle file

LSR (SPLADE)
~~~~~~~~~~~~

Learned sparse retrieval using SPLADE models.

.. code-block:: json

   {
     "name": "splade-search",
     "engine": "LSR",
     "config": {
       "index_path": "/path/to/anserini-index",
       "model_name": "naver/splade-cocondenser-ensembledistil",
       "max_length": 256,
       "batch_size": 32,
       "max_terms": 512
     }
   }

Configuration options:

* ``index_path``: Path to Anserini index
* ``model_name``: HuggingFace SPLADE model ID
* ``max_length``: Maximum sequence length
* ``batch_size``: Encoding batch size
* ``max_terms``: Maximum number of expansion terms
* ``id_to_subset_mapping``: Path to subset mapping pickle file

Qwen3
~~~~~

Dense retrieval using Qwen3 embeddings.

.. code-block:: json

   {
     "name": "qwen3-search",
     "engine": "Qwen3",
     "config": {
       "index_path": "/path/to/faiss-index",
       "embedding_model_name": "Qwen/Qwen3-Embedding-8B",
       "max_length": 8192,
       "batch_size": 8
     }
   }

Configuration options:

* ``index_path``: Path to FAISS index directory (contains ``index.faiss`` and ``index.ids``)
* ``embedding_model_name``: Model name for embeddings
* ``max_length``: Maximum sequence length
* ``batch_size``: Encoding batch size
* ``embedding_base_url``: Optional API endpoint for embeddings
* ``api_key``: API key for remote embedding service

MT5 Reranker
~~~~~~~~~~~~

Neural reranking using mT5 models.

.. code-block:: json

   {
     "name": "mt5-reranker",
     "engine": "MT5Reranker",
     "config": {
       "model_name_or_path": "unicamp-dl/mt5-base-en-msmarco",
       "q_max_length": 180,
       "d_max_length": 512,
       "batch_size": 32,
       "use_gpu": true,
       "upstream_service": {
         "engine": "PLAIDX",
         "config": {...}
       },
       "text_service": {
         "endpoint": "http://localhost:5000",
         "collection": "my-collection"
       }
     }
   }

Configuration options:

* ``model_name_or_path``: HuggingFace model ID or local path
* ``q_max_length``: Maximum query length
* ``d_max_length``: Maximum document length
* ``batch_size``: Reranking batch size
* ``use_gpu``: Enable GPU inference
* ``upstream_service``: Upstream retrieval configuration
* ``text_service``: Service for retrieving document text
* ``rerank_topk_max``: Maximum candidates to rerank (default: ``100``)
* ``rerank_multiplier``: Retrieval multiplier (default: ``5``)

Relay
~~~~~

Forward requests to remote or local services.

.. code-block:: json

   {
     "name": "remote-search",
     "engine": "Relay",
     "config": {
       "service": "target-service",
       "endpoint": "http://remote-server:5000"
     }
   }

Fusion
~~~~~~

Combine results from multiple search engines.

.. code-block:: json

   {
     "name": "hybrid-search",
     "engine": "Fusion",
     "config": {
       "upstream_service": [
         {
           "engine": "PLAIDX",
           "config": {...}
         },
         {
           "engine": "LSR",
           "config": {...}
         }
       ],
       "fusion_method": "RRF",
       "fusion_args": {
         "smoothing_k": 60
       }
     }
   }

Fusion methods:

* ``"RRF"``: Reciprocal Rank Fusion
* ``"score"``: Simple score summation

File Imports
------------

Load custom engines and processors from Python files:

.. code-block:: json

   {
     "file_imports": [
       "./extensions/custom_engine.py",
       "./extensions/custom_processor.py"
     ]
   }

Server Imports
--------------

Automatically import services from remote servers:

.. code-block:: json

   {
     "server_imports": [
       "http://remote-server-1:5000",
       "http://remote-server-2:5000"
     ]
   }

Complete Example
----------------

.. code-block:: json

   {
     "collections": [
       {
         "name": "wiki-docs",
         "doc_path": "/data/wikipedia.jsonl",
         "id_field": "id",
         "content_field": ["title", "text"],
         "cache_path": "/cache/wiki.offsetmap"
       }
     ],
     "services": [
       {
         "name": "colbert-wiki",
         "engine": "PLAIDX",
         "config": {
           "index_path": "/indexes/colbert-wiki",
           "checkpoint": "colbert-ir/colbertv2.0",
           "use_gpu": true
         },
         "cache": 10000,
         "cache_ttl": 3600,
         "batch_size": 64,
         "max_wait_time": 0.1
       },
       {
         "name": "splade-wiki",
         "engine": "LSR",
         "config": {
           "index_path": "/indexes/splade-wiki",
           "model_name": "naver/splade-cocondenser-ensembledistil",
           "max_length": 256,
           "batch_size": 32,
           "max_terms": 512
         }
       },
       {
         "name": "hybrid-wiki",
         "engine": "Fusion",
         "config": {
           "upstream_service": [
             {"engine": "PLAIDX", "config": {...}},
             {"engine": "LSR", "config": {...}}
           ],
           "fusion_method": "RRF"
         }
       }
     ],
     "file_imports": ["./extensions/custom_models.py"],
     "dynamic_pipeline": true
   }

.. note::
   For production deployments, consider using Redis caching and adjusting
   batch sizes and wait times based on your latency requirements.

.. seealso::

   * :doc:`examples/hybrid_search` - Example hybrid search configuration
   * :doc:`api/models` - Complete engine API reference
