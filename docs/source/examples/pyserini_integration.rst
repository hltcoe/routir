Pyserini Integration
====================

This example shows how to use Pyserini BM25 indexes with ``routir``.

Overview
--------

Pyserini is a Python toolkit for reproducible information retrieval research. It provides
easy access to Lucene indexes and pre-built collections.

.. note::
   This example requires the ``pyserini`` package:

   .. code-block:: bash

      pip install pyserini

Using Pre-built Indexes
-----------------------

Pyserini provides many pre-built indexes for popular collections. You can use these
directly with ``routir`` without building your own index.

Configuration
-------------

Create a configuration file ``pyserini_example_config.json``:

.. code-block:: json

   {
     "services": [
       {
         "name": "pyserinibm25-neuclir-zho-dt",
         "engine": "PyseriniEngine",
         "config": {
           "index_name": "neuclir22-zh-dt",
           "language": "zh"
         }
       }
     ],
     "file_imports": [
       "./examples/pyserini_extension.py"
     ]
   }

.. note::
   **TODO**: The ``pyserini_extension.py`` file should be added to the examples directory
   with a custom PyseriniEngine implementation.

Available Pre-built Indexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pyserini provides many pre-built indexes:

* MS MARCO Passage: ``"msmarco-v1-passage"``
* MS MARCO Document: ``"msmarco-v1-doc"``
* NeuCLIR Chinese: ``"neuclir22-zh-dt"``
* NeuCLIR Russian: ``"neuclir22-ru-dt"``
* NeuCLIR Persian: ``"neuclir22-fa-dt"``

See the `Pyserini documentation <https://github.com/castorini/pyserini>`_ for a complete list.

Starting the Service
--------------------

Start the ``routir`` service with Pyserini support:

.. code-block:: bash

   uvx --with pyserini routir ./examples/pyserini_example_config.json --port 8000

The ``--with pyserini`` flag ensures the Pyserini package is available.

Making Queries
--------------

Query the Pyserini service via the API:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/search",
       json={
           "service": "pyserinibm25-neuclir-zho-dt",
           "query": "新冠疫情",  # "COVID-19 pandemic" in Chinese
           "limit": 15
       }
   )

   results = response.json()
   print(f"Found {len(results['scores'])} results")

   # Print top 5 results
   for doc_id, score in sorted(results["scores"].items(),
                                key=lambda x: -x[1])[:5]:
       print(f"  {doc_id}: {score:.4f}")

Using cURL
~~~~~~~~~~

.. code-block:: bash

   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{
       "service": "pyserinibm25-neuclir-zho-dt",
       "query": "新冠疫情",
       "limit": 15
     }'

Custom Pyserini Engine
----------------------

.. note::
   **TODO**: Add complete implementation of PyseriniEngine

   The custom engine should:

   * Inherit from ``routir.models.Engine``
   * Load Pyserini index in ``__init__``
   * Implement ``search_batch`` method
   * Handle language-specific processing if needed

Example skeleton:

.. code-block:: python

   from routir.models import Engine
   from pyserini.search.lucene import LuceneSearcher

   class PyseriniEngine(Engine):
       def __init__(self, name=None, config=None, **kwargs):
           super().__init__(name, config, **kwargs)

           # Load from pre-built index or local path
           if "index_name" in self.config:
               self.searcher = LuceneSearcher.from_prebuilt_index(
                   self.config["index_name"]
               )
           else:
               self.searcher = LuceneSearcher(self.config["index_path"])

           # Set language if specified
           if "language" in self.config:
               self.searcher.set_language(self.config["language"])

       async def search_batch(self, queries, limit=20, **kwargs):
           # TODO: Implement batch search
           pass

Building Custom Indexes
------------------------

To build your own Pyserini index:

.. code-block:: bash

   # Index a JSONL collection
   python -m pyserini.index.lucene \
     --collection JsonCollection \
     --input /path/to/documents \
     --index /path/to/output/index \
     --generator DefaultLuceneDocumentGenerator \
     --threads 8 \
     --storePositions --storeDocvectors --storeRaw

Then use the local index path in your configuration:

.. code-block:: json

   {
     "config": {
       "index_path": "/path/to/output/index"
     }
   }

Multilingual Search
-------------------

Pyserini supports multilingual search with language-specific analyzers:

.. code-block:: json

   {
     "services": [
       {
         "name": "chinese-search",
         "engine": "PyseriniEngine",
         "config": {
           "index_name": "neuclir22-zh-dt",
           "language": "zh"
         }
       },
       {
         "name": "russian-search",
         "engine": "PyseriniEngine",
         "config": {
           "index_name": "neuclir22-ru-dt",
           "language": "ru"
         }
       }
     ]
   }

Performance Considerations
--------------------------

* Pyserini uses Lucene/Anserini for indexing, which is very efficient
* BM25 scoring is CPU-bound
* Consider using batch processing for higher throughput
* Pre-built indexes are downloaded on first use and cached

Next Steps
----------

* See :doc:`pyterrier_integration` for an alternative sparse retrieval framework
* Explore :doc:`hybrid_search` to combine BM25 with dense retrieval
* Check the :doc:`../api/models` documentation for available engines
