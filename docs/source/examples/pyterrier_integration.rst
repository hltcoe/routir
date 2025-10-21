PyTerrier Integration
=====================

This example shows how to use PyTerrier models with ``routir``.

Overview
--------

PyTerrier is a Python framework for information retrieval experimentation. You can
use PyTerrier models and indexes with ``routir`` by creating a custom extension.

.. note::
   This example requires the ``python-terrier`` package:

   .. code-block:: bash

      pip install python-terrier

Building the Index
------------------

First, build a PyTerrier index using the provided example script:

.. code-block:: bash

   python ./examples/pyterrier_extension.py

.. note::
   **TODO**: The ``pyterrier_extension.py`` script should be added to the examples directory.

   Expected functionality:
   * Load documents from a collection
   * Build a PyTerrier index
   * Save the index to disk

Configuration
-------------

Create a configuration file ``pyterrier_example_config.json``:

.. code-block:: json

   {
     "collections": [
       {
         "name": "cord19",
         "doc_path": "/path/to/cord19.jsonl",
         "id_field": "id",
         "content_field": "text"
       }
     ],
     "services": [
       {
         "name": "pyterrier-cord",
         "engine": "PyTerrierEngine",
         "config": {
           "index_path": "/path/to/pyterrier-index"
         }
       }
     ],
     "file_imports": [
       "./examples/pyterrier_extension.py"
     ]
   }

.. note::
   The ``file_imports`` field loads your custom PyTerrier engine implementation.

Starting the Service
--------------------

Start the ``routir`` service with PyTerrier support:

.. code-block:: bash

   uvx --with python-terrier routir ./examples/pyterrier_example_config.json --port 8000

The ``--with python-terrier`` flag ensures the PyTerrier package is available.

Making Queries
--------------

Query the PyTerrier service via the API:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/search",
       json={
           "service": "pyterrier-cord",
           "query": "covid-19 symptoms and treatment",
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
       "service": "pyterrier-cord",
       "query": "covid-19 symptoms and treatment",
       "limit": 15
     }'

Custom PyTerrier Engine
-----------------------

.. note::
   **TODO**: Add complete implementation of PyTerrierEngine

   The custom engine should:

   * Inherit from ``routir.models.Engine``
   * Load PyTerrier index in ``__init__``
   * Implement ``search_batch`` method
   * Handle query processing and result formatting

Example skeleton:

.. code-block:: python

   from routir.models import Engine
   import pyterrier as pt

   class PyTerrierEngine(Engine):
       def __init__(self, name=None, config=None, **kwargs):
           super().__init__(name, config, **kwargs)
           if not pt.started():
               pt.init()
           self.index = pt.IndexFactory.of(self.config["index_path"])
           self.retriever = pt.BatchRetrieve(self.index)

       async def search_batch(self, queries, limit=20, **kwargs):
           # TODO: Implement batch search
           pass

Performance Considerations
--------------------------

* PyTerrier uses Java-based indexing, which may have different memory characteristics
* Consider using batch processing for better throughput
* PyTerrier BM25 is CPU-bound, so CPU scaling is important

Next Steps
----------

* See :doc:`pyserini_integration` for another sparse retrieval option
* Explore :doc:`hybrid_search` to combine PyTerrier with dense retrieval
* Check the :doc:`../api/models` documentation for available engines
