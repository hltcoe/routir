Quick Start
===========

This guide will help you get started with Routir quickly.

Installation
------------

Install Routir using pip:

.. code-block:: bash

   pip install routir

Basic Usage
-----------

Start a search service with a configuration file:

.. code-block:: bash

   routir config.json

Or using uvx:

.. code-block:: bash

   uvx routir config.json

Faiss Indexing
--------------

Build a Faiss index from encoded vectors:

.. code-block:: bash

   python -m routir.utils.faiss_indexing \
   ./encoded_vectors/ ./faiss_index.PQ2048x4fs.IP/ \
   --index_string "PQ2048x4fs" --use_gpu --sampling_rate 0.25

Extension Examples
------------------

PyTerrier Integration
~~~~~~~~~~~~~~~~~~~~~

Build the index:

.. code-block:: bash

   python ./examples/pyterrier_extension.py

Serve the index:

.. code-block:: bash

   uvx --with python-terrier routir ./examples/pyterrier_example_config.json --port 8000

Query the service:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/search",
       json={
           "service": "pyterrier-cord",
           "query": "my test query",
           "limit": 15
       }
   )

   results = response.json()

Pyserini Integration
~~~~~~~~~~~~~~~~~~~~

Serve a Pyserini index:

.. code-block:: bash

   uvx --with pyserini routir ./examples/pyserini_example_config.json --port 8000

Query the service:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/search",
       json={
           "service": "pyserinibm25-neuclir-zho-dt",
           "query": "my test query",
           "limit": 15
       }
   )

   results = response.json()
