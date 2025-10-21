Examples
========

Real-world examples of using Routir with different search engines and frameworks.

.. toctree::
   :maxdepth: 2

   pyterrier_integration
   pyserini_integration
   hybrid_search

Overview
--------

These examples demonstrate how to use Routir in various scenarios:

* **PyTerrier Integration**: Use PyTerrier models with Routir
* **Pyserini Integration**: Serve Pyserini BM25 indexes via Routir
* **Hybrid Search**: Combine multiple search engines

Each example includes:

* Complete configuration files
* Sample code for building indexes (where applicable)
* API usage examples
* Performance considerations

Quick Example
-------------

Here's a quick example of using Routir with a search service:

.. code-block:: python

   import requests

   # Search for documents
   response = requests.post(
       "http://localhost:5000/search",
       json={
           "service": "my-search",
           "query": "information retrieval",
           "limit": 10
       }
   )

   results = response.json()

   # Print top results
   for doc_id, score in sorted(results["scores"].items(),
                                key=lambda x: -x[1])[:5]:
       print(f"{doc_id}: {score:.4f}")

Next Steps
----------

Browse the examples to find one that matches your use case, or start with
:doc:`pyterrier_integration` for a simple integration example.
