Hybrid Search Example
=====================

.. note::
   **TODO**: This example is under development.

   This section will demonstrate:

   * Combining dense (PLAIDX) and sparse (LSR/BM25) retrieval
   * Using the Fusion engine for result combination
   * Comparing RRF vs score-based fusion
   * Benchmarking hybrid search performance

Placeholder Content
-------------------

Overview
~~~~~~~~

Hybrid search combines multiple retrieval methods to achieve better results than
any single method alone. Common combinations include:

* Dense (ColBERT/PLAIDX) + Sparse (BM25/SPLADE)
* Multiple dense models
* Retrieval + Reranking

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "services": [
       {
         "name": "hybrid-search",
         "engine": "Fusion",
         "config": {
           "upstream_service": [
             {
               "engine": "PLAIDX",
               "config": {
                 "index_path": "/path/to/colbert-index",
                 "checkpoint": "colbert-ir/colbertv2.0"
               }
             },
             {
               "engine": "LSR",
               "config": {
                 "index_path": "/path/to/splade-index",
                 "model_name": "naver/splade-cocondenser-ensembledistil"
               }
             }
           ],
           "fusion_method": "RRF",
           "fusion_args": {
             "smoothing_k": 60
           }
         }
       }
     ]
   }

Coming Soon
-----------

This example is being written. For now, see :doc:`../configuration` for
Fusion engine configuration.
