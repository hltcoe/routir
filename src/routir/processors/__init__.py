"""
Request processors for handling search, scoring, and content retrieval.

Provides both async and batch processing implementations with caching support.

.. note::
   ``ContentProcessor``, ``IRDSProcessor``, and ``RelayContentProcessor`` have
   moved to ``routir.collections``.  They remain importable from this package
   for one release, emitting ``DeprecationWarning``.  Update imports to
   ``routir.collections.processor`` / ``routir.collections.relay``.
"""

from .abstract import BatchProcessor, LRUCache, Processor
from .query_processors import AsyncQueryProcessor, BatchDecomposeQueryProcessor, BatchQueryProcessor
from .registry import ProcessorRegistry, ServiceNotFound, auto_register
from .score_processors import AsyncPairwiseScoreProcessor, BatchPairwiseScoreProcessor


# PEP 562: lazy deprecation of moved symbols.  Importing the names below from
# routir.processors emits a DeprecationWarning and forwards to the new path.
_MOVED = {
    "ContentProcessor":      ("routir.collections.processor", "ContentProcessor"),
    "IRDSProcessor":         ("routir.collections.processor", "IRDSProcessor"),
    "RelayContentProcessor": ("routir.collections.relay",     "RelayContentProcessor"),
}


def __getattr__(name):
    if name in _MOVED:
        import importlib
        import warnings

        target_module, target_name = _MOVED[name]
        warnings.warn(
            f"`{name}` has moved to `{target_module}`. "
            f"Importing it from `routir.processors` is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(target_module), target_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
