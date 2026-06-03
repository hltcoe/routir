"""
Document collections: identity-to-bytes resolution and on-disk layout.

This subpackage owns everything about *where document content lives* and how
to fetch it by id.  Runtime/online concerns (batching, caching, queueing)
live in :mod:`routir.processors`.

Direction of the dependency is one-way: ``collections`` may import the base
``Processor`` class from ``processors.abstract``; ``processors`` does not
import from ``collections``.  Don't reverse this — the split exists precisely
so storage backends can grow without crowding online-request machinery.

Inside this subpackage, always use the qualified import path
``from routir.collections...`` rather than bare ``from collections import ...``
to avoid shadowing the Python stdlib ``collections`` module.
"""

from .processor import ContentProcessor, IRDSProcessor
from .relay import RelayContentProcessor


__all__ = ["ContentProcessor", "IRDSProcessor", "RelayContentProcessor"]
