"""Minimal in-process engine used by the test fixtures.

Imported by RoutIR via ``file_imports`` so the class registers itself on the
:class:`~routir.utils.FactoryEnabled` registry under the name
``TrivialSearchEngine``. Returns the same fixed result for every query, which
is all we need to verify REST/gRPC wire compatibility.
"""

from routir.models.abstract import Engine


class TrivialSearchEngine(Engine):
    """Returns ``{"doc1": 1.0, "doc2": 0.5}`` for every query."""

    async def search_batch(self, queries, limit=20, **kwargs):
        return [{"doc1": 1.0, "doc2": 0.5} for _ in queries]
