"""Trivial bytes-modality engine for PR4 tests.

Imported by RoutIR via ``file_imports`` so the class registers itself on the
:class:`~routir.utils.FactoryEnabled` registry under the name
``TrivialBytesScoreEngine``.  Returns a fixed list of zero scores; what matters
for the modality tests is that its class attribute ``accepts_view_kind`` is
``"bytes"`` so the score slot is registered with ``view_kind="bytes"``.
"""

from routir.models.abstract import Engine


class TrivialBytesScoreEngine(Engine):
    """Bytes-modality score engine. Returns zeros."""

    accepts_view_kind = "bytes"

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        if candidate_length is None:
            candidate_length = [len(passages)]
        out, i = [], 0
        for n in candidate_length:
            out.append([0.0] * n)
            i += n
        return out
