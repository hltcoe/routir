"""Module-level holder for the pipeline-level result cache.

The cache is built once in :func:`routir.config.load.load_config` based on the
top-level ``pipeline_cache*`` fields in :class:`~routir.config.config.Config`,
then accessed by :meth:`SearchPipeline.cached_run`.

Kept deliberately small: the holder is a single mutable global plus a setter
and getter, so test code and gRPC/REST handlers all see the same instance.
"""

from typing import Optional

from ..processors.cache import Cache


_PIPELINE_CACHE: Optional[Cache] = None


def set_pipeline_cache(cache: Optional[Cache]) -> None:
    """Install the process-wide pipeline cache (or clear it with ``None``)."""
    global _PIPELINE_CACHE
    _PIPELINE_CACHE = cache


def get_pipeline_cache() -> Optional[Cache]:
    """Return the installed pipeline cache, or ``None`` if disabled."""
    return _PIPELINE_CACHE
