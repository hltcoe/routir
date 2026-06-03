"""Module-level holders for pipeline-scoped configuration.

The pipeline result cache and the per-pipeline bytes-content cache cap are
both built once in :func:`routir.config.load.load_config` based on top-level
fields on :class:`~routir.config.config.Config`, then read by
:meth:`SearchPipeline.cached_run`.

Kept deliberately small: each holder is a single mutable global plus a setter
and getter, so test code and gRPC/REST handlers all see the same instance.
"""

from typing import Optional

from ..processors.cache import Cache


_PIPELINE_CACHE: Optional[Cache] = None
_BYTES_CONTENT_CACHE_MAX_BYTES: Optional[int] = None


def set_pipeline_cache(cache: Optional[Cache]) -> None:
    """Install the process-wide pipeline cache (or clear it with ``None``)."""
    global _PIPELINE_CACHE
    _PIPELINE_CACHE = cache


def get_pipeline_cache() -> Optional[Cache]:
    """Return the installed pipeline cache, or ``None`` if disabled."""
    return _PIPELINE_CACHE


def set_bytes_content_cache_max_bytes(n: Optional[int]) -> None:
    """Install the process-wide per-pipeline bytes-content cache cap.

    Sourced from :attr:`~routir.config.Config.bytes_content_cache_max_bytes`
    at startup.  ``None`` disables eviction.
    """
    global _BYTES_CONTENT_CACHE_MAX_BYTES
    _BYTES_CONTENT_CACHE_MAX_BYTES = n


def get_bytes_content_cache_max_bytes() -> Optional[int]:
    """Return the installed per-pipeline bytes-content cache cap, or ``None``."""
    return _BYTES_CONTENT_CACHE_MAX_BYTES
