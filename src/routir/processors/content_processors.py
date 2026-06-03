"""Deprecated module path. Re-exports from routir.collections.processor."""
import warnings

from ..collections.processor import ContentProcessor, IRDSProcessor  # noqa: F401


warnings.warn(
    "routir.processors.content_processors has moved to routir.collections.processor. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContentProcessor", "IRDSProcessor"]
