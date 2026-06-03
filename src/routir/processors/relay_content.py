"""Deprecated module path. Re-exports from routir.collections.relay."""
import warnings

from ..collections.relay import RelayContentProcessor  # noqa: F401


warnings.warn(
    "routir.processors.relay_content has moved to routir.collections.relay. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RelayContentProcessor"]
