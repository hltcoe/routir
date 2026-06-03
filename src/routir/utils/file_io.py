"""Deprecated module path. Re-exports from routir.collections.indexing.offset_file."""
import warnings

from ..collections.indexing.offset_file import (  # noqa: F401
    MSMARCOSegOffset,
    OffsetFile,
    RandomAccessReader,
)


warnings.warn(
    "routir.utils.file_io has moved to routir.collections.indexing.offset_file. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["RandomAccessReader", "OffsetFile", "MSMARCOSegOffset"]
