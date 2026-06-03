"""Concrete view backends.

PR1: TextJsonlBackend.  PR5a adds LocalPathBackend; PR5b adds TarBackend.
"""

from .abstract import ViewBackend
from .local_path import LocalPathBackend
from .tar import TarBackend
from .text_jsonl import TextJsonlBackend


__all__ = ["ViewBackend", "TextJsonlBackend", "LocalPathBackend", "TarBackend"]
