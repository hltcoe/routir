"""
Search and retrieval engine implementations.

Heavy engines (anything pulling in torch / faiss / colbert / vLLM / bsparse)
are loaded lazily on first attribute access via PEP 562, so a server config
that only uses one backend doesn't pay the import cost — or the warning
spam — of the others.

To add a new built-in engine:
- Cheap, no third-party deps, or needs ``auto_register`` at import time
  (e.g. ``Fusion``, ``RRF``, ``ScoreFusion``, ``Relay``):
  add a top-of-file ``from .my_module import MyEngine``.
- Heavy (drags in a large dep): add one entry to ``_LAZY``.

In both cases ``Engine.load("MyEngine", ...)`` resolves the class: the eager
path puts it in ``Engine.__subclasses__()`` at import time, the lazy path
triggers the module import on first lookup.
"""

from .abstract import Aggregation, Engine, Reranker
from .fusion import Fusion  # auto_register("fuse") for RRF / ScoreFusion runs here
from .relay import Relay  # cheap; used by auto_add_relay_services at startup


# Class name -> submodule that defines it.  One line per heavy engine.
_LAZY = {
    "LSR": ".lsr",
    "PLAIDX": ".plaidx",
    "Qwen3": ".qwen3",
    "Qwen3Reranker": ".qwen3reranker",
    "MT5Reranker": ".mt5",
    "LLMEngine": ".llm_engine",
    "SentenceTransformerEngine": ".st",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_LAZY})
