"""
Search and retrieval engine implementations.

This module provides various search engines including dense retrieval (PLAIDX, Qwen3),
sparse retrieval (LSR), rerankers (MT5), and utility engines (Relay, Fusion).
"""

from typing import Any, Dict, List

from .abstract import Aggregation, Engine, Reranker
from .fusion import Fusion
from .lsr import LSR
from .mt5 import MT5Reranker
from .plaidx import PLAIDX
from .qwen3 import Qwen3
from .relay import Relay
