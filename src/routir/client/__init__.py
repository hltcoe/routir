"""routir.client - gRPC-default, REST-fallback client for a RoutIR endpoint.

Layering rule: this subpackage depends on nothing in routir.models,
routir.processors, or routir.config. Only stdlib, aiohttp, and routir.utils
(logger only).
"""

from .client import AsyncClient
from .sync import Client
from .transport import RoutirClientError


__all__ = ["AsyncClient", "Client", "RoutirClientError"]
