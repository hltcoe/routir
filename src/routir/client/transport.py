import asyncio
from abc import ABC, abstractmethod
from typing import List


class RoutirClientError(RuntimeError):
    """Raised by client transports on unrecoverable HTTP / RPC failures.

    Carries a descriptive message including the offending URL (or endpoint)
    and the last response status / underlying exception. Caller code is
    expected to treat this as a terminal error after the transport has
    already exhausted its retry budget.
    """


class Transport(ABC):
    """Wire-format-agnostic transport interface.

    Single-call methods take/return the same plain dicts that the REST API
    accepts/returns today. ``search_batch``/``score_batch`` are present so a
    transport with a true batch RPC (e.g. gRPC streaming) can override them;
    the default implementation just fans out with ``asyncio.gather``.
    """

    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def ping(self) -> dict:
        ...

    @abstractmethod
    async def avail(self) -> dict:
        ...

    @abstractmethod
    async def search(self, payload: dict) -> dict:
        ...

    @abstractmethod
    async def score(self, payload: dict) -> dict:
        ...

    @abstractmethod
    async def content(self, payload: dict) -> dict:
        ...

    @abstractmethod
    async def pipeline(self, payload: dict) -> dict:
        ...

    async def search_batch(self, payloads: List[dict]) -> List[dict]:
        return await asyncio.gather(*[self.search(p) for p in payloads])

    async def score_batch(self, payloads: List[dict]) -> List[dict]:
        return await asyncio.gather(*[self.score(p) for p in payloads])
