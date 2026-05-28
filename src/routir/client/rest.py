import asyncio
from typing import Optional

import aiohttp

from ..utils import logger
from .transport import RoutirClientError, Transport


# Initial backoff and ceiling are tuned for a local cluster: the first retry
# is cheap (0.1s) and the cap (2s) keeps a deep retry chain bounded — at
# retries=3 the worst-case wait is 0.1 + 0.2 + 0.4 = 0.7s of sleep, not the
# server-side 600s timeout.
_BACKOFF_START = 0.1
_BACKOFF_CAP = 2.0


class RestTransport(Transport):
    """REST transport over aiohttp.

    Retries on transport errors and 5xx; raises immediately on 4xx because
    those signal request-shape problems that won't fix themselves.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: float = 600,
        retries: int = 3,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        if self._session is not None:
            return
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=headers,
        )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request(self, method: str, path: str, json: Optional[dict] = None) -> dict:
        if self._session is None:
            raise RoutirClientError(
                f"RestTransport used before start(); call await transport.start() first (path={path})"
            )

        url = f"{self.endpoint}{path}"
        backoff = _BACKOFF_START
        last_exc: Optional[BaseException] = None
        last_status: Optional[int] = None
        last_body: Optional[str] = None

        # ``retries=N`` means N additional attempts on top of the first one;
        # total attempts = retries + 1. This matches how Relay was used.
        total_attempts = self.retries + 1
        for attempt in range(total_attempts):
            try:
                async with self._session.request(method, url, json=json) as resp:
                    if 200 <= resp.status < 300:
                        return await resp.json()
                    if 400 <= resp.status < 500:
                        # Client error — surface immediately. Try to pull a
                        # JSON ``error`` field for a useful message.
                        error_msg = None
                        try:
                            body = await resp.json()
                            if isinstance(body, dict):
                                error_msg = body.get("error")
                        except (aiohttp.ContentTypeError, ValueError):
                            try:
                                error_msg = (await resp.text())[:500]
                            except Exception:
                                error_msg = None
                        msg = f"HTTP {resp.status} from {url}: {error_msg}" if error_msg else f"HTTP {resp.status} from {url}"
                        raise RoutirClientError(msg)
                    # 5xx — retry
                    last_status = resp.status
                    try:
                        last_body = (await resp.text())[:500]
                    except Exception:
                        last_body = None
            except RoutirClientError:
                # 4xx already logged below at raise site; re-raise without retry.
                raise
            except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                last_exc = e

            if attempt < total_attempts - 1:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP)

        # Retry budget exhausted.
        if last_exc is not None:
            logger.exception(
                f"RestTransport: {method} {url} failed after {total_attempts} attempts; last error: {type(last_exc).__name__}: {last_exc}"
            )
            raise RoutirClientError(
                f"{method} {url} failed after {total_attempts} attempts; last error: {type(last_exc).__name__}: {last_exc}"
            ) from last_exc
        logger.error(
            f"RestTransport: {method} {url} failed after {total_attempts} attempts; last HTTP {last_status}: {last_body}"
        )
        raise RoutirClientError(
            f"{method} {url} failed after {total_attempts} attempts; last HTTP {last_status}"
        )

    @staticmethod
    def _normalize_search(result: dict) -> dict:
        # Back-compat for servers that returned ``result`` instead of ``scores``;
        # this shim used to live in Relay._submit_payload.
        if isinstance(result, dict) and "scores" not in result and "result" in result:
            result["scores"] = result.pop("result")
        return result

    async def ping(self) -> dict:
        return await self._request("GET", "/ping")

    async def avail(self) -> dict:
        return await self._request("GET", "/avail")

    async def search(self, payload: dict) -> dict:
        result = await self._request("POST", "/search", json=payload)
        return self._normalize_search(result)

    async def score(self, payload: dict) -> dict:
        return await self._request("POST", "/score", json=payload)

    async def content(self, payload: dict) -> dict:
        return await self._request("POST", "/content", json=payload)

    async def pipeline(self, payload: dict) -> dict:
        return await self._request("POST", "/pipeline", json=payload)
