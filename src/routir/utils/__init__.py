"""
Utility functions and classes for the routir service.

Provides logging, file loading, HTTP requests, and factory pattern support.
"""

import logging
import pickle
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
from tqdm.auto import tqdm

from .lazy_import import _lazy_modules


logging.basicConfig(
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("search-service")


def pbar(it=None, **kwargs):
    """
    Create a progress bar with sensible defaults.

    Args:
        it: Iterable to wrap
        **kwargs: Additional tqdm arguments

    Returns:
        tqdm progress bar
    """
    if "dynamic_ncols" not in kwargs:
        kwargs["dynamic_ncols"] = True

    return tqdm(it, **kwargs)


_file_singleton = {}


def load_singleton(fn, load_fn=None):
    """
    Load and cache a file as a singleton.

    Args:
        fn: File path
        load_fn: Optional custom loading function (default: pickle.load)

    Returns:
        Loaded object (cached for subsequent calls)
    """
    global _file_singleton
    fn = str(Path(fn).absolute())
    if fn not in _file_singleton:
        if load_fn is None:
            with open(fn, "rb") as f:
                _file_singleton[fn] = pickle.load(f)
        else:
            _file_singleton[fn] = load_fn(fn)

    return _file_singleton[fn]


def dict_topk(scores: Dict[Any, float], k: int) -> Dict[Any, float]:
    """
    Get top-k entries from a score dictionary.

    Args:
        scores: Dict mapping items to scores
        k: Number of top items to return

    Returns:
        Dict with top-k items sorted by score
    """
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:k])

async def session_request(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]=None, method="POST"):
    """
    Make an HTTP request using an aiohttp session.

    Args:
        session: aiohttp ClientSession
        url: Request URL
        payload: Request payload (for POST)
        method: HTTP method ("POST" or "GET")

    Returns:
        Response JSON data
    """
    try:
        if method == "POST":
            async with session.post(url, json=payload) as response:
                return await response.json()
        elif method == "GET":
            async with session.get(url) as response:
                return await response.json()
    except Exception as e:
        logger.warning(f"HTTP request to {url} failed: {e}")


def _recursive_subclasses(cls: type):
    """
    Recursively yield all subclasses of a class.

    Args:
        cls: Class to find subclasses of

    Yields:
        Subclass types
    """
    for subcls in cls.__subclasses__():
        yield subcls
        yield from _recursive_subclasses(subcls)


class FactoryEnabled(ABC):
    """
    Mixin that enables factory-style class instantiation by name.

    Allows loading subclasses by their name string.
    """

    @classmethod
    def load(cls, cls_name: str, **kwargs):
        """
        Load a subclass by name.

        Args:
            cls_name: Name of the subclass to instantiate
            **kwargs: Arguments to pass to the subclass constructor

        Returns:
            Instance of the requested subclass

        Raises:
            TypeError: If no matching subclass is found
        """
        for subcls in _recursive_subclasses(cls):
            if subcls.__name__ == cls_name or subcls == cls_name:
                return subcls(**kwargs)

        raise TypeError(f"Unsupported subclass {cls_name} for {cls.__name__}")
