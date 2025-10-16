import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

from ..utils import logger


try:
    import redis.asyncio as redis
except ImportError:
    logger.warning("Redis is not installed, can only support in-memory caching.")


class Cache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self, capacity: int = -1, ttl: float = -1):
        """
        Initialize cache.

        Args:
            capacity: Maximum number of items to store (-1 for unlimited)
            ttl: Time-to-live in seconds for cache entries (-1 for no expiry)
        """
        self.capacity = capacity
        self.ttl = ttl

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        pass

    @abstractmethod
    async def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        pass

    @abstractmethod
    async def remove(self, key: str) -> None:
        """
        Remove an item from the cache.

        Args:
            key: The cache key
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all items from the cache."""
        pass

    @abstractmethod
    async def __len__(self) -> int:
        """Return the number of items in the cache."""
        pass

    async def connect(self) -> None:
        """Connect to cache backend (if needed). Override in subclasses."""
        pass

    async def close(self) -> None:
        """Close cache connection (if needed). Override in subclasses."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class LRUCache(Cache):
    """An async LRU cache implementation."""

    def __init__(self, capacity: int = -1, ttl: float = -1):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to store (-1 for unlimited)
            ttl: Time-to-live in seconds for cache entries (-1 for no expiry)
        """
        super().__init__(capacity, ttl)
        self.cache = OrderedDict()
        self.timestamps = {}

    async def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        if key not in self.cache:
            return None

        # Check if the item has expired
        if self.ttl > 0:
            timestamp = self.timestamps[key]
            if time.time() - timestamp > self.ttl:
                # Item has expired
                await self.remove(key)
                return None

        # Move the item to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    async def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        # If the key exists, update it and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            self.timestamps[key] = time.time()
            return

        # Add new item
        self.cache[key] = value
        self.timestamps[key] = time.time()

        # If over capacity, remove the oldest item
        if self.capacity > 0 and len(self.cache) > self.capacity:
            oldest_key, _ = self.cache.popitem(last=False)
            del self.timestamps[oldest_key]

    async def remove(self, key: str) -> None:
        """Remove an item from the cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    async def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()

    async def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self.cache)


class RedisCache(Cache):
    """A Redis-based cache implementation with async support."""

    def __init__(
        self,
        capacity: int = -1,
        ttl: float = -1,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "routircache:",
        **redis_kwargs,
    ):
        """
        Initialize Redis cache.

        Args:
            capacity: Maximum number of items to store (-1 for unlimited, not enforced by Redis)
            ttl: Time-to-live in seconds for cache entries (-1 for no expiry)
            redis_url: Redis connection URL
            key_prefix: Prefix for all cache keys to avoid collisions
            **redis_kwargs: Additional arguments to pass to Redis client
        """
        super().__init__(capacity, ttl)
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_kwargs = redis_kwargs
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Establish connection to Redis."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True, **self.redis_kwargs)

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None

    def _make_key(self, key: str) -> str:
        """Create a prefixed key."""
        # Convert key to string if it's a tuple or other type
        if isinstance(key, (tuple, list)):
            key = json.dumps(key)
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found or expired
        """
        await self.connect()

        redis_key = self._make_key(key)
        value = await self.redis_client.get(redis_key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    async def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.

        Args:
            key: The cache key
            value: The value to cache
        """
        await self.connect()

        redis_key = self._make_key(key)

        # Serialize the value
        try:
            serialized_value = json.dumps(value)
        except (TypeError, ValueError):
            # If value can't be JSON serialized, convert to string
            serialized_value = str(value)

        # Set the value with optional TTL
        if self.ttl and self.ttl > 0:
            await self.redis_client.setex(redis_key, int(self.ttl), serialized_value)
        else:
            await self.redis_client.set(redis_key, serialized_value)

    async def remove(self, key: str) -> None:
        """Remove an item from the cache."""
        await self.connect()

        redis_key = self._make_key(key)
        await self.redis_client.delete(redis_key)

    async def clear(self) -> None:
        """Clear all cache entries with the current prefix."""
        await self.connect()

        # Find all keys with our prefix and delete them
        pattern = f"{self.key_prefix}*"
        cursor = 0

        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                await self.redis_client.delete(*keys)
            if cursor == 0:
                break

    async def __len__(self) -> int:
        """Return the number of items in the cache with the current prefix."""
        await self.connect()

        pattern = f"{self.key_prefix}*"
        count = 0
        cursor = 0

        while True:
            cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break

        return count
