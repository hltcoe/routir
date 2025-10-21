import asyncio
import time
import uuid
from typing import Any, Dict, List

from ..utils import FactoryEnabled, logger
from .cache import Cache, LRUCache, RedisCache


class Processor(FactoryEnabled):
    """
    Base class for request processors with caching support.

    Handles request submission with optional caching via LRU or Redis.

    Attributes:
        cache: Cache instance (LRU or Redis)
        cache_key: Function to generate cache keys from requests
    """

    def __init__(self, cache_size=1024, cache_ttl=600, cache_key=None, redis_url: str = None, redis_kwargs: Dict[str, Any] = {}):
        """
        Initialize processor with caching.

        Args:
            cache_size: Maximum cache entries (-1 to disable)
            cache_ttl: Cache entry time-to-live in seconds
            cache_key: Function to generate cache keys from requests
            redis_url: Optional Redis URL for distributed caching
            redis_kwargs: Additional Redis configuration
        """
        self.cache: Cache = None
        if cache_size > 0 and redis_url is None:
            self.cache = LRUCache(cache_size, cache_ttl)
        elif cache_size > 0 and redis_url is not None:
            self.cache = RedisCache(cache_size, cache_ttl, redis_url, **redis_kwargs)
        self.cache_key = cache_key
        if self.cache_key is None:
            self.cache_key = lambda x: (x.get("service", "default"), x["query"], x.get("limit", "none"), x.get("subset", "none"))

    async def start(self):
        """
        Initialize the processor (called before serving requests).

        Heavy initialization tasks can be performed here.
        """
        pass

    async def submit(self, item: Any) -> Dict[str, Any]:
        """
        Submit a request for processing with caching.

        Args:
            item: Request data

        Returns:
            Response dict with 'cached' field indicating cache hit/miss
        """
        # check cache
        if self.cache is not None:
            cache_key = self.cache_key(item)
            cached = await self.cache.get(cache_key)

            if cached is not None:
                return cached

        result = await self._submit(item)

        # cache the result
        if self.cache is not None and "error" not in result:
            # eventually consistent; no need to wait for cache write before returning
            asyncio.create_task(self.cache.put(cache_key, {**result, "cached": True}))

        return {**result, "cached": False}

    async def _submit(self, item: Any) -> Dict[str, Any]:
        """Process a single request (to be implemented by subclasses)."""
        raise NotImplementedError


class BatchProcessor(Processor):
    def __init__(self, batch_size=32, max_wait_time=0.1, cache_size=1024, cache_ttl=600, cache_key=None, **kwargs):
        """
        Simple dynamic batch processor for batching similar requests.

        Args:
            batch_size: Maximum number of items in a batch
            max_wait_time: Maximum time to wait before processing a partial batch
        """
        super().__init__(cache_size, cache_ttl, cache_key)

        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        # All these will be initialized in start()
        self.queue = None
        self.results: Dict[str, Any] = None
        self.result_events: Dict[str, asyncio.Event] = None
        self.worker_task = None
        self._started = False

    async def start(self):
        """Initialize and start the batch processor."""
        if self._started:
            return

        # Initialize all attributes with the current event loop
        self.queue = asyncio.Queue()
        self.results = {}
        self.result_events = {}

        # Start the worker task
        self.worker_task = asyncio.create_task(self._worker())
        self._started = True
        logger.info("Batch processor started")

    async def _submit(self, item):
        """Submit an item for processing and wait for the result."""
        if not self._started:
            await self.start()

        # Create a unique ID for this request
        item_id = str(uuid.uuid4())

        # Create an event for this request
        event = asyncio.Event()
        self.result_events[item_id] = event

        # Add the item to the queue
        await self.queue.put((item_id, item))
        logger.debug(f"Item {item_id} added to queue")

        # Wait for the result
        await event.wait()

        # Get and return the result
        result = self.results.pop(item_id)
        del self.result_events[item_id]

        return result

    async def _worker(self):
        """Worker that collects items into batches and processes them."""
        logger.info("Worker started")

        while True:
            try:
                batch = []
                batch_ids = []

                # Get the first item (blocks until an item is available)
                try:
                    first_id, first_item = await self.queue.get()
                    batch.append(first_item)
                    batch_ids.append(first_id)
                    logger.debug(f"First item {first_id} received")
                except Exception as e:
                    logger.error(f"Error getting first item: {e}")
                    await asyncio.sleep(0.1)
                    continue

                # Try to fill the batch with more items
                batch_start_time = time.time()
                while len(batch) < self.batch_size and time.time() - batch_start_time < self.max_wait_time:
                    try:
                        # Calculate remaining time
                        remaining_time = max(0.01, self.max_wait_time - (time.time() - batch_start_time))

                        # Try to get another item with timeout
                        item_id, item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                        batch.append(item)
                        batch_ids.append(item_id)
                        logger.debug(f"Additional item {item_id} added to batch")
                    except asyncio.TimeoutError:
                        # No more items available within timeout
                        logger.debug("Timeout waiting for more items")
                        break
                    except Exception as e:
                        logger.error(f"Error collecting batch: {e}")
                        break

                # Process the batch
                batch_size = len(batch)
                logger.info(f"Processing batch of size {batch_size}, {self.queue.qsize()} pending")

                # TODO: should try to dedup the batch based on queries etc

                try:
                    # This is where you'd do your actual batch processing
                    # (e.g., model inference, database operations, etc.)
                    batch_results = await self._process_batch(batch)

                    # Distribute results to waiting clients
                    for i, item_id in enumerate(batch_ids):
                        if i < len(batch_results):
                            self.results[item_id] = batch_results[i]
                        else:
                            # Handle case of mismatched results
                            self.results[item_id] = {"error": "Processing error: missing result"}

                        # Signal that the result is ready
                        if item_id in self.result_events:
                            self.result_events[item_id].set()
                except Exception as e:
                    import traceback

                    logger.error(f"Error processing batch: {traceback.format_exc()}")
                    # Return error to all waiting requests
                    for item_id in batch_ids:
                        self.results[item_id] = {"error": f"Processing error: {str(e)}"}
                        if item_id in self.result_events:
                            self.result_events[item_id].set()

                # Mark tasks as done in the queue
                for _ in range(batch_size):
                    self.queue.task_done()

            except Exception as e:
                logger.error(f"Unexpected error in worker: {e}")
                # Small delay to prevent CPU spinning in case of persistent errors
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        raise NotImplementedError
