import asyncio
import time
import traceback
import uuid
from typing import Any, Dict, List

from ..utils import FactoryEnabled, logger
from .cache import Cache, LRUCache, RedisCache


class Processor(FactoryEnabled):
    """
    Base class for request processors with optional caching.

    A ``Processor`` sits between the HTTP layer and an engine.  It handles
    cache lookup/store and delegates to :meth:`_submit` for the actual work.

    **When to subclass Processor vs BatchProcessor**

    * Subclass :class:`Processor` when requests should be handled *one at a
      time* — e.g. content lookup by document ID where batching adds no value.
      Override :meth:`_submit` to implement the logic.

    * Subclass :class:`BatchProcessor` when the underlying engine (e.g. a GPU
      model) benefits from processing multiple requests together.  Override
      :meth:`~BatchProcessor._process_batch` instead.

    Both classes share the same caching interface; the cache is checked before
    :meth:`_submit` / :meth:`~BatchProcessor._process_batch` is called.

    Attributes:
        cache (Cache or None): Active cache instance (LRU or Redis), or ``None``
            when caching is disabled.
        cache_key (callable): Function ``(item: dict) -> hashable`` used to
            derive the cache key from a request dict.  Default key includes
            ``service``, ``query``, ``limit``, and ``subset``.
    """

    def __init__(self, cache_size=1024, cache_ttl=600, cache_key=None, redis_url: str = None, redis_kwargs: Dict[str, Any] = {}):
        """
        Initialize the processor with optional caching.

        Args:
            cache_size (int): Maximum number of cached entries.  ``-1`` or
                ``0`` disables caching entirely.  When ``redis_url`` is set,
                this controls the Redis key-count budget (approximate).
            cache_ttl (int): Cache entry time-to-live in seconds (default 600).
            cache_key (callable, optional): ``(item: dict) -> hashable``
                function to derive a cache key from a request dict.  The
                default key is ``(service, query, limit, subset)``.  Override
                when additional request fields affect the result (e.g. pass
                ``cache_key_fields`` via a closure, as :func:`~routir.config.load.load_config`
                does per service).
            redis_url (str, optional): Redis connection URL.  When provided,
                Redis is used instead of the in-memory LRU cache.
            redis_kwargs (dict): Additional keyword arguments forwarded to the
                Redis client.
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
    """
    Processor that accumulates requests into batches before engine inference.

    Requests are queued in an :class:`asyncio.Queue` by :meth:`_submit`.  A
    background worker collects items until either ``batch_size`` is reached or
    ``max_wait_time`` seconds elapse, then calls :meth:`_process_batch` with
    the whole batch.  This amortises GPU/model overhead across concurrent
    requests, improving throughput at the cost of a small latency increase.

    Subclasses must override :meth:`_process_batch`; all other machinery is
    provided here.

    The worker is started lazily on the first request (or explicitly via
    :meth:`start`), and runs for the lifetime of the process.
    """

    def __init__(self, batch_size=32, max_wait_time=0.1, cache_size=1024, cache_ttl=600, cache_key=None, **kwargs):
        """
        Initialize the batch processor.

        Args:
            batch_size (int): Maximum number of requests accumulated into one
                batch before the engine is called (default 32).
            max_wait_time (float): Maximum seconds to wait for the batch to
                fill before processing a partial batch (default 0.1 s).
                Tune this to balance latency vs. GPU utilisation — lower values
                reduce wait time, higher values pack more requests per batch.
            cache_size (int): LRU cache size; ``-1`` disables caching.
            cache_ttl (int): Cache TTL in seconds.
            cache_key (callable, optional): Custom cache-key function; see
                :class:`Processor` for details.
            **kwargs: Forwarded to :class:`Processor.__init__`.
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
                    logger.exception("Error getting first item")
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
                        logger.exception("Error collecting batch")
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
                    logger.exception("Error processing batch")
                    # Return error to all waiting requests
                    for item_id in batch_ids:
                        self.results[item_id] = {"error": f"Processing error: {str(e)}"}
                        if item_id in self.result_events:
                            self.result_events[item_id].set()

                # Mark tasks as done in the queue
                for _ in range(batch_size):
                    self.queue.task_done()

            except Exception as e:
                logger.exception("Unexpected error in worker")
                # Small delay to prevent CPU spinning in case of persistent errors
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a batch of requests and return one result dict per request.

        Override this in subclasses to implement the actual engine call.
        The result list must have the **same length** as ``batch`` and preserve
        order so results can be mapped back to their waiting callers.

        Args:
            batch (list[dict]): List of request dicts accumulated from the
                queue.  Each dict is whatever was passed to :meth:`_submit`.

        Returns:
            list[dict]: One result dict per input request, in the same order.
                Each result dict is returned directly to the caller that
                submitted the corresponding request.
        """
        raise NotImplementedError
