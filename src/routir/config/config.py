"""Pydantic models for service configuration."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    """
    Configuration for a search/ranking service.

    Attributes:
        name: Service identifier
        engine: Engine class to use; can be defined in external scripts or extensions
                (e.g., 'PLAIDX', 'LSR', 'Qwen3')
        config: Engine-specific configuration parameters
        processor: Processor class name for handling requests
        cache: Cache size (-1 to disable)
        batch_size: Number of requests to batch together
        cache_ttl: Time-to-live for cache entries in seconds
        max_wait_time: Maximum time to wait for batching in seconds
        cache_key_fields: Fields to use for cache key generation
        cache_redis_url: Redis URL for distributed caching
        cache_redis_kwargs: Additional Redis configuration parameters
        scoring_disabled: Whether to disable scoring functionality
    """

    name: str
    engine: str
    # collection: str # mostly for book keeping purpose and allow service name to be cleaner
    config: Dict[str, Any]
    processor: str = "BatchQueryProcessor"
    cache: int = -1
    batch_size: int = 32
    cache_ttl: int = 600
    max_wait_time: float = 0.05
    cache_key_fields: List[str] = Field(default_factory=lambda: ["query", "limit"])
    cache_redis_url: Optional[str] = None
    cache_redis_kwargs: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    scoring_disabled: bool = False


class ColllectionConfig(BaseModel):
    """
    Configuration for a document collection.

    Attributes:
        name: Collection identifier
        doc_path: Path to the document file (typically JSONL)
        offset_source: Method for generating document offsets
        id_field: JSON field name for document IDs
        content_field: JSON field name(s) for document content
        id_to_lang_mapping: Optional path to language mapping file
        cache_path: Optional path for offset map cache
    """

    name: str
    doc_path: str
    offset_source: Literal["msmarco_seg", "offsetfile"] = "offsetfile"
    id_field: str = "id"
    content_field: Union[str, List[str]] = "text"
    id_to_lang_mapping: Optional[str] = None
    cache_path: Optional[str] = None

    def model_post_init(self, __context):
        """Ensure content_field is always a list."""
        if not isinstance(self.content_field, list):
            self.content_field = [self.content_field]


class Config(BaseModel):
    """
    Main configuration for the routir service.

    Attributes:
        services: List of engine service configurations
        collections: List of document collection configurations
        server_imports: List of remote servers to import services from
        file_imports: List of Python files to import for custom extensions
        dynamic_pipeline: Whether to enable dynamic pipeline creation
    """

    services: List[ServiceConfig] = Field(default_factory=list)
    collections: List[ColllectionConfig] = Field(default_factory=list)
    server_imports: List[str] = Field(default_factory=list)  # not yet implemented
    file_imports: List[str] = Field(default_factory=list)
    dynamic_pipeline: bool = True
