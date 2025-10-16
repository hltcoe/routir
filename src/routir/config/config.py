from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
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
    name: str
    doc_path: str
    offset_source: Literal["msmarco_seg", "offsetfile"] = "offsetfile"
    id_field: str = "id"
    content_field: Union[str, List[str]] = "text"
    id_to_lang_mapping: Optional[str] = None
    cache_path: Optional[str] = None

    def model_post_init(self, __context):
        if not isinstance(self.content_field, list):
            self.content_field = [self.content_field]


class Config(BaseModel):
    services: List[ServiceConfig] = Field(default_factory=list)
    collections: List[ColllectionConfig] = Field(default_factory=list)
    server_imports: List[str] = Field(default_factory=list)  # not yet implemented
    file_imports: List[str] = Field(default_factory=list)
    dynamic_pipeline: bool = True
