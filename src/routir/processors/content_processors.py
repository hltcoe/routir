import json
from typing import Any, Dict

from ..config import ColllectionConfig
from ..utils import load_singleton
from ..utils.file_io import MSMARCOSegOffset, OffsetFile, RandomAccessReader
from .abstract import Processor


class ContentProcessor(Processor):
    """
    Processor for retrieving document content by ID.

    Provides fast random access to documents in JSONL files using offset maps.

    Attributes:
        config: Collection configuration
        line_reader: Random access reader for document file
        content_field: Field(s) containing document text
        lang_mapping: Optional mapping of document IDs to language codes
    """

    def __init__(self, collection_config: ColllectionConfig, cache_size=0, cache_ttl=600):
        """
        Initialize content processor.

        Args:
            collection_config: Collection configuration with doc_path, id_field, etc.
            cache_size: Maximum cache entries
            cache_ttl: Cache TTL in seconds
        """
        # always use `id` from the request as the key, this is different from id_field in config
        super().__init__(cache_size, cache_ttl, lambda x: x["id"])

        self.config = collection_config
        self.line_reader = self._load_reader()

        self.content_field = collection_config.content_field

        self.lang_mapping = None
        if collection_config.id_to_lang_mapping is not None:
            self.lang_mapping: Dict[str, str] = load_singleton(collection_config.id_to_lang_mapping)

    def _load_reader(self) -> RandomAccessReader:
        if self.config.offset_source == "offsetfile":
            return OffsetFile(
                self.config.doc_path,
                key=lambda line: json.loads(line)[self.config.id_field],
                offset_fn=self.config.cache_path,
                id_field=self.config.id_field,  # Added for parallel offset map building
            )
        elif self.config.offset_source == "msmarco_seg":
            return MSMARCOSegOffset(self.config.doc_path)

    def __getitem__(self, idx: str):
        doc = json.loads(self.line_reader[idx])
        results = {"text": "\n".join(doc[c] for c in self.content_field)}
        if "title" in doc:
            results["title"] = doc["title"]
        if self.lang_mapping is not None:
            results["language"] = self.lang_mapping.get(idx, "")

        return results

    def __contains__(self, idx: str):
        return idx in self.line_reader

    async def _submit(self, item: Dict[str, Any]) -> Dict[str, str]:
        return (
            self[self.cache_key(item)] if self.cache_key(item) in self else {"error": f"ID {self.cache_key(item)} is not found."}
        )
