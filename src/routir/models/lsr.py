from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

from ..utils import load_singleton, logger
from .abstract import Engine


try:
    from bsparse.anserini import Anserini
    from bsparse.models import SpladeModel
except ImportError:
    logger.warning("Failed to import bsparse for LSR")


class LSR(Engine):
    """
    Learned Sparse Retrieval engine using SPLADE model as the query encoder.

    Performs sparse retrieval with learned term weights using Anserini indexes.

    Attributes:
        anserini: Anserini index interface
        model: SPLADE model for query encoding
        subset_mapper: Optional mapping of document IDs to subsets
    """

    def __init__(self, name: str = "LSR", config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        """
        Initialize LSR engine.

        Args:
            name: Engine name
            config: Configuration with index_path, model_name, max_length, etc.
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        index = self.config["index_path"]
        self.anserini = Anserini(index)

        @dataclass
        class Args:
            model_name: str = self.config["model_name"]
            max_length: int = self.config["max_length"]
            batch_size: int = self.config["batch_size"]
            max_terms: int = self.config["max_terms"]
            tokenizer_name: str = None

        config = Args()
        # we use SpladeV3 as the query encoder when MultiLSR is the document encoder
        self.model = SpladeModel(config)

        self.subset_mapper: Dict[str, str] = None
        if "id_to_subset_mapping" in self.config:
            if self.config["id_to_subset_mapping"].endswith(".pkl"):
                self.subset_mapper = load_singleton(self.config["id_to_subset_mapping"])
            else:
                logger.warning(f"Unable to load subset mapping file {self.config['id_to_subset_mapping']}")

    def filter_subset(self, scores: Dict[str, float], only_subset: str = None):
        if only_subset is None or self.subset_mapper is None:
            return scores
        return {doc_id: score for doc_id, score in scores.items() if self.subset_mapper[doc_id] == only_subset}

    async def search_batch(
        self, queries: List[str], limit: Union[int, List[int]] = 20, subsets: List[str] = None, maxp: bool = True
    ) -> List[Dict[str, float]]:
        if isinstance(limit, int):
            limit = [int(limit)] * len(queries)

        if subsets is None:
            subsets = [None] * len(queries)

        # anserini only takes 1 k, send max and remove any extra results.
        results = self.anserini.query_from_raw_text(queries, model=self.model, k=max(limit) * self.config.get("k_scale", 20))
        return [
            dict(sorted(self.filter_subset(scores, subset).items(), key=lambda x: -x[1])[:l])
            for subset, l, scores in zip(subsets, limit, results)
        ]
