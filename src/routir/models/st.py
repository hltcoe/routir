import itertools
from pathlib import Path
from typing import Any, Dict, List, Union

from tqdm import tqdm
from trecrun import TRECRun

from ..utils import cumsum, dict_topk, load_singleton, logger
from .abstract import Engine


try:
    import faiss
except ImportError:
    logger.warning("Failed to import Faiss for SentenceTransformerEngine")

try:
    import torch
except ImportError:
    logger.warning("Failed to import torch for SentenceTransformerEngine")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.warning("Failed to import SentenceTransformer for SentenceTransformerEngine")


class SentenceTransformerEmbeddingModel:
    """
    Thin wrapper around sentence_transformers.SentenceTransformer for batch encoding.

    Supports flexible configuration to implement different models.

    Args:
        model_name: HuggingFace model name or local path
        batch_size: Encoding batch size
        trust_remote_code: Pass trust_remote_code=True to SentenceTransformer
        encode_based_on_type: If True, use encode_query or encode_document instead
            of encode (for models that expose separate query/doc entry points)
        input_type: Either "query" or "doc", controls which encode variant is used
            when encode_based_on_type is True
        instruction: Optional format string applied to input text before
            encoding. Use ``{query}`` as placeholder, e.g.
            ``"Instruct: ...\nQuery: {query}"``
        normalize_embeddings: Passed to the encode call. None means omit the arg
            and rely on the model's default behaviour
        prompt_name: Passed as ``prompt_name`` encode kwarg (e.g., "query")
        task: Passed as ``task`` encode kwarg (e.g. "retrieval.query" for jinaai/jina-embeddings-v3)
        device: Torch device string; defaults to CUDA if available
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        trust_remote_code: bool = False,
        encode_based_on_type: bool = False,
        input_type: str = "query",
        instruction: str = None,
        normalize_embeddings: bool = None,
        prompt_name: str = None,
        task: str = None,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)

        if not encode_based_on_type:
            self.model_encode = self.model.encode
        elif input_type == "query":
            self.model_encode = self.model.encode_query
        else:
            self.model_encode = self.model.encode_document

        self.encode_kwargs: Dict[str, Any] = {"batch_size": batch_size, "convert_to_tensor": True}
        if normalize_embeddings is not None:
            self.encode_kwargs["normalize_embeddings"] = normalize_embeddings
        if prompt_name is not None:
            self.encode_kwargs["prompt_name"] = prompt_name
        if task is not None:
            self.encode_kwargs["task"] = task

        if instruction is not None:
            self.input_processor = lambda text: instruction.format(query=text)
        else:
            self.input_processor = lambda text: text

    def encode(self, data):
        ids = []
        encoded = []

        data_iter = iter(data)
        with torch.no_grad():
            for _ in tqdm(
                range(0, len(data), self.batch_size),
                desc="st.encode",
                leave=True,
            ):
                id_batch, text_batch = zip(*list(itertools.islice(data_iter, self.batch_size)))
                assert text_batch
                ids.extend(id_batch)

                preprocessed = [self.input_processor(txt) for txt in text_batch]
                embeddings = self.model_encode(preprocessed, **self.encode_kwargs)
                encoded.append(embeddings.cpu().float())

        stacked_embeddings = torch.vstack(encoded)
        return ids, stacked_embeddings


class SentenceTransformerEngine(Engine):
    """
    Dense retrieval engine backed by sentence_transformers.SentenceTransformer and FAISS.

    Flexible enough to implement models like MultilingualE5Instruct, Gemma300,
    ArcticEmbed, BGEM3, and JinaEmbedV3 purely through config.

    Config keys
    -----------
    index_path : str (required)
        Directory containing ``index.faiss`` and ``index.ids``.
    embedding_model_name : str
        HuggingFace model name or local path.
        Default: ``"sentence-transformers/all-MiniLM-L6-v2"``
    batch_size : int
        Encoding batch size.  Default: 32
    trust_remote_code : bool
        Pass ``trust_remote_code=True`` when loading the model.  Default: false
    encode_based_on_type : bool
        Use ``encode_query`` instead of ``encode`` for queries (for models that
        expose separate entry points, e.g. Gemma300).  Default: false
    instruction : str or null
        Format string applied to each query before encoding.
        Use ``{query}`` as the placeholder, e.g.
        ``"Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: {query}"``
        Default: null (no preprocessing)
    normalize_embeddings : bool or null
        Passed directly to the SentenceTransformer encode call.
        null means use the model's own default.  Default: null
    prompt_name_query : str or null
        ``prompt_name`` encode kwarg for queries (e.g. ``"query"`` for
        ArcticEmbed).  Default: null
    task_query : str or null
        ``task`` encode kwarg for queries (e.g. ``"retrieval.query"`` for
        JinaEmbedV3).  Default: null
    k_scale : int
        Multiplier on top of ``limit`` used when calling FAISS search.
        Default: 20
    id_to_subset_mapping : str or null
        Optional path to a ``.pkl`` file mapping doc IDs to subsets.
    """

    def __init__(self, name: str = "SentenceTransformerEngine", config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        super().__init__(name, config, **kwargs)

        model_name = self.config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")

        self.local_embedding_model = SentenceTransformerEmbeddingModel(
            model_name=model_name,
            batch_size=self.config.get("batch_size", 32),
            trust_remote_code=self.config.get("trust_remote_code", False),
            encode_based_on_type=self.config.get("encode_based_on_type", False),
            input_type="query",
            instruction=self.config.get("instruction", None),
            normalize_embeddings=self.config.get("normalize_embeddings", None),
            prompt_name=self.config.get("prompt_name_query", None),
            task=self.config.get("task_query", None),
        )

        index_dir = Path(self.config["index_path"])
        index_path = index_dir / "index.faiss"
        ids_path = index_dir / "index.ids"

        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))

        logger.info(f"Loading document IDs from: {ids_path}")
        with ids_path.open("r") as f:
            self.doc_ids = [line.strip() for line in f]

        logger.info(f"Index contains {self.index.ntotal} vectors")

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
        self, queries: List[str], limit: Union[int, List[int]] = 20, subsets: List[str] = None, **kwargs
    ) -> List[Dict[str, float]]:
        if isinstance(limit, int):
            limit = [int(limit)] * len(queries)

        if subsets is None:
            subsets = [None] * len(queries)

        _, query_embeddings = self.local_embedding_model.encode(list(enumerate(queries)))
        query_embeddings = query_embeddings.numpy()

        scores, ids = self.index.search(x=query_embeddings, k=int(max(limit) * self.config.get("k_scale", 20)))

        qmap = dict(enumerate(queries))
        run = TRECRun({qid: dict(zip([self.doc_ids[x] for x in ids[qid]], scores[qid])) for qid in qmap})
        results = [run[str(qid)] for qid, _ in enumerate(queries)]

        return [dict_topk(self.filter_subset(r, subset), l) for subset, l, r in zip(subsets, limit, results)]

    async def score_batch(self, queries: List[str], passages: List[str], candidate_length: List[int]) -> List[List[float]]:
        assert len(candidate_length) == len(queries)
        assert sum(candidate_length) == len(passages)
        offsets = cumsum([0] + candidate_length)

        _, query_embeddings = self.local_embedding_model.encode(list(enumerate(queries)))

        with torch.no_grad():
            passage_embeddings = self.local_embedding_model.model_encode(
                passages, **self.local_embedding_model.encode_kwargs
            ).cpu().float()

        return [
            (query_embeddings[i] @ passage_embeddings[bidx:eidx].T).ravel().tolist()
            for i, (bidx, eidx) in enumerate(zip(offsets[:-1], offsets[1:]))
        ]
