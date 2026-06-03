from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from ..utils import cumsum, dict_topk, load_singleton, logger, pbar
from .abstract import Aggregation, Engine


try:
    import torch
    from colbert import Searcher
    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.modeling.colbert import ColBERT, colbert_score
    from colbert.search.index_storage import IndexScorer
except ImportError:
    logger.warning("Failed to import packages for PLAID-X")

_plaidx_checkpoint_singletons: Dict[Union[str, Path], Tuple["Checkpoint", int]] = {}


def _set_xmod_language(checkpoint: "Checkpoint", lang: str):
    """
    Set the default language code for the model. This is used when the language is not specified in the input.
    Source: https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/xmod/modeling_xmod.py#L687
    """
    assert checkpoint.model.base_model.__class__.__name__ == "XmodModel", (
        f"Only Xmod model can set language, but got {str(checkpoint.model.base_model.__class__)}"
    )
    checkpoint.model.set_default_language(lang)
    logger.info(f"Set Xmod language to {lang}")


def colbert_all_pair_scores(Q: "torch.Tensor", D: "torch.Tensor", Dm: "torch.Tensor" = None):
    import torch

    assert len(Q.shape) == 3
    assert len(D.shape) == 3
    assert D.shape[-1] == Q.shape[-1]
    if Dm is not None:
        assert Dm.shape[1] == D.shape[1]
    dim = Q.shape[-1]

    x = (Q.to(D.dtype).view(-1, dim) @ D.view(-1, dim).T).view(-1, *D.shape[:2])
    if Dm is not None:
        x = torch.mul(x, Dm.squeeze(2).unsqueeze(0))
    return x.max(-1).values.view(*Q.shape[:2], -1).sum(1)


def _create_sliding_windows(passages, tokenizer, window_size, stride) -> Tuple[List[Tuple], List[int]]:
    """Create sliding windows from passages without truncation, preserving CLS and D marker tokens."""
    import torch

    all_windows = []
    nwindow_each_pid = []

    # Encode all passages without truncation
    encoded = tokenizer.encode(passages, add_special_tokens=True)
    pad_token_id = tokenizer.tok.pad_token_id or 0

    for pid, ids in enumerate(encoded):
        # Format: [CLS, D, ...content..., SEP]
        # Extract content (skip CLS, D at start and SEP at end)
        special_prefix = ids[:2]  # [CLS, D]
        content = ids[2:-1] if len(ids) > 2 and ids[-1] == tokenizer.tok.sep_token_id else ids[2:]
        length = len(content)

        nw_count = 0
        for start in range(0, max(1, length), stride):  # Ensure at least one window
            end = min(start + window_size, length)
            window_content = content[start:end]

            # Pad to window_size if needed
            pad = window_size - len(window_content)
            window_content = window_content + [pad_token_id] * pad

            # Create window: [CLS, D, content...]
            w_ids = special_prefix + window_content + [tokenizer.tok.sep_token_id]
            w_mask = [1] * (2 + (end - start + 1)) + [0] * pad

            all_windows.append((torch.tensor(w_ids, dtype=torch.long), torch.tensor(w_mask, dtype=torch.long), pid))
            nw_count += 1
            if end >= length:
                break

        nwindow_each_pid.append(nw_count)

    return all_windows, nwindow_each_pid


def _load_mapping(fn, containing_passage_id=True):
    if containing_passage_id:
        return ["_".join(line.strip().split("\t")[1].split("_")[:-1]) for line in pbar(open(fn))]
    else:
        return [line.strip().split("\t")[1] for line in pbar(open(fn))]


class PLAIDX(Engine):
    """
    PLAID-X engine for late-interaction dense retrieval using ColBERT.

    Provides fast and effective neural retrieval with token-level interactions.

    Attributes:
        colbert_config: ColBERT configuration
        searcher: ColBERT searcher instance
        passage_mapper: Optional mapping for passage-to-document aggregation
        subset_mapper: Optional mapping of document IDs to subsets
        inference_batch_size: Batch size for inference operations
    """

    def __init__(self, name: str = "PLAID-X", config: Union[str, Path, Dict[str, Any]] = None, **kwargs):
        """
        Initialize PLAID-X engine.

        Args:
            name: Engine name
            config: Configuration with index_path, checkpoint, use_gpu, etc.
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        # make sure ninja can load
        IndexScorer.try_load_torch_extensions(False)

        if self.index_path is None:
            logger.warning(f"Do not have index for engine `{name}`, will load it as a reranker only.")
            self.colbert_config = ColBERTConfig()
        else:
            self.colbert_config = ColBERTConfig.load_from_index(self.index_path)

        if "checkpoint" in self.config:
            self.colbert_config.checkpoint = self.config["checkpoint"]

        self.checkpoint = PLAIDX._get_existing_checkpoint_instance(self.colbert_config.checkpoint)

        if self.checkpoint is not None:
            logger.info(f"Reuse model in memory for {self.colbert_config.checkpoint}")
        else:
            ColBERT.try_load_torch_extensions(False)  # make sure it is loaded
            self.checkpoint = Checkpoint(self.colbert_config.checkpoint, colbert_config=self.colbert_config)
            assert isinstance(config, dict)
            if config.get("use_gpu", False):
                self.checkpoint = self.checkpoint.half().to(device=f"cuda:{self.config.get('gpu_assignment', 0)}")
                self.checkpoint.compile()
            if "xmod_language" in config:
                _set_xmod_language(self.checkpoint, config["xmod_language"])
            PLAIDX._cache_checkpoint_instance(self.colbert_config.checkpoint, self.checkpoint)

        self.inference_batch_size = int(config.get("inference_batch_size", 32))

        if self.index_path is not None:
            self.passage_mapper = None
            if "passage_mapping" not in self.config:
                if (Path(self.index_path) / "passage_mapping.tsv").exists():
                    self.config["passage_mapping"] = Path(self.index_path) / "passage_mapping.tsv"
                elif (Path(self.index_path) / "mapping.tsv").exists():
                    self.config["passage_mapping"] = Path(self.index_path) / "mapping.tsv"

            if "passage_mapping" in self.config:
                self.passage_mapper = Aggregation(
                    _load_mapping(self.config["passage_mapping"], self.config.get("mapping_containing_passage_id", True))
                )

            self.subset_mapper: Dict[str, str] = None
            if "id_to_subset_mapping" in self.config:
                if self.config["id_to_subset_mapping"].endswith(".pkl"):
                    self.subset_mapper = load_singleton(self.config["id_to_subset_mapping"])
                else:
                    logger.warning(f"Unable to load subset mapping file {self.config['id_to_subset_mapping']}")

            with Run().context(RunConfig(index_root=self.index_path.parent, gpus=0)):
                logger.info(f"Loading PLAID-X index `{self.index_path}` and checkpoint `{self.colbert_config.checkpoint}`")
                self.searcher = Searcher(index=self.index_path.name, checkpoint=self.checkpoint, load_collection=False)

    def __del__(self):
        if hasattr(self, "searcher"):
            PLAIDX._delete_checkpoint_reference(self.searcher.checkpoint)

    @staticmethod
    def _get_existing_checkpoint_instance(checkpoint_path: str):
        global _plaidx_checkpoint_singletons
        abs_path = Path(checkpoint_path).absolute()
        if abs_path not in _plaidx_checkpoint_singletons:
            return None
        else:
            _plaidx_checkpoint_singletons[abs_path] = (
                _plaidx_checkpoint_singletons[abs_path][0],
                _plaidx_checkpoint_singletons[abs_path][1] + 1,
            )
            return _plaidx_checkpoint_singletons[abs_path][0]

    @staticmethod
    def _cache_checkpoint_instance(checkpoint_path: str, checkpoint: "Checkpoint"):
        global _plaidx_checkpoint_singletons
        abs_path = Path(checkpoint_path).absolute()
        assert abs_path not in _plaidx_checkpoint_singletons
        _plaidx_checkpoint_singletons[abs_path] = (checkpoint, 1)

    @staticmethod
    def _delete_checkpoint_reference(checkpoint: "Checkpoint"):
        global _plaidx_checkpoint_singletons
        for path in _plaidx_checkpoint_singletons:
            if _plaidx_checkpoint_singletons[path][0] == checkpoint:
                _plaidx_checkpoint_singletons[path] = (
                    _plaidx_checkpoint_singletons[path][0],
                    _plaidx_checkpoint_singletons[path][1] - 1,
                )
                if _plaidx_checkpoint_singletons[path][1] == 0:
                    del _plaidx_checkpoint_singletons[path]
                return

    def filter_subset(self, scores: Dict[str, float], only_subset: str = None):
        if only_subset is None or self.subset_mapper is None:
            return scores
        return {doc_id: score for doc_id, score in scores.items() if self.subset_mapper[doc_id] == only_subset}

    async def search_batch(
        self, queries: List[str], limit: Union[int, List[int]] = 1000, subsets: List[str] = None, maxp: bool = True
    ) -> List[Dict[str, float]]:
        if self.index_path is None:
            raise RuntimeError(f"Engine `{self.name}` does not have an index, can only be used as a reranker.")

        if isinstance(limit, int):
            limit = [int(limit * 1.5)] * len(queries)

        if subsets is None:
            subsets = [None] * len(queries)

        Q = self.searcher.encode(queries)
        scores = []

        for query_idx, (k, sub) in enumerate(zip(limit, subsets)):
            # TODO: expose these settings into config
            if k <= 10:
                self.searcher.configure(ncells=1, centroid_score_threshold=0.5, ndocs=256)
            elif k <= 100:
                self.searcher.configure(ncells=2, centroid_score_threshold=0.45, ndocs=1024)
            else:
                self.searcher.configure(ncells=2, centroid_score_threshold=0.4, ndocs=2048)

            pids, q_scores = self.searcher.ranker.rank(self.searcher.config, Q[query_idx : query_idx + 1])
            scores.append(dict(zip(pids, q_scores)))
            # s = dict((pid, score) for pid, _ , score in zip(*self.searcher.dense_search(Q[query_idx:query_idx+1], max(100, k*2))) )

        if not maxp or self.passage_mapper is None:
            return scores

        return [
            dict_topk(self.filter_subset(scores, subset), l)
            for subset, l, scores in zip(subsets, limit, map(self.passage_mapper.maxp, scores))
        ]

    async def score_batch(self, queries: List[str], passages: List[str], candidate_length: List[int]) -> List[List[float]]:
        if len(candidate_length) != len(queries):
            raise ValueError(f"len(candidate_length)={len(candidate_length)} does not match len(queries)={len(queries)}")
        if sum(candidate_length) != len(passages):
            raise ValueError(f"sum(candidate_length)={sum(candidate_length)} does not match len(passages)={len(passages)}")
        offsets = cumsum([0] + candidate_length)

        window_size = self.config.get("sliding_window_size", 180)
        stride = self.config.get("sliding_window_stride", 90)
        batch_size = self.config.get("passage_batch_size", 512)

        # Create windows without truncation using encode
        windows, nwindow_each_pid = _create_sliding_windows(passages, self.checkpoint.doc_tokenizer, window_size, stride)
        window_input_ids = torch.stack([w[0] for w in windows])
        Dm = torch.stack([w[1] for w in windows])
        window_to_pid = [w[2] for w in windows]

        expanded_qids = sum([[i] * l for i, l in enumerate(candidate_length)], [])
        # Map window indices to query indices: window -> passage -> query
        window_to_qid = [expanded_qids[pid] for pid in window_to_pid]

        results = [[-999 for _ in range(cl)] for cl in candidate_length]
        with torch.inference_mode():
            Q = self.checkpoint.queryFromText(queries)

            for i in range(0, window_input_ids.shape[0], batch_size):
                end_idx = min(i + batch_size, window_input_ids.shape[0])
                batch_Q = torch.stack([Q[window_to_qid[wid]] for wid in range(i, end_idx)])
                batch_D, batch_Dm = self.checkpoint.doc(window_input_ids[i:end_idx], Dm[i:end_idx], keep_dims="return_mask")

                for wid, score in zip(range(i, end_idx), colbert_score(batch_Q, batch_D, batch_Dm).cpu().tolist()):
                    qid = window_to_qid[wid]
                    pid = window_to_pid[wid]
                    local_pid = pid - offsets[qid]  # Convert global to local passage index
                    results[qid][local_pid] = max(results[qid][local_pid], score)

            return results
