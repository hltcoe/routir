from typing import List, Tuple, Union

from tqdm.auto import tqdm

from ..utils import logger
from .abstract import Reranker


try:
    import torch
    from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer, T5ForConditionalGeneration
except ImportError:
    logger.warning("Failed to import torch and transformers for T5 Rerankers")

DecodedOutput = Union["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]

# copied from PyGaggle repo
# https://github.com/castorini/pygaggle/blob/master/pygaggle/rerank/transformer.py
prediction_tokens = {
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    # 'unicamp-dl/mt5-13b-mmarco-100k':            ['▁false', '▁true'], # bug
    "unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
}


def _get_prediction_tokens(
    pretrained_model_name_or_path: str, tokenizer: "PreTrainedTokenizer", token_false=None, token_true=None
):
    if not (token_false and token_true):
        if pretrained_model_name_or_path in prediction_tokens:
            token_false, token_true = prediction_tokens[pretrained_model_name_or_path]
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id
        else:
            raise Exception(
                f"We don't know the indexes for the non-relevant/relevant tokens for\
                    the checkpoint {pretrained_model_name_or_path} and you did not provide any."
            )
    else:
        token_false_id = tokenizer.get_vocab()[token_false]
        token_true_id = tokenizer.get_vocab()[token_true]
        return token_false_id, token_true_id


class MT5Reranker(Reranker):
    """
    mT5-based reranker for passage scoring.

    Uses sequence-to-sequence T5 models fine-tuned for relevance scoring.

    Attributes:
        prompt: Template for formatting query-document pairs
        model: T5 model instance
        tokenizer: T5 tokenizer
        q_max_length: Maximum query length
        d_max_length: Maximum document length
        batch_size: Batch size for inference
        token_false_id: Token ID for "false" prediction
        token_true_id: Token ID for "true" prediction
    """

    prompt = "Query: {query} Document: {document} Relevant:"

    def __init__(self, name=None, config=None, **kwargs):
        """
        Initialize MT5 reranker.

        Args:
            name: Reranker name
            config: Configuration with model_name_or_path, max_lengths, etc.
            **kwargs: Additional configuration
        """
        super().__init__(name, config, **kwargs)

        assert "model_name_or_path" in self.config
        self.q_max_length = int(self.config.get("q_max_length", 180))
        self.d_max_length = int(self.config.get("d_max_length", 512))
        self.batch_size = int(self.config.get("batch_size", 32))

        self.model = T5ForConditionalGeneration.from_pretrained(self.config["model_name_or_path"])
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.config["model_name_or_path"], use_fast=True)
        self.model.eval()

        if self.config.get("use_gpu", True):
            self.model = self.model.bfloat16().cuda()

        self.token_false_id, self.token_true_id = _get_prediction_tokens(self.config["model_name_or_path"], self.tokenizer)

    def tokenize_pairs(self, batch: List[Tuple[str, str]]) -> "BatchEncoding":
        return self.tokenizer(
            [self.prompt.format(query=q, document=d) for q, d in batch],
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=self.q_max_length + self.d_max_length + 9,
        )

    # copied and modified from PyGaggle repo
    # https://github.com/castorini/pygaggle/blob/master/pygaggle/model/decode.py
    def forward(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor" = None,
    ) -> DecodedOutput:
        encoder_outputs = self.model.get_encoder()(input_ids, attention_mask=attention_mask)

        with torch.no_grad():
            decode_ids = torch.full(
                (input_ids.size(0), 1), self.model.config.decoder_start_token_id, dtype=torch.long, device=input_ids.device
            )

            model_inputs = self.model.prepare_inputs_for_generation(
                decode_ids, encoder_outputs=encoder_outputs, past_key_values=None, attention_mask=attention_mask, use_cache=True
            )
            outputs = self.model(**model_inputs)  # (batch_size, cur_len, vocab_size)
            true_false_logits = outputs[0][:, -1, [self.token_false_id, self.token_true_id]]  # (batch_size, vocab_size)

            return torch.nn.functional.log_softmax(true_false_logits, dim=1)[:, 1].contiguous()

    async def score(self, queries, passages, candidate_length=None, with_progress=False, **kwargs):
        if candidate_length is None:
            if len(queries) == len(passages):  # pairwise
                candidate_length = [1] * len(queries)
            if len(passages) % len(queries) == 0:  # assume same number of candidates
                candidate_length = [len(passages) // len(queries)] * len(queries)
            else:
                raise ValueError("mT5 Reranker does not support all pair scoring.")

        assert len(candidate_length) == len(queries)
        assert sum(candidate_length) == len(passages)

        expanded_queries = sum([[query] * l for query, l in zip(queries, candidate_length)], [])
        # qidx = sum([[qid] * l for qid, l in zip(range(len(queries)), candidate_length)], [])

        all_scores = (
            torch.concat(
                [
                    self.forward(
                        **self.tokenize_pairs(
                            list(zip(expanded_queries[i : i + self.batch_size], passages[i : i + self.batch_size]))
                        ).to(self.model.device)
                    ).cpu()
                    for i in tqdm(range(0, len(passages), self.batch_size), disable=not with_progress, dynamic_ncols=True)
                ]
            )
            .flatten()
            .tolist()
        )

        offsets = [0] + [sum(candidate_length[: i + 1]) for i in range(len(candidate_length))]

        return [all_scores[sidx:eidx] for sidx, eidx in zip(offsets[:-1], offsets[1:])]
