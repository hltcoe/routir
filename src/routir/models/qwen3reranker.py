from typing import List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .abstract import Reranker


class Qwen3Reranker(Reranker):
    def __init__(self, name="Qwen3Reranker", config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        model_name = self.config["model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
            .cuda()
            .eval()
        )

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        self.batch_size = config.get("batch_size", 8)

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def tokenize_pairs(self, batch: List[Tuple[str, str]]):
        formatted_pairs = [self.format_instruction(None, query, doc) for query, doc in batch]
        return self.process_inputs(formatted_pairs)

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

        all_scores = (
            torch.concat(
                [
                    self.compute_logits(
                        self.tokenize_pairs(
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

    # Based on example code from https://huggingface.co/Qwen/Qwen3-Reranker-8B
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp()  # .tolist()
        return scores
