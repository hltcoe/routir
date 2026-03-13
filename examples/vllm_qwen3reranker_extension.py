"""
Qwen3 Reranker wrapper for RoutIR integration
"""

import logging
import math
import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from routir.models.abstract import Engine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["VLLM_SKIP_P2P_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Qwen3Reranker:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-4B",
        context_size: int = 8192,
        fp_options: str = "bfloat16",
        num_gpus: int = 1,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        """
        Qwen3 Reranker is a yes/no classification reranker model that judges document relevance.

        Args:
            model_name_or_path: Model identifier (default: Qwen/Qwen3-Reranker-4B)
            context_size: Maximum context length (default: 8192)
            fp_options: Floating point precision (default: bfloat16)
            num_gpus: Number of GPUs for tensor parallelism (default: 1)
            instruction: Task instruction for the reranker
        """
        self.context_size = context_size - 10
        self.num_gpus = num_gpus
        self.instruction = instruction
        self.model_name_or_path = model_name_or_path

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache token IDs for yes/no
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]

        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=0.8,
            dtype=fp_options,
            enable_prefix_caching=True,
        )

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

        # Suffix for model generation (assistant thinking prefix)
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_instruction(self, instruction: str, query: str, doc: str) -> List[dict]:
        """Format input as chat messages for Qwen3 Reranker."""
        return [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"
            }
        ]

    def _process_inputs(self, pairs: List[Tuple[str, str]]) -> List[TokensPrompt]:
        """Process query-document pairs into tokenized inputs for vLLM."""
        messages = [self._format_instruction(self.instruction, query, doc) for query, doc in pairs]

        # Apply chat template
        token_ids_list = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Truncate and append suffix
        max_length = self.context_size
        token_ids_list = [
            ele[: max_length - len(self.suffix_tokens)] + self.suffix_tokens
            for ele in token_ids_list
        ]

        return [TokensPrompt(prompt_token_ids=ele) for ele in token_ids_list]

    def _compute_scores(self, outputs) -> List[float]:
        """Compute relevance scores from yes/no token probabilities."""
        scores = []

        for output in outputs:
            try:
                final_logprobs = output.outputs[0].logprobs[-1]

                # Extract logprobs for yes/no tokens
                true_logit = final_logprobs.get(self.true_token, None)
                false_logit = final_logprobs.get(self.false_token, None)

                # Handle missing logprobs
                true_logit = true_logit.logprob if true_logit else -10.0
                false_logit = false_logit.logprob if false_logit else -10.0

                # Convert logprobs to probabilities and compute score
                true_score = math.exp(true_logit)
                false_score = math.exp(false_logit)
                score = true_score / (true_score + false_score)
                scores.append(score)

            except Exception as e:
                logger.warning(f"Error computing score: {e}, setting to 0.5")
                scores.append(0.5)

        return scores

    @torch.inference_mode()
    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs.

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores in [0, 1]
        """
        prompts = self._process_inputs(pairs)
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        scores = self._compute_scores(outputs)
        return scores


class Qwen3VllmRerankerEngine(Engine):
    """RoutIR Engine wrapper for Qwen3 Reranker"""

    def __init__(self, name: str = None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)
        self.model = Qwen3Reranker(
            model_name_or_path=config.get("model", "Qwen/Qwen3-Reranker-4B"),
            num_gpus=config.get("ngpus", 1),
            fp_options=config.get("fp_options", "bfloat16"),
            instruction=config.get("instruction", "Given a web search query, retrieve relevant passages that answer the query"),
        )

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        """
        Score a batch of query-document pairs.

        Args:
            queries: List of query strings
            passages: List of passage strings
            candidate_length: List of candidate counts per query

        Returns:
            List of score lists, one per query
        """
        if candidate_length is None:
            candidate_length = [len(passages)]
        assert len(candidate_length) == len(queries)
        assert sum(candidate_length) == len(passages)

        # Expand queries to match passages
        expanded_queries = sum([[queries[i]] * l for i, l in enumerate(candidate_length)], [])
        pairs = list(zip(expanded_queries, passages))

        # Get scores
        all_scores = self.model.score(pairs)

        # Group scores by query
        start = 0
        res = []
        for l in candidate_length:
            res.append(all_scores[start: start + l])
            start = start + l
        return res
