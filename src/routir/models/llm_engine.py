import asyncio
import json
import math
import re
from functools import partial
from typing import Any, Dict, List, Optional

from ..utils import logger
from .abstract import Engine


# -- LLM backend helpers --

def _init_openai_backend(config):
    import os

    import openai

    client = openai.AsyncOpenAI(
        api_key=config.get("api_key", os.getenv("OPENAI_API_KEY", "noneset")),
        base_url=config["server_endpoint"],
    )
    model = config["model_name_or_path"]
    return "openai", {"client": client, "model": model}


def _init_vllm_backend(config):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    model_name = config["model_name_or_path"]
    ngpus = config.get("ngpus", torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_name", model_name), use_fast=True
    )

    llm = LLM(
        model=model_name,
        tensor_parallel_size=int(ngpus),
        trust_remote_code=True,
        max_model_len=config.get("max_model_len", 8192),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
        dtype=config.get("dtype", "bfloat16"),
        enable_prefix_caching=config.get("enable_prefix_caching", True),
        **config.get("vllm_init_kwargs", {}),
    )

    return "vllm", {"llm": llm, "tokenizer": tokenizer}


# -- Score helpers --

def _extract_score_from_logprobs(resp, pos_token, neg_token):
    """Extract a relevance score from a single OpenAI response's logprobs."""
    top_logprobs = resp.choices[0].logprobs.content[0].top_logprobs
    logprob_map = {entry.token.strip().lower(): entry.logprob for entry in top_logprobs}
    pos_lp = logprob_map.get(pos_token.lower(), -10.0)
    neg_lp = logprob_map.get(neg_token.lower(), -10.0)
    pos_p = math.exp(pos_lp)
    neg_p = math.exp(neg_lp)
    return pos_p / (pos_p + neg_p)


async def _openai_score(backend, prompts, pos_token, neg_token, **request_kwargs):
    """Score via OpenAI-compatible API using logprobs on constrained generation."""
    client = backend["client"]
    model = backend["model"]
    total = len(prompts)

    kwargs = {
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        **request_kwargs,
    }

    # Submit all requests, track by index to preserve ordering
    async def _score_one(idx, prompt):
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return idx, resp

    tasks = [_score_one(i, p) for i, p in enumerate(prompts)]
    scores = [None] * total
    completed = 0
    logger.info(f"Scoring 0/{total} pairs...")

    for coro in asyncio.as_completed(tasks):
        idx, resp = await coro
        try:
            scores[idx] = _extract_score_from_logprobs(resp, pos_token, neg_token)
        except Exception as e:
            logger.warning(f"Failed to extract score from logprobs: {e}, defaulting to 0.5")
            scores[idx] = 0.5
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            logger.info(f"Scoring {completed}/{total} pairs...")

    return scores


def _vllm_score(backend, prompts, pos_token_id, neg_token_id):
    """Score via local vLLM using constrained token generation."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[pos_token_id, neg_token_id],
    )

    total = len(prompts)
    logger.info(f"Scoring {total} pairs with vLLM (single batch)...")
    outputs = backend["llm"].generate(prompts, sampling_params, use_tqdm=False)
    logger.info(f"Scoring {total}/{total} pairs done")

    scores = []
    for output in outputs:
        try:
            final_logprobs = output.outputs[0].logprobs[-1]
            pos_entry = final_logprobs.get(pos_token_id, None)
            neg_entry = final_logprobs.get(neg_token_id, None)
            pos_lp = pos_entry.logprob if pos_entry else -10.0
            neg_lp = neg_entry.logprob if neg_entry else -10.0
            pos_p = math.exp(pos_lp)
            neg_p = math.exp(neg_lp)
            scores.append(pos_p / (pos_p + neg_p))
        except Exception as e:
            logger.warning(f"Failed to compute score from vLLM output: {e}, defaulting to 0.5")
            scores.append(0.5)

    return scores


# -- Generate helpers --

async def _openai_generate(backend, prompts, json_mode=False, **request_kwargs):
    """Generate text via OpenAI-compatible API."""
    client = backend["client"]
    model = backend["model"]

    kwargs = {
        "temperature": request_kwargs.pop("temperature", 0.0),
        "max_tokens": request_kwargs.pop("max_tokens", 2048),
        **request_kwargs,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    tasks = [
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": p}],
            **kwargs,
        )
        for p in prompts
    ]
    responses = await asyncio.gather(*tasks)
    return [r.choices[0].message.content.strip() for r in responses]


def _vllm_generate(backend, prompts, **sampling_kwargs):
    """Generate text via local vLLM."""
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=sampling_kwargs.pop("temperature", 0.0),
        max_tokens=sampling_kwargs.pop("max_tokens", 2048),
        **sampling_kwargs,
    )
    outputs = backend["llm"].generate(prompts, params, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


# -- Prompt formatting --

DEFAULT_SCORE_PROMPT = (
    "Given the following query and document, determine if the document is relevant to the query.\n"
    "Answer with only \"{pos_token}\" or \"{neg_token}\".\n\n"
    "Query: {query}\n\n"
    "Document: {document}\n\n"
    "Relevance({pos_token}/{neg_token}): "
)

DEFAULT_DECOMPOSE_PROMPT = (
    "Input: {query}\n\n Generate 10 google-like queries based on the input."
)

DECOMPOSE_FORMAT_INSTRUCTION = (
    '\nReturn the sub-queries as a JSON array of strings, e.g. ["sub-query 1", "sub-query 2"]. '
    "Output only the JSON array, no other text."
)


def _format_score_prompt(template, query, document, pos_token, neg_token):
    return template.format(query=query, document=document, pos_token=pos_token, neg_token=neg_token)


def _format_decompose_prompt(template, query, format_instruction):
    return template.format(query=query) + format_instruction


def _parse_decompose_response(text, limit=None):
    """Extract a list of sub-queries from an LLM response. Try JSON first, fall back to line parsing."""
    # Try JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            result = [str(q).strip() for q in parsed if str(q).strip()]
            return result[:limit] if limit else result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = [str(q).strip() for q in parsed if str(q).strip()]
                return result[:limit] if limit else result
        except json.JSONDecodeError:
            pass

    # Fall back to line-based parsing
    lines = [line.strip().lstrip("0123456789.-) ") for line in text.strip().split("\n")]
    result = [line for line in lines if line]
    return result[:limit] if limit else result


class LLMEngine(Engine):
    """
    Built-in engine that uses an LLM for pointwise reranking (score_batch)
    and query decomposition (decompose_query_batch).

    Supports two backends:
    - **OpenAI-compatible API relay**: set ``server_endpoint`` in config to relay
      requests to an external LLM server (e.g. vLLM serving, OpenAI, etc.).
    - **Local vLLM**: omit ``server_endpoint`` to spin up a local vLLM instance.
      Requires ``pip install "routir[vllm]"``.

    Config keys:
        model_name_or_path (str, required): Model name or HF path.
        server_endpoint (str, optional): OpenAI-compatible base URL. If omitted,
            a local vLLM instance is started.

        # Prompt configuration
        score_prompt (str): Prompt template for scoring. Must contain {query}
            and {document} placeholders. May contain {pos_token} and {neg_token}.
        decompose_prompt (str): Prompt template for decomposition. Must contain
            {query} placeholder.
        pos_token (str): Positive relevance token (default: "Yes").
        neg_token (str): Negative relevance token (default: "No").
        allow_custom_prompts (bool): If True, callers can override prompts at
            runtime via kwargs (default: False).

        # vLLM-specific (ignored when using server_endpoint)
        ngpus (int): Number of GPUs for tensor parallelism.
        max_model_len (int): Max model context length (default: 8192).
        gpu_memory_utilization (float): GPU memory fraction (default: 0.85).
        dtype (str): Model dtype (default: "bfloat16").
        enable_prefix_caching (bool): Enable vLLM prefix caching (default: True).
        vllm_init_kwargs (dict): Extra kwargs passed to vLLM LLM().

        # OpenAI-specific (ignored when using local vLLM)
        api_key (str): API key. Falls back to OPENAI_API_KEY env var.
        request_kwargs (dict): Extra kwargs for chat completion requests.
    """

    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        if "model_name_or_path" not in self.config:
            raise ValueError("LLMEngine requires 'model_name_or_path' in config")

        # Prompt configuration
        self.score_prompt = self.config.get("score_prompt", DEFAULT_SCORE_PROMPT)
        self.decompose_prompt = self.config.get("decompose_prompt", DEFAULT_DECOMPOSE_PROMPT)
        self.pos_token = self.config.get("pos_token", "Yes")
        self.neg_token = self.config.get("neg_token", "No")
        self.allow_custom_prompts = self.config.get("allow_custom_prompts", False)
        self.request_kwargs = dict(self.config.get("request_kwargs", {}))

        # Initialize backend
        if "server_endpoint" in self.config:
            self.backend_type, self.backend = _init_openai_backend(self.config)
            self.pos_token_id = None
            self.neg_token_id = None
            # For OpenAI relay, load a tokenizer for truncation
            from transformers import AutoConfig, AutoTokenizer
            model_name = self.config.get("tokenizer_name", self.config["model_name_or_path"])
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            try:
                model_cfg = AutoConfig.from_pretrained(model_name)
                self._model_max_len = getattr(model_cfg, "max_position_embeddings", None) or getattr(model_cfg, "n_positions", 8192)
            except Exception:
                self._model_max_len = 8192
                logger.warning(f"Could not determine model max length for '{model_name}', defaulting to {self._model_max_len}")
        else:
            self.backend_type, self.backend = _init_vllm_backend(self.config)
            self._tokenizer = self.backend["tokenizer"]
            self.pos_token_id = self._tokenizer(self.pos_token, add_special_tokens=False).input_ids[0]
            self.neg_token_id = self._tokenizer(self.neg_token, add_special_tokens=False).input_ids[0]
            # vLLM already resolved max_model_len; read it back
            self._model_max_len = self.backend["llm"].llm_engine.model_config.max_model_len

        # Resolve max_doc_tokens: leave 512 tokens for prompt/query/generation overhead
        config_max_doc = self.config.get("max_doc_tokens", None)
        effective_limit = self._model_max_len - 512
        if config_max_doc is not None:
            if config_max_doc > effective_limit:
                logger.warning(
                    f"Configured max_doc_tokens={config_max_doc} exceeds model capacity "
                    f"({self._model_max_len} - 512 overhead = {effective_limit}). "
                    f"Clamping to {effective_limit}."
                )
                config_max_doc = effective_limit
            self.max_doc_tokens = config_max_doc
        else:
            self.max_doc_tokens = effective_limit

        logger.info(
            f"LLMEngine '{self.name}' initialized with {self.backend_type} backend, "
            f"model={self.config['model_name_or_path']}, "
            f"model_max_len={self._model_max_len}, max_doc_tokens={self.max_doc_tokens}, "
            f"tokens=({self.pos_token}/{self.neg_token}), "
            f"allow_custom_prompts={self.allow_custom_prompts}"
        )

    def _truncate_doc(self, text):
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > self.max_doc_tokens:
            logger.warning(
                f"Document too long ({len(token_ids)} tokens), truncating to {self.max_doc_tokens} tokens"
            )
            return self._tokenizer.decode(token_ids[:self.max_doc_tokens])
        return text

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        if candidate_length is None:
            candidate_length = [len(passages)]
        if len(candidate_length) != len(queries):
            raise ValueError(f"len(candidate_length)={len(candidate_length)} != len(queries)={len(queries)}")
        if sum(candidate_length) != len(passages):
            raise ValueError(f"sum(candidate_length)={sum(candidate_length)} != len(passages)={len(passages)}")

        # Resolve prompt template
        prompt_template = self.score_prompt
        if self.allow_custom_prompts and "prompt" in kwargs:
            prompt_template = kwargs["prompt"]

        # Build prompts for all pairs (truncate documents to fit model context)
        expanded_queries = [q for q, l in zip(queries, candidate_length) for _ in range(l)]
        prompts = [
            _format_score_prompt(prompt_template, q, self._truncate_doc(d), self.pos_token, self.neg_token)
            for q, d in zip(expanded_queries, passages)
        ]

        # Score
        if self.backend_type == "openai":
            all_scores = await _openai_score(
                self.backend, prompts, self.pos_token, self.neg_token, **self.request_kwargs
            )
        else:
            # TODO: vLLM sync generate blocks the event loop. Consider wrapping with
            # asyncio.to_thread() once vLLM's async engine is stable, or if this
            # becomes a bottleneck for concurrent request handling.
            loop = asyncio.get_event_loop()
            all_scores = await loop.run_in_executor(
                None, _vllm_score, self.backend, prompts, self.pos_token_id, self.neg_token_id
            )

        # Regroup by query
        start = 0
        result = []
        for l in candidate_length:
            result.append(all_scores[start:start + l])
            start += l
        return result

    async def decompose_query_batch(self, queries, limit=None, **kwargs):
        if limit is None:
            limit = [None] * len(queries)
        if len(limit) != len(queries):
            raise ValueError(f"len(limit)={len(limit)} != len(queries)={len(queries)}")

        # Resolve prompt template
        prompt_template = self.decompose_prompt
        if self.allow_custom_prompts and "prompt" in kwargs:
            prompt_template = kwargs["prompt"]

        prompts = [
            _format_decompose_prompt(prompt_template, q, DECOMPOSE_FORMAT_INSTRUCTION)
            for q in queries
        ]

        # Generate
        if self.backend_type == "openai":
            responses = await _openai_generate(
                self.backend, prompts, json_mode=True, **self.request_kwargs
            )
        else:
            # TODO: vLLM sync generate blocks the event loop (same as score_batch).
            loop = asyncio.get_event_loop()
            responses = await loop.run_in_executor(
                None, _vllm_generate, self.backend, prompts
            )

        logger.info(f"From prompt `{prompts}` --> {responses}")

        return [_parse_decompose_response(resp, lim) for resp, lim in zip(responses, limit)]
