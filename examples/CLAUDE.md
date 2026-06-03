# Wrapping Retrieval Models with RoutIR

This guide shows Claude Code agents how to wrap arbitrary retrieval/reranking models with RoutIR's Engine interface.

## The Two-Layer Pattern

All RoutIR extensions follow this pattern:

```python
from routir.models.abstract import Engine

# Layer 1: Core model wrapper (your model's inference logic)
class MyModelWrapper:
    def __init__(self, model_path, **kwargs):
        self.model = load_your_model(model_path)

    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs, return scores"""
        return self.model.predict(pairs)

# Layer 2: RoutIR Engine (connects to RoutIR's batching/caching)
class MyModelEngine(Engine):
    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)
        self.model = MyModelWrapper(
            model_path=config.get("model", "default/model"),
            **config
        )

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        """THE critical method - handles batch scoring for RoutIR"""
        # Implementation below...
```

**Why two layers?** Layer 1 contains your model-specific code. Layer 2 adapts it to RoutIR's async batching interface.

## Implementing score_batch: The Critical Pattern

This is the **most important method**. All rerankers must implement it correctly:

```python
async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
    """
    Score batches of query-document pairs.

    Args:
        queries: List[str] - N queries
        passages: List[str] - Flattened list of ALL passages for ALL queries
        candidate_length: List[int] - How many candidates per query

    Example:
        queries = ["query1", "query2"]
        candidate_length = [2, 3]  # query1 has 2 docs, query2 has 3
        passages = ["doc1", "doc2", "doc3", "doc4", "doc5"]  # flattened

    Returns:
        List[List[float]] - One score list per query
        [[score1, score2], [score3, score4, score5]]
    """
    # Step 1: Handle default case
    if candidate_length is None:
        candidate_length = [len(passages)]

    # Step 2: Validate inputs
    assert len(candidate_length) == len(queries), "Mismatch: queries vs candidate_length"
    assert sum(candidate_length) == len(passages), "Mismatch: candidate_length vs passages"

    # Step 3: Expand queries to match passages
    # [[q1, q1], [q2, q2, q2]] -> [q1, q1, q2, q2, q2]
    expanded_queries = sum([[queries[i]] * length for i, length in enumerate(candidate_length)], [])

    # Step 4: Create (query, passage) pairs
    pairs = list(zip(expanded_queries, passages))

    # Step 5: Score all pairs (sync or async)
    all_scores = self.model.score(pairs)  # Your model's scoring method

    # Step 6: Re-group scores by query
    start = 0
    result = []
    for length in candidate_length:
        result.append(all_scores[start:start + length])
        start += length

    return result
```

**Critical**: Always expand queries first, score all pairs, then regroup by `candidate_length`.

## Declaring the Content Modality (`accepts_view_kind`)

RoutIR collections expose **named views** of each document — e.g. `ocr`
(text), `asr` (text), `keyframe` (bytes), `audio` (bytes).  A reranker
stage in a pipeline picks one with the `@view` DSL suffix
(`kf-rerank@keyframe%50`), and the corresponding view payload is what
arrives in `passages`.

Every engine that implements `score_batch` **must** declare what kind of
payload it expects via the `accepts_view_kind` class attribute on the
`Engine` subclass:

```python
class TextReranker(Engine):
    accepts_view_kind = "text"          # passages: List[str]   (default)

class KeyframeReranker(Engine):
    accepts_view_kind = "bytes"         # passages: List[List[bytes]]
                                        # one inner list per doc;
                                        # length 0 is legal (e.g. audio-only chunks)
```

The attribute is read by `config.load` and `auto_register` at startup;
the value is stored on the registry slot.  `SearchPipeline.verify()`
matches each rerank stage's view kind against the score slot's
`view_kind` and rejects mismatches before any documents are fetched.

There is no `"both"` value — if an engine consumes both modalities,
register it under two service names.

**Bytes engines and REST.**  `/score` over REST is text-only and refuses
bytes engines with HTTP 400.  Reach bytes rerankers via either:

- the pipeline DSL (in-process — the engine receives `List[List[bytes]]`
  straight from the collection's `ContentProcessor`, no wire encoding), or
- gRPC `Score` (uses the `BytesParts` `oneof` in `Passage`).

For multi-blob payloads (multiple keyframes per video, multiple audio
segments per chunk), each inner list contains one or more byte blobs;
single-blob views are length-1 lists.  The list-of-lists shape is universal
so multi-blob and single-blob views work without a separate code path.

## Model Type: Reranker vs Search Engine

### Reranker (Cross-Encoder)
Scores query-document pairs. Implements `score_batch()`:

```python
class RerankerEngine(Engine):
    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        # Pattern shown above
        pass
```

### Search Engine (Dense/Sparse Retrieval)
Searches an index. Implements `search_batch()`:

```python
class SearchEngine(Engine):
    async def search_batch(self, queries, limit=20, **kwargs):
        """
        Args:
            queries: List[str] - Query strings
            limit: int or List[int] - How many results per query

        Returns:
            List[Dict[str, float]] - One dict per query mapping docid -> score
        """
        if isinstance(limit, int):
            limit = [limit] * len(queries)

        results = []
        for query, k in zip(queries, limit):
            scores_dict = self.index.search(query, k=k)  # {docid: score}
            results.append(scores_dict)

        return results
```

## Complete Working Example: LLM Reranker

Here's a complete implementation for an LLM-based reranker using yes/no token probabilities:

```python
import math
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from routir.models.abstract import Engine

class LLMReranker:
    """Core model wrapper - handles tokenization and inference"""

    def __init__(
        self,
        model_name: str = "your-org/reranker-model",
        num_gpus: int = 1,
        context_size: int = 4096,
        batch_size: int = 32,
    ):
        self.context_size = context_size
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache token IDs for yes/no (or true/false, relevant/not-relevant, etc.)
        self.yes_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

    def _format_prompt(self, query: str, passage: str) -> str:
        """Format query-passage pair into model's expected prompt"""
        # Adapt this to your model's prompt format
        return f"Query: {query}\nPassage: {passage}\nRelevant: "

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit context window"""
        tokens = self.tokenizer(text)["input_ids"]
        if len(tokens) > max_length:
            return self.tokenizer.decode(tokens[:max_length])
        return text

    @torch.inference_mode()
    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs"""
        all_scores = []

        # Process in batches
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]

            # Format prompts
            prompts = [
                self._format_prompt(query, self._truncate_text(doc, self.context_size - 100))
                for query, doc in batch_pairs
            ]

            # Tokenize
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.context_size,
                return_tensors="pt",
            ).to(self.model.device)

            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits

            # Extract yes/no probabilities
            batch_scores = []
            for i in range(len(batch_pairs)):
                try:
                    yes_logit = logits[i, self.yes_token_id].item()
                    no_logit = logits[i, self.no_token_id].item()

                    # Convert to probability
                    yes_prob = math.exp(yes_logit)
                    no_prob = math.exp(no_logit)
                    score = yes_prob / (yes_prob + no_prob)

                    batch_scores.append(score)
                except Exception as e:
                    # Fallback to neutral score on error
                    batch_scores.append(0.5)

            all_scores.extend(batch_scores)

        return all_scores


class LLMRerankerEngine(Engine):
    """RoutIR Engine wrapper"""

    def __init__(self, name=None, config=None, **kwargs):
        super().__init__(name, config, **kwargs)

        self.model = LLMReranker(
            model_name=config.get("model", "your-org/reranker-model"),
            num_gpus=config.get("ngpus", 1),
            context_size=config.get("context_size", 4096),
            batch_size=config.get("batch_size", 32),
        )

    async def score_batch(self, queries, passages, candidate_length=None, **kwargs):
        """Standard score_batch implementation"""
        if candidate_length is None:
            candidate_length = [len(passages)]
        assert len(candidate_length) == len(queries)
        assert sum(candidate_length) == len(passages)

        # Expand queries
        expanded_queries = sum([[queries[i]] * l for i, l in enumerate(candidate_length)], [])
        pairs = list(zip(expanded_queries, passages))

        # Score
        all_scores = self.model.score(pairs)

        # Regroup
        start = 0
        result = []
        for length in candidate_length:
            result.append(all_scores[start:start + length])
            start += length
        return result
```

## Configuration Template

Create a JSON config file to load your extension:

```json
{
    "file_imports": ["./examples/my_reranker_extension.py"],
    "services": [
        {
            "name": "my-reranker",
            "engine": "LLMRerankerEngine",
            "cache": 1024,
            "cache_ttl": 600,
            "batch_size": 32,
            "max_wait_time": 0.05,
            "config": {
                "model": "your-org/reranker-model",
                "ngpus": 1,
                "context_size": 4096,
                "batch_size": 32
            }
        }
    ]
}
```

**Key config fields:**
- `file_imports`: Python files to load (your extension)
- `name`: Service identifier (used in API requests)
- `engine`: Your Engine class name (must match class name exactly)
- `cache`, `cache_ttl`: Request caching settings
- `batch_size`, `max_wait_time`: RoutIR's dynamic batching settings
- `config`: Passed to your Engine's `__init__` as the `config` parameter

## Alternative: Classification Model Reranker

For models using `AutoModelForSequenceClassification` (not generative LLMs):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ClassificationReranker:
    def __init__(self, model_name: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use SequenceClassification, not CausalLM
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()

    def _format_pair(self, query: str, doc: str) -> str:
        """Format according to your model's template"""
        return f"query: {query} document: {doc}"

    @torch.inference_mode()
    def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # Format inputs
        texts = [self._format_pair(q, d) for q, d in pairs]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.model.device)

        # Get logits from classification head
        logits = self.model(**inputs).logits

        # Extract scores (usually from single output neuron)
        scores = logits.view(-1).cpu().tolist()

        return scores
```

## Testing Your Implementation

### 1. Start the Service

```bash
# Install dependencies
pip install routir transformers torch

# Start server
routir my_reranker_config.json --port 5000
```

### 2. Test with curl

```bash
# Test scoring endpoint
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "service": "my-reranker",
    "query": "What is machine learning?",
    "passages": [
      "Machine learning is a subset of AI",
      "Pizza is a popular Italian dish",
      "Neural networks are used in ML"
    ]
  }'
```

**Expected output:**
```json
{
  "cached": false,
  "processed": true,
  "query": "What is machine learning?",
  "scores": [0.95, 0.05, 0.87],
  "service": "my-reranker",
  "timestamp": 1234567890.123
}
```

### 3. Test in Python

```python
import requests

response = requests.post(
    "http://localhost:5000/score",
    json={
        "service": "my-reranker",
        "query": "test query",
        "passages": ["doc1", "doc2", "doc3"]
    }
)

print(response.json()["scores"])  # [score1, score2, score3]
```

### 4. Check Available Services

```bash
curl http://localhost:5000/avail
```

Should show your service under the "score" category.

## Common Patterns

### Error Handling
Always provide fallback scores (0.5 for neutral):

```python
try:
    score = compute_score(query, doc)
except Exception as e:
    logger.warning(f"Scoring failed: {e}")
    score = 0.5  # Neutral score
```

### Text Truncation
Truncate documents to fit context window:

```python
def truncate_doc(self, doc: str, max_len: int) -> str:
    tokens = self.tokenizer(doc)["input_ids"]
    if len(tokens) > max_len:
        return self.tokenizer.decode(tokens[:max_len])
    return doc
```

### Batch Processing
Process large inputs in smaller batches:

```python
def score(self, pairs: List[Tuple[str, str]]) -> List[float]:
    all_scores = []
    for i in range(0, len(pairs), self.batch_size):
        batch = pairs[i:i + self.batch_size]
        batch_scores = self._score_batch(batch)
        all_scores.extend(batch_scores)
    return all_scores
```

## Key Takeaways

1. **Two-layer pattern**: Core model class + Engine wrapper
2. **score_batch is critical**: Expand queries, score pairs, regroup results
3. **Always validate**: Check `candidate_length` matches queries and passages
4. **Use fallbacks**: Return 0.5 for failed scores, don't crash
5. **Follow the template**: Your Engine's config dict goes to `__init__(config=...)`
6. **Test thoroughly**: Use curl to verify scoring endpoint works
7. **File imports**: Add your .py file to config's `file_imports` array

## Reference Implementations

See these examples in the `examples/` directory:
- `rank1_extension.py` - LLM reranker with reasoning
- `vllm_qwen3reranker_extension.py` - vLLM-based reranker
- `pyserini_extension.py` - BM25 search engine
- `pyterrier_extension.py` - PyTerrier retrieval pipelines

All follow the same two-layer pattern shown in this guide.
