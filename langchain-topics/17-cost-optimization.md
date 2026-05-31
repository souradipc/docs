# Cost Optimization

## LLM Cost Drivers

Understanding costs lets you architect cheaper applications:

| Driver | What It Costs | Optimization |
|---|---|---|
| Input tokens | Per prompt + context | Shorter prompts, smaller context |
| Output tokens | Per response | Limit `max_tokens`, concise instructions |
| Model tier | Premium vs economy | Use cheaper models where possible |
| Embedding calls | Per embed_documents call | Cache embeddings, don't re-embed |
| API call frequency | Per request | Cache responses, batch requests |

---

## Token Counting

Before optimizing, measure what you're spending:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import tiktoken

# Count tokens before sending (GPT-4 tokenizer)
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Check usage metadata after every call
model = ChatOpenAI(model="gpt-4o")
response = model.invoke([HumanMessage(content="What is phishing?")])

usage = response.usage_metadata
print(f"Input tokens:  {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")
print(f"Total tokens:  {usage['total_tokens']}")

# Estimate cost (update prices as needed)
GPT4O_IN = 2.50 / 1_000_000   # $2.50 per 1M input tokens
GPT4O_OUT = 10.00 / 1_000_000  # $10.00 per 1M output tokens

cost = (
    usage['input_tokens'] * GPT4O_IN +
    usage['output_tokens'] * GPT4O_OUT
)
print(f"Estimated cost: ${cost:.6f}")
```

---

## Caching Model Responses

Cache identical prompts to avoid redundant API calls:

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache, RedisCache

# Dev / testing
set_llm_cache(InMemoryCache())

# Single-instance production (survives restarts)
set_llm_cache(SQLiteCache(database_path="/app/data/llm_cache.db"))

# Multi-worker production (shared across processes)
import redis
redis_client = redis.from_url("redis://localhost:6379")
set_llm_cache(RedisCache(redis_client))

# Result: Identical prompts return instantly from cache
model.invoke([("human", "What is 2 + 2?")])  # 1st call → API ($$$)
model.invoke([("human", "What is 2 + 2?")])  # 2nd call → cache (free)
```

Cache hit rates of 10-30% are common in production, providing significant savings.

---

## Model Tiering Strategy

Not every task needs GPT-4o. Route to cheaper models when appropriate:

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch

# Define model tiers
cheap_model = ChatOpenAI(model="gpt-4o-mini")    # ~15x cheaper than GPT-4o
premium_model = ChatOpenAI(model="gpt-4o")

# Use cheap model for simple tasks
def classify_complexity(input_text: str) -> str:
    """Returns 'simple' or 'complex' based on question."""
    # Simple heuristics or a very cheap classification call
    words = input_text.split()
    if len(words) < 10 and "?" in input_text:
        return "simple"
    return "complex"

# Route to appropriate model
def tiered_model(input_text: str) -> str:
    complexity = classify_complexity(input_text)
    model = cheap_model if complexity == "simple" else premium_model
    return model.invoke([("human", input_text)]).content

# Example routing
print(tiered_model("What is 2+2?"))         # → gpt-4o-mini
print(tiered_model("Analyze this complex security incident report..."))  # → gpt-4o
```

---

## Prompt Compression

Reduce token count without losing information:

```python
# ❌ Verbose prompt
verbose_system = """
You are an extremely helpful, friendly, and knowledgeable assistant who has been trained
on a large corpus of data. You should provide thoughtful, accurate, and comprehensive
responses to all questions. Please make sure your answers are clear and well-organized.
When answering, consider all aspects of the question carefully.
"""
# → ~70 tokens for system alone

# ✅ Concise equivalent
concise_system = "You are a helpful assistant. Provide accurate, clear answers."
# → ~12 tokens

# Savings: ~58 tokens × every API call × millions of calls = significant cost
```

---

## Context Window Budgeting

The RAG context is the biggest token cost. Manage it carefully:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Calculate your token budget
MODEL_CONTEXT_WINDOW = 128_000  # gpt-4o
SYSTEM_PROMPT_TOKENS = 200
OUTPUT_RESERVE_TOKENS = 2_000
HISTORY_TOKENS = 1_000

CONTEXT_BUDGET = MODEL_CONTEXT_WINDOW - SYSTEM_PROMPT_TOKENS - OUTPUT_RESERVE_TOKENS - HISTORY_TOKENS
# = 124_800 tokens available for retrieved context

# But cost-optimize by using less than max
COST_OPTIMAL_CONTEXT = 8_000  # Retrieve ~8K tokens (8 × 1K-char chunks)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # 6 × ~500 tokens = 3K tokens
```

---

## Batch Processing

When you have many inputs, batch them to reduce overhead:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# ❌ Sequential — slow and expensive (separate requests)
results = []
for question in questions:
    result = model.invoke([("human", question)])
    results.append(result.content)

# ✅ Batch — more efficient (single connection overhead)
results = model.batch([
    [("human", q)] for q in questions
], config={"max_concurrency": 5})  # 5 concurrent requests

# ✅ Async gather — also efficient
import asyncio

async def batch_async(questions: list[str]) -> list[str]:
    tasks = [model.ainvoke([("human", q)]) for q in questions]
    responses = await asyncio.gather(*tasks)
    return [r.content for r in responses]
```

---

## Embedding Cost Reduction

Embeddings are called once (at index time), but cost can be significant for large corpora:

```python
# 1. Cache embeddings to disk — don't re-embed on restart
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db",  # Cache to disk
)
# On restart: load from cache (zero embedding cost)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Use smaller embedding model (free local model)
from langchain_community.embeddings import HuggingFaceEmbeddings

local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, free
)
# Compare quality — often 80-90% of OpenAI quality at 0% cost

# 3. Reduce dimensions (OpenAI text-embedding-3 supports this)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=512,   # Reduce from 3072 → 512: 6x smaller, modest quality drop
)
```

---

## Output Length Control

Output tokens are 4x more expensive than input tokens on GPT-4o. Constrain output length:

```python
# 1. Limit via max_tokens
model = ChatOpenAI(model="gpt-4o", max_tokens=500)

# 2. Instruct model to be concise in system prompt
system_prompt = """Answer in 2-3 sentences maximum.
Be direct and concise. No preamble or repetition."""

# 3. For structured outputs — use Pydantic to constrain shape
class Answer(BaseModel):
    summary: str = Field(description="One sentence summary")
    details: list[str] = Field(description="Up to 3 bullet points", max_length=3)
```

---

## Cost Tracking Dashboard

```python
from langchain_core.callbacks import BaseCallbackHandler
from collections import defaultdict
from datetime import datetime
import threading

class CostTracker(BaseCallbackHandler):
    """Track costs across all model calls."""

    # Pricing per 1M tokens (update as needed)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    }

    def __init__(self):
        self.usage: dict[str, dict] = defaultdict(lambda: {"input": 0, "output": 0})
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        model_name = kwargs.get("invocation_params", {}).get("model_name", "unknown")
        usage = response.llm_output.get("token_usage", {})

        with self._lock:
            self.usage[model_name]["input"] += usage.get("prompt_tokens", 0)
            self.usage[model_name]["output"] += usage.get("completion_tokens", 0)

    def get_total_cost(self) -> float:
        total = 0.0
        for model, tokens in self.usage.items():
            pricing = self.PRICING.get(model, {"input": 0, "output": 0})
            total += (tokens["input"] * pricing["input"] / 1_000_000)
            total += (tokens["output"] * pricing["output"] / 1_000_000)
        return total

    def summary(self) -> dict:
        return {
            "models": dict(self.usage),
            "total_cost_usd": self.get_total_cost(),
        }

# Use globally
tracker = CostTracker()
# Register as global handler or pass per-request
```

---

## Cost Comparison: Strategy Impact

| Strategy | Token Reduction | Cost Reduction | Quality Impact |
|---|---|---|---|
| Model tiering (mini vs 4o) | N/A | ~85% | Low for simple tasks |
| Response caching | Up to 30% reduction | 30% | None |
| Prompt compression | 20-40% input reduction | 10-20% | Minimal |
| Local embeddings | 100% embed cost | ~40% total | Minimal |
| Reduced context (k=4 vs k=10) | 60% context reduction | 20-30% | Some retrieval |
| Output length control | 50% output tokens | 25-35% | Minimal |

---

## Related Topics
- [`03-chat-models-llms.md`](./03-chat-models-llms.md) — Model tiers and caching
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — Token usage monitoring
- [`16-production-deployment.md`](./16-production-deployment.md) — Production configuration
