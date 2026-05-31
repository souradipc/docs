# Chat Models, LLMs & Embeddings

## Chat Models vs LLMs — The Distinction

LangChain supports two model types, but they are architecturally different:

| Aspect | ChatModel | LLM (legacy) |
|---|---|---|
| Input | List of `Messages` | Plain string |
| Output | `AIMessage` | Plain string |
| Architecture | Trained for conversations | Completion model |
| Use today | **Default choice** | Legacy, use rarely |
| Examples | GPT-4o, Claude 3.5, Gemini 2 | GPT-3, legacy Llama |

> **Modern rule**: Always use ChatModels. Nearly all cutting-edge models are chat-optimized. LLMs in LangChain are preserved for backward compatibility.

---

## Initializing Chat Models

### Method 1: `init_chat_model` (Recommended — Provider-Agnostic)

```python
from langchain.chat_models import init_chat_model

# Specify provider:model
model = init_chat_model("openai:gpt-4o")
model = init_chat_model("anthropic:claude-sonnet-4-5")
model = init_chat_model("google_genai:gemini-2.0-flash")
model = init_chat_model("ollama:llama3.2")

# With parameters
model = init_chat_model(
    "openai:gpt-4o",
    temperature=0.0,
    max_tokens=2048,
)
```

### Method 2: Provider-Specific Classes

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

openai_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_tokens=4096,
    timeout=60,
    max_retries=3,
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
)

anthropic_model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=4096,
)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
)
```

---

## Invoking Models

### Basic Invocation

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="Write a Python function to reverse a string."),
]

response = model.invoke(messages)
print(response.content)           # The text response
print(response.usage_metadata)    # Token counts
print(response.model)             # Which model actually responded
```

### Simple String Input (Auto-Converts to HumanMessage)

```python
# String is automatically wrapped as HumanMessage
response = model.invoke("What is 2 + 2?")
```

---

## Streaming

```python
# Synchronous streaming
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)

# Async streaming (for FastAPI, asyncio)
async for chunk in model.astream(messages):
    print(chunk.content, end="", flush=True)

# Stream with events (get reasoning, tool calls, etc.)
async for event in model.astream_events(messages, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## Structured Output

Force the model to return data matching a schema:

### Using Pydantic (Recommended)

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 0 to 10", ge=0, le=10)
    summary: str = Field(description="One-sentence summary")
    recommended: bool

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(MovieReview)

review = structured_model.invoke("Review the movie Inception")
print(review.title)        # "Inception"
print(review.rating)       # 9.5
print(type(review))        # <class 'MovieReview'>
```

### Using JSON Schema

```python
json_schema = {
    "title": "ExtractedData",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

structured_model = model.with_structured_output(json_schema, method="json_schema")
result = structured_model.invoke("Extract: Alice is 30 years old")
# result = {"name": "Alice", "age": 30}
```

### Structured Output Methods

| Method | Description | Best for |
|---|---|---|
| `"function_calling"` | Uses tool calling | Most reliable |
| `"json_schema"` | JSON schema enforcement | OpenAI, latest models |
| `"json_mode"` | JSON response mode | Fallback option |

```python
# With include_raw=True — get both raw and parsed
result = model.with_structured_output(
    MovieReview,
    include_raw=True
).invoke("Review Inception")

print(result["parsed"])   # MovieReview object
print(result["raw"])      # Original AIMessage
print(result["parsing_error"])  # None if successful
```

---

## Tool Calling / Function Calling

Bind tools to the model so it can call them:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is 22°{unit[0].upper()}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Bind tools to model
model_with_tools = model.bind_tools([get_weather, calculator])

# Model decides whether to call a tool
response = model_with_tools.invoke("What's the weather in Paris?")

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(tool_call["name"])    # "get_weather"
        print(tool_call["args"])    # {"city": "Paris", "unit": "celsius"}
```

---

## Multimodal Inputs

Pass images, audio, or video alongside text:

```python
from langchain_core.messages import HumanMessage
import base64

# Image from URL
message = HumanMessage(content=[
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"},
    },
    {
        "type": "text",
        "text": "What's in this image?"
    }
])
response = model.invoke([message])

# Image from base64
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = HumanMessage(content=[
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
    },
    {"type": "text", "text": "Describe this image."}
])
```

---

## Model Configuration — Key Parameters

| Parameter | Type | Effect |
|---|---|---|
| `temperature` | 0.0 – 2.0 | Randomness. 0 = deterministic, 1+ = creative |
| `max_tokens` | int | Max output tokens |
| `top_p` | 0.0 – 1.0 | Nucleus sampling threshold |
| `stop` | list[str] | Stop generation at these tokens |
| `timeout` | int | API request timeout in seconds |
| `max_retries` | int | Retry count on transient errors |
| `streaming` | bool | Enable streaming by default |
| `model_kwargs` | dict | Provider-specific extra params |

---

## Embeddings

Embedding models convert text to high-dimensional vectors for semantic search:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# OpenAI embeddings (paid, cloud)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace (free, local)
local_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Embed a query
vector = embeddings.embed_query("What is LangChain?")
print(len(vector))  # 1536 for text-embedding-3-small

# Embed multiple documents (batch)
vectors = embeddings.embed_documents([
    "LangChain is a framework.",
    "LangGraph is for agents.",
    "LangSmith is for observability.",
])
```

### Choosing an Embedding Model

| Model | Dimensions | Cost | Quality | Use Case |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | Low | Good | Most production use |
| `text-embedding-3-large` | 3072 | Medium | Better | High-stakes retrieval |
| `text-embedding-ada-002` | 1536 | Low | Legacy | Avoid for new projects |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Free | Good | Local/private data |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Free | Better | Local/private data |
| `BAAI/bge-large-en-v1.5` | 1024 | Free | Excellent | Best open-source option |

---

## Caching Model Responses

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache, RedisCache

# In-memory cache (dev)
set_llm_cache(InMemoryCache())

# Persistent cache (production)
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Distributed cache (multi-worker)
set_llm_cache(RedisCache(redis_url="redis://localhost:6379"))

# Now identical calls return cached responses instantly
# Saves API costs and latency for repeated prompts
model.invoke("What is 2 + 2?")  # First call → API request
model.invoke("What is 2 + 2?")  # Second call → cache hit
```

---

## Fallbacks and Resilience

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

primary = ChatOpenAI(model="gpt-4o")
fallback = ChatAnthropic(model="claude-3-5-sonnet")

# Fall back to Claude if GPT-4o fails
resilient_model = primary.with_fallbacks([fallback])

# Retry with exponential backoff
retrying_model = primary.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

---

## Rate Limiting and Token Management

```python
# Track token usage
response = model.invoke(messages)
usage = response.usage_metadata
print(f"Input tokens: {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")

# Estimate cost (OpenAI gpt-4o example)
INPUT_COST_PER_1K = 0.0025   # $2.50/M tokens
OUTPUT_COST_PER_1K = 0.01    # $10.00/M tokens
cost = (
    usage['input_tokens'] / 1000 * INPUT_COST_PER_1K +
    usage['output_tokens'] / 1000 * OUTPUT_COST_PER_1K
)
```

---

## Common Pitfalls

### 1. Not Handling Rate Limits
```python
# ❌ No retry — 429 errors crash the app
model = ChatOpenAI(model="gpt-4o")

# ✅ Built-in retry
model = ChatOpenAI(model="gpt-4o", max_retries=3)
# OR
model = model.with_retry(stop_after_attempt=3)
```

### 2. Using Temperature=0 for All Tasks
```python
# ❌ Creative tasks need temperature > 0
summarizer = ChatOpenAI(model="gpt-4o", temperature=0)  # Too rigid for writing

# ✅ Match temperature to use case
extractor = ChatOpenAI(model="gpt-4o", temperature=0)     # Extraction: deterministic
writer = ChatOpenAI(model="gpt-4o", temperature=0.7)       # Writing: creative
coder = ChatOpenAI(model="gpt-4o", temperature=0.1)        # Code: mostly deterministic
```

### 3. Ignoring `usage_metadata` — No Cost Tracking
Always track token usage in production to monitor spend and set budgets.

---

## Security Considerations

- **Never log full messages** in production — they may contain PII
- **Store API keys** in secrets manager (not in code or `.env`)
- **Set `max_tokens`** to prevent runaway generation costs
- **Validate model outputs** before using in critical paths
- **Prompt injection** — sanitize user input before including in system prompts

---

## Related Topics
- [`02-lcel-runnables.md`](./02-lcel-runnables.md) — Models as Runnables
- [`04-prompts-templates.md`](./04-prompts-templates.md) — Formatting model inputs
- [`05-output-parsers.md`](./05-output-parsers.md) — Parsing model outputs
- [`06-document-loaders-splitters.md`](./06-document-loaders-splitters.md) — Embeddings for RAG
