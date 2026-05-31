# LCEL — LangChain Expression Language & Runnables

## What is LCEL?

**LangChain Expression Language (LCEL)** is the composition system at the heart of modern LangChain. It provides a declarative, pipe-operator-based syntax for building chains from composable primitives called **Runnables**.

Every LangChain component (prompts, models, parsers, retrievers, tools) implements the `Runnable` interface — making them interchangeable building blocks that can be connected with the `|` operator.

---

## Why LCEL Exists

Before LCEL, LangChain had "legacy chains" — hardcoded `LLMChain`, `RetrievalQAChain`, etc. These were:
- **Not composable** — each chain had fixed structure
- **Not streamable** — no native token streaming
- **Not batchable** — no concurrent execution support
- **Not async-native** — bolted on later

LCEL solves all of these in a single unified interface.

---

## The Runnable Interface

Every LCEL component implements these methods:

```python
from langchain_core.runnables import Runnable

# Core execution methods (all components implement these)
runnable.invoke(input)             # Single synchronous call
runnable.stream(input)             # Streaming generator
runnable.batch([input1, input2])   # Concurrent batch execution

# Async variants
await runnable.ainvoke(input)
async for chunk in runnable.astream(input):  ...
await runnable.abatch([input1, input2])

# Streaming structured data (not just text)
async for chunk in runnable.astream_events(input, version="v2"):  ...
```

---

## Basic Chain Composition

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Three Runnables composed into a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}."),
    ("user", "{question}")
])
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# | operator creates a RunnableSequence
chain = prompt | model | parser

# Invoke
answer = chain.invoke({
    "domain": "distributed systems",
    "question": "Explain the CAP theorem in 2 sentences."
})
```

The `|` operator creates a `RunnableSequence` — the output of each step is passed as input to the next.

---

## Streaming

LCEL provides first-class streaming support. Tokens stream as they're generated:

```python
# Stream text tokens
for chunk in chain.stream({"domain": "AI", "question": "What is LangChain?"}):
    print(chunk, end="", flush=True)

# Async streaming (for FastAPI / asyncio contexts)
async for chunk in chain.astream({"domain": "AI", "question": "..."}):
    print(chunk, end="", flush=True)
```

### `astream_events` — Granular Event Streaming

For complex chains, stream individual component events:

```python
async for event in chain.astream_events(input, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif kind == "on_tool_start":
        print(f"\nCalling tool: {event['name']}")
    elif kind == "on_retriever_end":
        print(f"\nRetrieved {len(event['data']['output'])} docs")
```

---

## Batch Execution

```python
# Executes concurrently (default: max_concurrency=None)
results = chain.batch([
    {"domain": "biology", "question": "What is DNA?"},
    {"domain": "physics", "question": "What is entropy?"},
    {"domain": "chemistry", "question": "What is pH?"},
], config={"max_concurrency": 3})

# With error handling — return exceptions instead of raising
results = chain.batch(
    inputs,
    config={"max_concurrency": 5},
    return_exceptions=True,
)
```

---

## `RunnableParallel` — Fan-Out

Run multiple chains on the same input simultaneously:

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Compute multiple things from the same input
parallel = RunnableParallel({
    "summary": summary_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain,
})

result = parallel.invoke("LangChain is a powerful framework for building LLM applications.")
# result = {
#   "summary": "...",
#   "keywords": [...],
#   "sentiment": "positive"
# }
```

Dict shorthand (equivalent):
```python
chain = {
    "summary": summary_chain,
    "keywords": keyword_chain,
} | combine_results_prompt | model | parser
```

---

## `RunnablePassthrough` — Pass Input Through

Useful to include original input in the pipeline alongside transformed values:

```python
from langchain_core.runnables import RunnablePassthrough

# RAG pattern: pass original question alongside retrieved context
rag_chain = (
    {
        "context": retriever,          # retriever receives question, returns docs
        "question": RunnablePassthrough()  # original question passed through
    }
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("What is LangChain?")
```

---

## `RunnableLambda` — Wrap Any Function

Convert any Python function into a Runnable:

```python
from langchain_core.runnables import RunnableLambda

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Wrap function as Runnable
format_runnable = RunnableLambda(format_docs)

# Use in chain
chain = retriever | format_runnable | prompt | model | parser
```

---

## `RunnableBranch` — Conditional Routing

Route to different chains based on the input:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "urgent" in x["question"].lower(), urgent_chain),
    (lambda x: "technical" in x["question"].lower(), technical_chain),
    default_chain,  # fallback
)
```

Modern alternative — conditional with `RunnableLambda`:

```python
def route(input):
    if "code" in input["question"]:
        return code_chain
    else:
        return general_chain

chain = RunnableLambda(route)
```

---

## `RunnableWithMessageHistory` — Conversation Memory

Wrap a chain to maintain conversation history across calls:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Chain that takes messages as input
chain = prompt | model | parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Different session_ids are independent conversations
result = chain_with_history.invoke(
    {"question": "My name is Alice."},
    config={"configurable": {"session_id": "user_123"}},
)
result2 = chain_with_history.invoke(
    {"question": "What's my name?"},
    config={"configurable": {"session_id": "user_123"}},
)
# result2 = "Your name is Alice."
```

> **Production note**: Use LangGraph with a persistent checkpointer instead of `RunnableWithMessageHistory` for production conversation apps. LangGraph gives you more control.

---

## Config — Runtime Configuration

Pass runtime config to any Runnable:

```python
from langchain_core.runnables import RunnableConfig

config: RunnableConfig = {
    "configurable": {           # Values accessible via configurable_fields
        "temperature": 0.7,
        "session_id": "abc123",
    },
    "callbacks": [my_tracer],   # Custom callbacks
    "tags": ["prod", "rag"],    # Tags for observability
    "metadata": {               # Metadata for tracing
        "user_id": "user_456",
        "request_id": "req_789",
    },
    "max_concurrency": 5,       # Batch concurrency limit
    "recursion_limit": 25,      # LangGraph recursion limit
}

result = chain.invoke(input, config=config)
```

---

## `configurable_fields` — Runtime Model Swapping

Allow runtime configuration of Runnable parameters:

```python
from langchain_core.runnables import ConfigurableField

model = ChatOpenAI(model="gpt-4o").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Controls randomness of output",
    )
)

# Use default temperature
result = model.invoke("What is 2+2?")

# Override at runtime
result = model.invoke(
    "Write a poem",
    config={"configurable": {"llm_temperature": 0.9}}
)
```

---

## Error Handling & Fallbacks

```python
# Fallback to a different model if primary fails
primary = ChatOpenAI(model="gpt-4o")
fallback = ChatOpenAI(model="gpt-4o-mini")

chain_with_fallback = primary.with_fallbacks([fallback])

# Retry with exponential backoff
chain_with_retry = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

---

## Inspecting Chains

```python
# Visualize chain structure
print(chain.get_graph().print_ascii())

# Get input schema
chain.input_schema.schema()

# Get output schema
chain.output_schema.schema()
```

---

## Common Pitfalls

### 1. Forgetting `RunnablePassthrough` in RAG Chains
```python
# ❌ The question is consumed by the retriever — not available to prompt
chain = retriever | prompt | model

# ✅ Pass question through while also retrieving context
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | model
)
```

### 2. Blocking Async Chains with Sync I/O
```python
# ❌ Running sync chain in async context blocks event loop
result = chain.invoke(input)  # inside async function

# ✅ Use async variant
result = await chain.ainvoke(input)
```

### 3. Not Using `return_exceptions=True` in Batch
```python
# ❌ One failure aborts the entire batch
results = chain.batch(100_inputs)

# ✅ Collect failures, process successes
results = chain.batch(100_inputs, return_exceptions=True)
successes = [r for r in results if not isinstance(r, Exception)]
failures = [r for r in results if isinstance(r, Exception)]
```

---

## Performance Considerations

| Pattern | Performance | Notes |
|---|---|---|
| `invoke()` | Baseline | Single request |
| `batch()` | High | Concurrent; set `max_concurrency` |
| `stream()` | Better UX | TTFT reduced for user |
| `astream()` | Best for async | Non-blocking event loop |
| `astream_events()` | Most detailed | Higher overhead |

- **Set `max_concurrency`** in `.batch()` to avoid rate-limit errors
- **Use `astream()`** in FastAPI endpoints for streaming responses
- **Cache intermediate results** for repeated inputs using `cache=True` on the model

---

## Related Topics
- [`03-chat-models-llms.md`](./03-chat-models-llms.md) — Chat model Runnables
- [`04-prompts-templates.md`](./04-prompts-templates.md) — Prompt Runnables
- [`05-output-parsers.md`](./05-output-parsers.md) — Parser Runnables
- [`07-rag-pipeline.md`](./07-rag-pipeline.md) — LCEL in RAG chains
