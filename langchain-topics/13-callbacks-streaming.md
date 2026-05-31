# Callbacks & Streaming

## Callbacks — Hooking into Execution

Callbacks let you observe what's happening inside chains, models, and agents at every step. Use cases:

- **Logging** — Log every LLM call with inputs/outputs
- **Monitoring** — Track latency, token usage, errors
- **Streaming** — Pipe tokens to the UI in real time
- **Debugging** — Inspect intermediate chain values
- **Cost tracking** — Count tokens per request

---

## `BaseCallbackHandler` — Custom Callbacks

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any
import time

class LoggingCallbackHandler(BaseCallbackHandler):
    """Log all LLM calls with timing information."""

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self.start_time = time.time()
        print(f"[LLM Start] Model: {serialized.get('name', 'unknown')}")
        print(f"[LLM Start] Prompt preview: {prompts[0][:100]}...")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        elapsed = time.time() - self.start_time
        token_usage = response.llm_output.get("token_usage", {})
        print(f"[LLM End] Duration: {elapsed:.2f}s")
        print(f"[LLM End] Tokens: {token_usage}")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        print(f"[LLM Error] {type(error).__name__}: {error}")

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        print(f"[Chain Start] {serialized.get('name', 'chain')}")

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        print(f"[Chain End] Output keys: {list(outputs.keys())}")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        print(f"[Tool Start] {serialized.get('name')}: {input_str[:100]}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"[Tool End] Result: {output[:100]}")

# Use callback
callback = LoggingCallbackHandler()

result = model.invoke(
    [("human", "What is 2 + 2?")],
    config={"callbacks": [callback]},
)
```

---

## Async Callbacks

```python
from langchain_core.callbacks import AsyncCallbackHandler

class AsyncStreamingCallback(AsyncCallbackHandler):
    """Async callback for real-time streaming."""

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called for each new token during streaming."""
        await self.queue.put(token)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        await self.queue.put(None)  # Sentinel: stream complete
```

---

## Built-in Streaming Callbacks

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Stream directly to stdout
model = ChatOpenAI(
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
model.invoke([("human", "Tell me a story")])
# Tokens print in real time
```

---

## LCEL Streaming — Token-by-Token

The modern way to stream:

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# Synchronous streaming
for token in chain.stream({"question": "Explain quantum computing"}):
    print(token, end="", flush=True)

# Async streaming
async for token in chain.astream({"question": "Explain quantum computing"}):
    print(token, end="", flush=True)
```

---

## `astream_events` — Rich Event Stream

Get detailed events including tool calls, LLM tokens, chain boundaries:

```python
async for event in chain.astream_events(
    {"question": "What tools are available?"},
    version="v2",
):
    kind = event["event"]
    name = event["name"]

    if kind == "on_chat_model_stream":
        # LLM token
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)

    elif kind == "on_tool_start":
        print(f"\n[Tool Started]: {name}")
        print(f"Input: {event['data']['input']}")

    elif kind == "on_tool_end":
        print(f"[Tool Result]: {event['data']['output']}")

    elif kind == "on_chain_start":
        print(f"\n[Chain]: {name} started")

    elif kind == "on_chain_end":
        print(f"[Chain]: {name} completed")
```

### Event Types Reference

| Event | When Fired | Data |
|---|---|---|
| `on_chat_model_start` | LLM call begins | `input` messages |
| `on_chat_model_stream` | Each token | `chunk` with content |
| `on_chat_model_end` | LLM call completes | Full response |
| `on_tool_start` | Tool begins | `input` |
| `on_tool_end` | Tool completes | `output` |
| `on_chain_start` | Any chain/runnable starts | `input` |
| `on_chain_end` | Any chain/runnable ends | `output` |
| `on_retriever_start` | Retriever begins | `query` |
| `on_retriever_end` | Retriever completes | `documents` |

---

## Streaming in FastAPI (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

app = FastAPI()

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

@app.post("/stream")
async def stream_response(request: dict):
    async def generate():
        async for token in chain.astream({"question": request["question"]}):
            # Server-Sent Events format
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx: disable buffering
        },
    )
```

### Streaming with `astream_events` in FastAPI

```python
@app.post("/agent/stream")
async def agent_stream(request: AgentRequest):
    async def generate():
        async for event in agent.astream_events(
            {"messages": [("human", request.message)]},
            version="v2",
            config={"configurable": {"thread_id": request.thread_id}},
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            elif event["event"] == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name']})}\n\n"

            elif event["event"] == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'output': str(event['data']['output'])})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Global Callbacks — Tracing All Calls

```python
from langchain_core.callbacks import set_handler

# Set a global callback handler — applies to ALL LLM calls
handler = LoggingCallbackHandler()
set_handler(handler)

# Now all model calls are logged automatically
model.invoke("Hello")  # Logged
chain.invoke({"question": "Hi"})  # Also logged
```

---

## Callback Scope — Propagation

Callbacks propagate down through chains:

```python
# Callback set on the chain applies to all inner components
chain = prompt | model | parser

result = chain.invoke(
    {"question": "Hello"},
    config={"callbacks": [my_callback]},
    # ↑ Applies to: prompt formatting, LLM call, parser
)

# Callback set on model only — doesn't propagate up
model_with_callback = model.with_config(callbacks=[my_callback])
chain = prompt | model_with_callback | parser
# Only the LLM call is logged, not the prompt or parser
```

---

## Token Streaming in LangGraph

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)

# Stream with LangGraph
async for event in agent.astream_events(
    {"messages": [("human", "What's the weather in Paris?")]},
    version="v2",
):
    event_type = event["event"]
    node_name = event.get("metadata", {}).get("langgraph_node", "")

    if event_type == "on_chat_model_stream" and node_name == "agent":
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)
```

---

## Common Pitfalls

### 1. Blocking Async Code in Callbacks
```python
# ❌ Sync sleep blocks the event loop in async contexts
class BadCallback(AsyncCallbackHandler):
    async def on_llm_end(self, *args, **kwargs):
        import time
        time.sleep(1)  # Blocks everything!

# ✅ Use asyncio.sleep
class GoodCallback(AsyncCallbackHandler):
    async def on_llm_end(self, *args, **kwargs):
        import asyncio
        await asyncio.sleep(1)  # Non-blocking
```

### 2. Not Flushing in Token Streaming
```python
# ❌ Buffer builds up, tokens appear in batches
for token in chain.stream(input):
    print(token, end="")  # May buffer

# ✅ Force flush
for token in chain.stream(input):
    print(token, end="", flush=True)
```

---

## Related Topics
- [`02-lcel-runnables.md`](./02-lcel-runnables.md) — Streaming Runnables
- [`11-langgraph.md`](./11-langgraph.md) — Graph streaming modes
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — Production tracing
