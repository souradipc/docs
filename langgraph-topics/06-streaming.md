# Streaming

## Why Streaming Matters for Agents

Agent workflows can take seconds to minutes. Without streaming:
- Users see a blank screen until the entire workflow completes
- You can't show which step is running
- LLM token generation appears as a single delayed response

LangGraph has the most comprehensive streaming support in the LangChain ecosystem, with five distinct stream modes.

---

## Stream Modes Reference

| Mode | What You Get | Best For |
|---|---|---|
| `"values"` | Full state after every step | State inspection, debugging |
| `"updates"` | Only changed state per step | Observing progress |
| `"messages"` | LLM token chunks + metadata | Real-time token streaming to UI |
| `"debug"` | Low-level execution events | Deep debugging |
| `"custom"` | User-emitted events via `StreamWriter` | Custom progress signals |
| Multiple | Combine any of the above | Production APIs |

---

## `stream_mode="values"` — Full State at Each Step

Returns the complete graph state after each superstep:

```python
for chunk in graph.stream(
    {"topic": "cybersecurity"},
    stream_mode="values",
    version="v2",
):
    if chunk["type"] == "values":
        state = chunk["data"]
        print(f"Messages: {len(state.get('messages', []))}")
        print(f"Status: {state.get('status', 'unknown')}")
```

---

## `stream_mode="updates"` — Only Deltas

Returns only the state changes produced by each node. More efficient than `"values"`:

```python
for chunk in graph.stream(
    {"messages": [("human", "Analyze this threat")]},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state_update in chunk["data"].items():
            print(f"Node '{node_name}' updated:")
            for key, value in state_update.items():
                print(f"  {key}: {str(value)[:100]}")
```

Output:
```
Node 'classify_threat' updated:
  threat_type: malware
  confidence: 0.95
Node 'gather_intel' updated:
  intel_results: [{'source': 'vtotal', ...}]
Node 'llm' updated:
  messages: [AIMessage(content='Based on the analysis...')]
```

---

## `stream_mode="messages"` — Token-by-Token Streaming

Returns individual LLM token chunks as they're generated. Essential for chat UIs:

```python
for chunk in graph.stream(
    {"messages": [("human", "Write a threat report")]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        node_name = metadata.get("langgraph_node", "unknown")

        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
```

### Filter by Node

```python
async for chunk in graph.astream(
    {"messages": [("human", "Analyze this")]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        # Only show tokens from the "report_writer" node
        if metadata.get("langgraph_node") == "report_writer":
            if message_chunk.content:
                print(message_chunk.content, end="", flush=True)
```

---

## `stream_mode="custom"` — Application Events

Emit custom events from inside node functions using `StreamWriter`:

```python
from langgraph.types import StreamWriter

def research_node(state: State, writer: StreamWriter) -> dict:
    writer({"status": "starting", "phase": "web_search"})

    results = search_web(state["query"])
    writer({"status": "search_complete", "count": len(results)})

    intel = enrich_results(results)
    writer({"status": "enrichment_complete", "phase": "analysis"})

    return {"results": intel}

# Consumer
for chunk in graph.stream(
    state,
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "custom":
        progress = chunk["data"]
        print(f"Progress: {progress['status']} - {progress.get('phase', '')}")
    elif chunk["type"] == "updates":
        print(f"Node complete: {list(chunk['data'].keys())}")
```

---

## `stream_mode="debug"` — Full Execution Trace

Low-level execution events including checkpoint data:

```python
for chunk in graph.stream(
    state,
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        event = chunk["data"]
        print(f"Event type: {event['type']}")
        # event types: "checkpoint", "task", "task_result"
```

---

## Combining Multiple Stream Modes

```python
for chunk in graph.stream(
    state,
    stream_mode=["updates", "custom", "messages"],
    version="v2",
):
    event_type = chunk["type"]

    if event_type == "messages":
        token, metadata = chunk["data"]
        if token.content:
            yield f"data: {json.dumps({'type':'token','text':token.content})}\n\n"

    elif event_type == "custom":
        yield f"data: {json.dumps({'type':'progress','data':chunk['data']})}\n\n"

    elif event_type == "updates":
        node = list(chunk["data"].keys())[0]
        yield f"data: {json.dumps({'type':'node_done','node':node})}\n\n"
```

---

## Streaming in FastAPI (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.types import Command
import json

app = FastAPI()

@app.post("/agent/stream")
async def agent_stream(request: AgentRequest):
    async def generate():
        config = {"configurable": {"thread_id": request.thread_id}}
        input_data = {"messages": [("human", request.message)]}

        async for chunk in graph.astream(
            input_data,
            config=config,
            stream_mode=["messages", "custom", "updates"],
            version="v2",
        ):
            event_type = chunk["type"]

            if event_type == "messages":
                token, metadata = chunk["data"]
                node = metadata.get("langgraph_node", "")
                if token.content and node in ("llm", "responder"):
                    yield f"data: {json.dumps({'type': 'token', 'content': token.content})}\n\n"

            elif event_type == "custom":
                yield f"data: {json.dumps({'type': 'status', 'data': chunk['data']})}\n\n"

            elif event_type == "updates":
                for node_name in chunk["data"]:
                    yield f"data: {json.dumps({'type': 'node_complete', 'node': node_name})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
```

---

## Streaming Through Subgraphs

By default, subgraph events are hidden from the parent stream. Enable with `subgraphs=True`:

```python
for chunk in graph.stream(
    state,
    stream_mode="updates",
    subgraphs=True,  # Include events from subgraphs
    version="v2",
):
    if chunk["type"] == "updates":
        namespace = chunk.get("ns", [])  # e.g., ["parent", "child_subgraph"]
        print(f"Namespace: {namespace}")
        print(f"Data: {chunk['data']}")
```

---

## Streaming with Interrupts

When a graph hits `interrupt()`, the stream yields the interrupt and stops:

```python
async for chunk in graph.astream(
    state,
    config=config,
    stream_mode="values",
    version="v2",
):
    if chunk["type"] == "values":
        # Process state updates
        pass

# After stream ends, check if interrupted
current_state = graph.get_state(config)
if current_state.next:
    # Graph is paused — surface interrupt to user
    interrupts = current_state.tasks[0].interrupts if current_state.tasks else []
    for interrupt_val in interrupts:
        print(f"Waiting for human input: {interrupt_val.value}")
```

---

## `astream_events` — LangChain-Compatible Event Stream

For compatibility with LangChain's event streaming interface:

```python
async for event in graph.astream_events(
    {"messages": [("human", "Hello")]},
    version="v2",
    config=config,
):
    event_type = event["event"]
    name = event["name"]

    if event_type == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        if token:
            print(token, end="", flush=True)

    elif event_type == "on_tool_start":
        print(f"\n[Tool] Starting: {name}")

    elif event_type == "on_tool_end":
        print(f"[Tool] Done: {name}")

    elif event_type == "on_chain_start" and name in graph.nodes:
        print(f"\n[Node] {name} starting")
```

---

## Streaming Performance Considerations

| Consideration | Recommendation |
|---|---|
| Mode for UI chat | `"messages"` — lowest latency to first token |
| Mode for API status | `"updates"` or `"custom"` — minimal payload |
| Mode for debugging | `"debug"` or `"values"` |
| Subgraph streaming | Only enable `subgraphs=True` if needed — adds overhead |
| Buffering | Ensure Nginx/load balancer has `X-Accel-Buffering: no` |
| SSE vs WebSocket | SSE for server→client unidirectional; WS for bidirectional |
| Backpressure | Implement `asyncio.Queue` between generator and consumer |

---

## Common Pitfalls

### 1. Not Flushing Tokens in Sync Streaming
```python
# ❌ Tokens buffer and appear in batches
for chunk in graph.stream(state, stream_mode="messages"):
    print(chunk[0].content, end="")

# ✅ Force flush
for chunk in graph.stream(state, stream_mode="messages"):
    print(chunk[0].content, end="", flush=True)
```

### 2. Incorrect `version` Parameter
```python
# ❌ Old API — returns tuples, confusing format
for chunk in graph.stream(state, stream_mode="values"):
    ...

# ✅ Use version="v2" — returns typed dicts with "type" and "data" fields
for chunk in graph.stream(state, stream_mode="values", version="v2"):
    if chunk["type"] == "values":
        process(chunk["data"])
```

---

## Related Topics
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Streaming with checkpointed graphs
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — Streaming with interrupts
- [`13-langgraph-platform.md`](./13-langgraph-platform.md) — LangGraph Server streaming API
