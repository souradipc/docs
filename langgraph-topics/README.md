# LangGraph Knowledge Base

Production-focused reference for building LangGraph applications. Assumes familiarity with LangChain, Python async, and agent/RAG patterns.

---

## Topic Index

| # | File | Core Concepts |
|---|---|---|
| 00 | [Overview](./00-overview.md) | What LangGraph is, why vs LCEL, execution model, key primitives |
| 01 | [StateGraph Fundamentals](./01-state-graph-fundamentals.md) | TypedDict, Pydantic, MessagesState, reducers, nodes, edges, compile |
| 02 | [State Design Patterns](./02-state-design-patterns.md) | Schema options, reducer patterns, input/output schemas, private state, context |
| 03 | [Nodes, Edges & Routing](./03-nodes-edges-routing.md) | Node contract, conditional edges, `Command` object, routing patterns |
| 04 | [Persistence & Checkpointing](./04-persistence-checkpointing.md) | Checkpointers, thread IDs, time travel, `update_state`, custom checkpointer |
| 05 | [Human-in-the-Loop](./05-human-in-the-loop.md) | `interrupt()`, `Command(resume=)`, approval workflows, breakpoints |
| 06 | [Streaming](./06-streaming.md) | 5 stream modes, `StreamWriter`, FastAPI SSE, subgraph streaming |
| 07 | [Multi-Agent Systems](./07-multi-agent-systems.md) | Supervisor, handoff/swarm, subagents, namespace isolation |
| 08 | [Subgraphs](./08-subgraphs.md) | Direct reference, wrapper function, checkpointing, streaming |
| 09 | [Map-Reduce & Parallelism](./09-map-reduce-parallelism.md) | `Send` API, static fan-out, `operator.add` reducers, batching |
| 10 | [Memory Store](./10-memory-store.md) | `InMemoryStore`, cross-thread memory, `Runtime`, namespaces, semantic search |
| 11 | [Functional API](./11-functional-api.md) | `@entrypoint`, `@task`, durable execution, `RetryPolicy`, `TimeoutPolicy` |
| 12 | [Production Error Handling](./12-production-error-handling.md) | `RetryPolicy`, `error_handler`, `NodeError`, saga, circuit breaker |
| 13 | [LangGraph Platform](./13-langgraph-platform.md) | Server, `langgraph.json`, REST API, SDK, background runs, crons, double-texting |
| 14 | [Prebuilt Components](./14-prebuilt-components.md) | `create_react_agent`, `ToolNode`, `tools_condition`, `MessagesState`, `InjectedToolArg` |

---

## Decision Guide

### Which API to use?

```
Simple ReAct agent (LLM + tools)?
  └─ create_react_agent()

Sequential pipeline / ETL / batch job?
  └─ Functional API (@entrypoint + @task)

Complex routing / cycles / multi-agent?
  └─ StateGraph

Both cycles AND durable task execution?
  └─ StateGraph with @task inside nodes
```

### Which state schema?

```
Chat history only?                → MessagesState
Chat + custom fields?             → class MyState(MessagesState): ...
No messages, pure data?           → TypedDict
Need field validation at runtime? → Pydantic BaseModel
```

### Which checkpointer?

```
Development / testing?            → MemorySaver (in-process)
Production single-node?           → AsyncSqliteSaver
Production multi-node / cloud?    → AsyncPostgresSaver
LangGraph Platform?               → Built-in (auto-managed)
```

### Cross-thread memory vs checkpointer?

```
Per-conversation history?         → Checkpointer (thread-scoped)
User profile / long-term facts?   → Store (cross-thread)
Request-scoped read-only data?    → context_schema + context=
```

---

## Quick-Start Patterns

### Minimal Graph
```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def agent(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

graph = (
    StateGraph(MessagesState)
    .add_node("agent", agent)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)
result = graph.invoke({"messages": [("user", "Hello!")]})
```

### ReAct Agent (one-liner)
```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model=llm, tools=[search, calculator], checkpointer=checkpointer)
```

### HITL Approval Workflow
```python
from langgraph.types import interrupt, Command

def review_step(state):
    decision = interrupt({"data": state["result"], "message": "Approve?"})
    return {"approved": decision == "approve"}

# Resume after human action
graph.invoke(Command(resume="approve"), config=config)
```

### Parallel Fan-Out (Send API)
```python
from langgraph.types import Send

def spawn_workers(state) -> list[Send]:
    return [Send("worker", {"item": x}) for x in state["items"]]

builder.add_conditional_edges("split", spawn_workers, ["worker"])
```

### FastAPI Streaming Endpoint
```python
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(req: ChatRequest):
    async def generate():
        async for chunk in graph.astream(
            {"messages": [HumanMessage(req.message)]},
            config={"configurable": {"thread_id": req.thread_id}},
            stream_mode="messages",
        ):
            if isinstance(chunk[0], AIMessageChunk):
                yield f"data: {chunk[0].content}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Key Imports Cheat Sheet

```python
# Core graph
from langgraph.graph import StateGraph, START, END, MessagesState

# Types
from langgraph.types import (
    Send,           # Dynamic fan-out
    interrupt,      # HITL pause
    Command,        # HITL resume + routing
    RetryPolicy,    # Node/task retry config
    TimeoutPolicy,  # Task timeout config
    StreamWriter,   # Custom stream events
    Runtime,        # Store + config injection in nodes
)

# Persistence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Store
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore

# Prebuilt
from langgraph.prebuilt import (
    create_react_agent,
    ToolNode,
    tools_condition,
    InjectedToolArg,
)

# Functional API
from langgraph.func import entrypoint, task

# Errors
from langgraph.errors import NodeError, GraphRecursionError
```

---

## Common Gotchas

| Issue | Cause | Fix |
|---|---|---|
| Parallel writes overwrite each other | No reducer on shared state key | Add `Annotated[list, operator.add]` |
| Graph never reaches END | Missing edge from last node | Add `.add_edge("last_node", END)` |
| `interrupt()` does nothing | No checkpointer attached | Add `checkpointer=MemorySaver()` to `compile()` |
| `update_state` doesn't stick | Wrong `as_node` parameter | Pass `as_node="node_name"` to trigger correct reducer |
| Async node blocks event loop | Using sync I/O in async node | Use `httpx.AsyncClient`, `asyncpg`, etc. |
| Recursion limit hit | Cycle without exit condition | Add exit condition or increase `recursion_limit` in config |
| Store search returns nothing | No embedding index configured | Add `index={"embed": ..., "dims": ...}` to `InMemoryStore` |
| `Runtime` not injected | Wrong parameter position | `Runtime` must be the **second** parameter |

---

## Related Knowledge Bases
- [`../fastapi-topics/`](../fastapi-topics/) — FastAPI production patterns
- [`../langchain-topics/`](../langchain-topics/) — LangChain chains, retrievers, tools
