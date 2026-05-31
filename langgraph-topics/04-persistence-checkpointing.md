# Persistence & Checkpointing

## Why Persistence Changes Everything

Without persistence, a LangGraph graph is just a fancy function call — stateless, amnesiac, and fragile. With a checkpointer:

- Every node execution creates a **checkpoint** (state snapshot)
- Workflows can **resume after crashes** from the last checkpoint
- Multi-turn conversations have **real memory**
- You can **travel back in time** to any prior state
- **Human-in-the-loop** becomes possible (graph pauses, waits, resumes)
- Parallel background work can be **monitored and inspected**

---

## Checkpointer Implementations

### `InMemorySaver` — Development

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Data lives in RAM — lost on process restart
# Suitable for: tests, prototypes, local development
```

### `AsyncSqliteSaver` — Single-Process Production

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    async with AsyncSqliteSaver.from_conn_string("agent_checkpoints.db") as checkpointer:
        await checkpointer.setup()  # Creates tables (idempotent)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "user-alice-1"}}
        result = await graph.ainvoke({"messages": [("human", "Hello")]}, config=config)
```

### `AsyncPostgresSaver` — Multi-Worker Production

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def lifespan(app):
    DB_URL = "postgresql://user:pass@localhost:5432/agents_db"
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
        await checkpointer.setup()  # Run once on startup
        app.state.graph = builder.compile(checkpointer=checkpointer)
        yield

# This checkpointer is safe for multi-process, multi-worker deployments
# Multiple FastAPI workers share the same PostgreSQL checkpoints
```

### Checkpointer Comparison

| Checkpointer | Persistence | Multi-process | Use Case |
|---|---|---|---|
| `InMemorySaver` | RAM only | ❌ | Dev, tests |
| `AsyncSqliteSaver` | SQLite file | ❌ single-proc | Small apps, single instance |
| `AsyncPostgresSaver` | PostgreSQL | ✅ | Production, multi-worker |
| Custom `BaseCheckpointSaver` | Any backend | Depends | Enterprise (Redis, DynamoDB, etc.) |

---

## Thread IDs — The Session Key

The `thread_id` is the primary key that identifies a conversation session. All checkpoints for the same thread are linked together in a chain.

```python
# Each unique thread_id is an independent conversation
alice_config = {"configurable": {"thread_id": "user-alice-conv-1"}}
bob_config   = {"configurable": {"thread_id": "user-bob-conv-1"}}

# Turn 1
graph.invoke({"messages": [("human", "My name is Alice")]}, config=alice_config)
graph.invoke({"messages": [("human", "My name is Bob")]},   config=bob_config)

# Turn 2 — each remembers their own history
r1 = graph.invoke({"messages": [("human", "What's my name?")]}, config=alice_config)
r2 = graph.invoke({"messages": [("human", "What's my name?")]}, config=bob_config)

print(r1["messages"][-1].content)  # "Your name is Alice."
print(r2["messages"][-1].content)  # "Your name is Bob."
```

### Thread ID Naming Strategy

```python
# Pattern: {entity_type}-{entity_id}-{conversation_id}
thread_id = f"user-{user.id}-conv-{conversation.id}"

# Multi-tenant pattern
thread_id = f"tenant-{tenant_id}-user-{user_id}-session-{session_id}"

# Background task pattern
thread_id = f"task-{task_type}-{task_id}"
```

---

## How Checkpointing Works Internally

After every **superstep** (all parallel nodes in a step complete), LangGraph:

1. Serializes the full state to a checkpoint
2. Assigns a unique `checkpoint_id` (UUID)
3. Stores it with metadata: timestamp, step number, `parent_checkpoint_id`
4. On next invocation, loads the latest checkpoint for the `thread_id`

```python
# Inspect checkpoint structure
state_snapshot = graph.get_state(config)
print(state_snapshot.values)           # Full state dict
print(state_snapshot.next)             # Next nodes to execute ([] if complete)
print(state_snapshot.config)           # Config including checkpoint_id
print(state_snapshot.metadata)         # Step number, source, timestamps
print(state_snapshot.created_at)       # ISO datetime string
print(state_snapshot.parent_config)    # Reference to previous checkpoint
```

---

## Inspecting and Querying State

### Get Current State

```python
config = {"configurable": {"thread_id": "conv-1"}}
snapshot = graph.get_state(config)

# Is the graph done or paused?
if snapshot.next:
    print(f"Paused at: {snapshot.next}")   # e.g., ("human_review",)
else:
    print("Graph completed")

# Access state values
messages = snapshot.values["messages"]
```

### Get State at a Specific Checkpoint

```python
# Using specific checkpoint ID
specific_config = {
    "configurable": {
        "thread_id": "conv-1",
        "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c",
    }
}
old_snapshot = graph.get_state(specific_config)
```

### Get Full State History

```python
config = {"configurable": {"thread_id": "conv-1"}}

# Returns iterator of StateSnapshot, most recent first
history = list(graph.get_state_history(config))

print(f"Total checkpoints: {len(history)}")

for snapshot in history:
    print(f"  [{snapshot.metadata.get('step', '?')}] "
          f"{snapshot.created_at} → next: {snapshot.next}")
```

---

## Time Travel — Replaying and Forking

One of LangGraph's most powerful features: travel to any prior state and re-execute from there.

### Replay (Re-execute from a Prior Point)

```python
history = list(graph.get_state_history(config))

# Find the checkpoint just before the "dangerous_action" node
target = next(s for s in history if s.next == ("dangerous_action",))

# Replay from that point — pauses at interrupt again
result = graph.invoke(None, target.config)
```

### Fork (Create a New Branch from Prior State)

Forking creates a new checkpoint with modified state, branching the history:

```python
# Get history
history = list(graph.get_state_history(config))
old_snapshot = history[3]  # Choose a past checkpoint

# Modify state and create fork
fork_config = graph.update_state(
    old_snapshot.config,
    {"messages": [("human", "Actually, ignore that. Let's try something different.")]},
)

# Run from the fork — creates a new branch
result = graph.invoke(None, fork_config)

# The original branch is untouched — fork creates a new thread_id descendant
```

---

## Manually Updating State

```python
config = {"configurable": {"thread_id": "conv-1"}}

# Update state while graph is paused (e.g., after interrupt)
graph.update_state(
    config,
    {
        "messages": [("human", "I've reviewed this and it looks good.")],
        "approved": True,
    },
    as_node="human_reviewer",  # Attribute this update to this node name
)

# Resume from the updated state
result = graph.invoke(None, config)
```

---

## Checkpoint Lifecycle Management

```python
from langsmith import Client

# For production: implement TTL / cleanup of old threads
# LangGraph doesn't have built-in TTL — implement via direct DB queries

# PostgreSQL example: delete threads older than 30 days
# DELETE FROM checkpoints
# WHERE thread_id IN (
#     SELECT DISTINCT thread_id FROM checkpoints
#     WHERE created_at < NOW() - INTERVAL '30 days'
# );

# Or delete a specific thread entirely
# DELETE FROM checkpoints WHERE thread_id = 'user-alice-conv-1';
```

---

## Custom Checkpointer

Implement `BaseCheckpointSaver` to use any backend:

```python
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from typing import Iterator, Optional

class RedisCheckpointSaver(BaseCheckpointSaver):
    def __init__(self, redis_client):
        self.redis = redis_client

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        data = self.redis.get(f"checkpoint:{thread_id}:latest")
        if data:
            return deserialize(data)
        return None

    def put(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata) -> dict:
        thread_id = config["configurable"]["thread_id"]
        self.redis.set(f"checkpoint:{thread_id}:latest", serialize(checkpoint))
        return {**config, "configurable": {**config["configurable"], "checkpoint_id": checkpoint["id"]}}

    def list(self, config: dict, ...) -> Iterator[CheckpointTuple]:
        # Implement list for time travel support
        ...
```

---

## Production Checkpointing Patterns

### FastAPI Integration with PostgreSQL

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer: AsyncPostgresSaver = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as cp:
        await cp.setup()
        checkpointer = cp
        app.state.graph = build_agent_graph(cp)
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/chat/{thread_id}")
async def chat(thread_id: str, request: ChatRequest):
    config = {"configurable": {"thread_id": thread_id}}
    result = await app.state.graph.ainvoke(
        {"messages": [("human", request.message)]},
        config=config,
    )
    return {"reply": result["messages"][-1].content}
```

---

## Common Pitfalls

### 1. Not Calling `setup()` on DB Checkpointers
```python
# ❌ Tables don't exist → errors on first run
checkpointer = AsyncPostgresSaver.from_conn_string(DB_URL)

# ✅ setup() creates tables — idempotent, safe to call every startup
async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
    await checkpointer.setup()
```

### 2. Sharing Thread IDs Across Users
```python
# ❌ Security violation — users share state
config = {"configurable": {"thread_id": "global"}}

# ✅ Unique thread per user per session
config = {"configurable": {"thread_id": f"user-{user_id}-{session_id}"}}
```

### 3. Compiling Without Checkpointer Then Adding It
```python
# ❌ Checkpointer must be set at compile time
graph = builder.compile()
# Can't add checkpointer later

# ✅ Pass checkpointer at compile time
graph = builder.compile(checkpointer=my_checkpointer)
```

---

## Related Topics
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — Interrupts require checkpointing
- [`06-streaming.md`](./06-streaming.md) — Streaming with checkpointed graphs
- [`10-memory-store.md`](./10-memory-store.md) — Cross-thread memory (different from checkpoints)
