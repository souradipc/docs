# LangGraph — Persistence, Checkpointing & Human-in-the-Loop

## Why Persistence Matters

Without persistence, every agent invocation starts from scratch. With checkpointing:

- **Multi-turn conversations** — Agent "remembers" previous messages
- **Fault tolerance** — Resume from last checkpoint after crashes
- **Human-in-the-loop** — Pause agent, get human approval, resume
- **Time travel** — Replay a conversation from any earlier state
- **Debugging** — Inspect exact state at every execution step

---

## Checkpointers

A checkpointer persists the graph state at every node execution step.

### `MemorySaver` — In-Memory (Dev)

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

checkpointer = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [("human", "My name is Alice.")]},
    config=config,
)

# Turn 2 — automatically loads previous state
result2 = agent.invoke(
    {"messages": [("human", "What's my name?")]},
    config=config,
)
print(result2["messages"][-1].content)  # "Your name is Alice."
```

### `AsyncSqliteSaver` — File-Based Persistence

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def run():
    async with AsyncSqliteSaver.from_conn_string("agent_state.db") as checkpointer:
        await checkpointer.setup()  # Creates tables on first run
        agent = create_react_agent(model, tools, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "session-1"}}
        result = await agent.ainvoke(
            {"messages": [("human", "Hello")]},
            config=config,
        )
```

### `AsyncPostgresSaver` — Production PostgreSQL

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def run():
    conn_string = "postgresql://user:pass@localhost:5432/agents_db"
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()  # Idempotent, run on startup
        agent = create_react_agent(model, tools, checkpointer=checkpointer)

        # Each thread_id is a separate conversation
        config = {"configurable": {"thread_id": f"user-{user_id}-conv-{conv_id}"}}
        result = await agent.ainvoke(
            {"messages": [("human", input_text)]},
            config=config,
        )
```

---

## Thread IDs — Conversation Isolation

The `thread_id` is the key that isolates different conversations:

```python
# Each user gets their own thread
alice_config = {"configurable": {"thread_id": "user-alice-1"}}
bob_config = {"configurable": {"thread_id": "user-bob-1"}}

agent.invoke({"messages": [("human", "My name is Alice.")]}, config=alice_config)
agent.invoke({"messages": [("human", "My name is Bob.")]}, config=bob_config)

# Alice's thread is separate from Bob's
result = agent.invoke({"messages": [("human", "What's my name?")]}, config=alice_config)
# "Your name is Alice." (not Bob)
```

---

## Inspecting State

```python
# Get current state of a thread
snapshot = agent.get_state(config)

print(snapshot.values)          # Full state dict
print(snapshot.values["messages"])  # All messages in this thread
print(snapshot.next)            # Next nodes to execute ([] if done)
print(snapshot.created_at)      # Timestamp

# Check if agent is done
if not snapshot.next:
    print("Agent completed")
else:
    print(f"Agent paused at: {snapshot.next}")
```

---

## State History — Time Travel

```python
# Get full history of a thread (all checkpoints)
history = list(agent.get_state_history(config))

print(f"Total steps: {len(history)}")

# Most recent first
for i, snapshot in enumerate(history):
    print(f"Step {i}: {snapshot.created_at}")
    print(f"  Messages: {len(snapshot.values.get('messages', []))}")
    print(f"  Next: {snapshot.next}")

# Replay from a specific checkpoint
old_snapshot = history[3]  # 4th step
old_config = old_snapshot.config  # Config with checkpoint_id

# Resume from that checkpoint
result = agent.invoke(
    {"messages": [("human", "Let's try a different approach")]},
    config=old_config,
)
```

---

## Human-in-the-Loop

### Interrupt Before a Node

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)
builder.add_node("research", research_node)
builder.add_node("take_action", action_node)  # Dangerous action
builder.add_edge(START, "research")
builder.add_edge("research", "take_action")
builder.add_edge("take_action", END)

# Pause BEFORE executing "take_action"
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["take_action"],
)

config = {"configurable": {"thread_id": "approval-flow-1"}}

# Agent runs, pauses before take_action
result = graph.invoke({"messages": [("human", "Delete all test data")]}, config=config)

# Check what the agent is about to do
state = graph.get_state(config)
print(f"Agent wants to: {state.next}")  # ["take_action"]
print(f"Planned action: {state.values['planned_action']}")  # Whatever the agent planned

# Human review — approve or reject
human_approved = True  # or False based on review

if human_approved:
    # Resume from where it paused
    result = graph.invoke(None, config=config)  # None = continue from pause
    print(result["messages"][-1].content)
else:
    # Override the state and redirect
    graph.update_state(
        config,
        {"messages": [("human", "Action rejected by human reviewer.")]},
    )
    result = graph.invoke(None, config=config)
```

### Interrupt After a Node

```python
# Pause AFTER "generate_plan" — review the plan before execution
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["generate_plan"],
)

# Agent generates plan, then pauses
result = graph.invoke(input, config=config)

# Review the generated plan
state = graph.get_state(config)
plan = state.values.get("plan")
print(f"Proposed plan:\n{plan}")

# Approve → resume
graph.invoke(None, config=config)

# OR modify the plan
graph.update_state(config, {"plan": "Modified plan here"})
graph.invoke(None, config=config)
```

---

## Updating State Manually

```python
# Directly modify graph state (e.g., after human review)
graph.update_state(
    config=config,
    values={
        "messages": [("human", "Actually, ignore the previous plan.")],
        "approved": False,
    },
    as_node="human_reviewer",  # Attribute update to this node name
)
```

---

## Multi-Turn Conversation Pattern (Production)

```python
from fastapi import FastAPI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

app = FastAPI()
agent = None

@app.on_event("startup")
async def startup():
    global agent
    async with AsyncPostgresSaver.from_conn_string(DB_URL) as checkpointer:
        await checkpointer.setup()
        agent = create_react_agent(model, tools, checkpointer=checkpointer)

@app.post("/chat/{user_id}/{session_id}")
async def chat(user_id: str, session_id: str, request: ChatRequest):
    config = {
        "configurable": {
            "thread_id": f"{user_id}:{session_id}",
        }
    }

    result = await agent.ainvoke(
        {"messages": [("human", request.message)]},
        config=config,
    )
    return {"reply": result["messages"][-1].content}

@app.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str):
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    return {
        "messages": [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in messages
        ]
    }
```

---

## Checkpointer Selection Guide

| Checkpointer | Persistence | Multi-process | Use Case |
|---|---|---|---|
| `MemorySaver` | RAM only | ❌ Single process | Development, testing |
| `AsyncSqliteSaver` | SQLite file | ❌ Single process | Small/medium apps, single instance |
| `AsyncPostgresSaver` | PostgreSQL | ✅ Multi-process | Production, multi-worker |
| Custom | Any DB | Depends | Enterprise custom backends |

---

## Common Pitfalls

### 1. Not Calling `setup()` on DB Checkpointers
```python
# ❌ Tables don't exist → database errors
checkpointer = AsyncPostgresSaver.from_conn_string(DB_URL)

# ✅ Always run setup on first start
await checkpointer.setup()  # Idempotent — safe to run on every startup
```

### 2. Reusing Thread IDs Across Users
```python
# ❌ All users share the same thread → memory leak + privacy violation
thread_id = "global-thread"

# ✅ Unique thread per user per session
thread_id = f"user-{user.id}-session-{session.id}"
```

### 3. Resuming Without Checking State First
```python
# ❌ Blindly resuming without knowing if agent is actually paused
graph.invoke(None, config)  # What if agent already finished?

# ✅ Check state before resuming
state = graph.get_state(config)
if state.next:  # Only resume if there's a pending step
    graph.invoke(None, config)
```

---

## Related Topics
- [`11-langgraph.md`](./11-langgraph.md) — Building graphs
- [`09-agents-tools.md`](./09-agents-tools.md) — Agent patterns
- [`10-memory-conversation-history.md`](./10-memory-conversation-history.md) — Conversation memory strategies
