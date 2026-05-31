# Memory Store & Cross-Thread Memory

## Checkpoints vs Store — Critical Distinction

LangGraph has **two independent persistence systems**:

| System | What it stores | Scope | Access |
|---|---|---|---|
| **Checkpointer** | Conversation state per thread | Per-thread only | `graph.get_state(config)` |
| **Store** | Long-term, cross-thread memory | Across all threads | `store.get/put/search` |

Think of it this way:
- Checkpointer = conversation history for one session
- Store = user profile, preferences, learned facts — persisted forever across all sessions

---

## InMemoryStore (Development)

```python
from langgraph.store.memory import InMemoryStore

# Basic (no semantic search)
store = InMemoryStore()

# With semantic search support
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["text", "$"],  # which fields to embed ("$" = full doc)
    }
)
```

---

## Namespace Convention

Store items live at a `(namespace_tuple, key)` address:

```python
# Namespace is a tuple of strings — typically (user_id, category)
namespace = ("user_123", "memories")
key = "preferred_language"
value = {"text": "User prefers Python over JavaScript"}

store.put(namespace, key, value)

# Retrieve by exact key
item = store.get(namespace, key)
print(item.value)   # {"text": "User prefers Python over JavaScript"}
print(item.key)     # "preferred_language"

# List all items in namespace
items = store.search(namespace)

# Semantic search (requires embedding index)
results = store.search(namespace, query="what language does the user prefer?", limit=3)
for result in results:
    print(result.key, result.value, result.score)  # score = similarity

# Delete
store.delete(namespace, key)
```

---

## Accessing Store Inside Nodes — `Runtime` Object

The `Runtime` object injects the store + config into any node that requests it:

```python
from langgraph.types import Runtime

# Declare Runtime as the SECOND parameter type annotation
async def chat(state: AgentState, runtime: Runtime) -> dict:
    user_id = runtime.config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    # Retrieve relevant memories
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=5,
    )
    memory_context = "\n".join(
        f"- {m.value['text']}" for m in memories
    )

    # Store new information after this turn
    await runtime.store.aput(
        namespace,
        f"memory_{datetime.now().isoformat()}",
        {"text": f"User asked about: {state['messages'][-1].content}"}
    )

    response = await llm.ainvoke(
        f"Memories:\n{memory_context}\n\nChat history: {state['messages']}"
    )
    return {"messages": [response]}
```

**Key rules for `Runtime`:**
- Must be the **second parameter** with type annotation `Runtime`
- Must use `async` functions (use `asearch`, `aput`, etc.)
- Config values passed at `invoke` time are accessible via `runtime.config`

---

## Passing Store to the Graph

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(...)
checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,             # Pass store here
)

# Store is automatically available to all nodes that request Runtime
result = await graph.ainvoke(
    {"messages": [HumanMessage("Hello")]},
    config={
        "configurable": {
            "thread_id": "thread-abc",
            "user_id": "user-123",      # Custom config accessible via runtime.config
        }
    }
)
```

---

## Production Store — BaseStore

For production, implement `BaseStore` or use LangGraph Platform's built-in store:

```python
from langgraph.store.base import BaseStore
from langgraph.store.postgres import AsyncPostgresStore  # LangGraph Platform

# Self-hosted Postgres store
async with AsyncPostgresStore.from_conn_string(
    "postgresql://user:pass@host/db"
) as store:
    await store.setup()  # Create tables
    # Use same API as InMemoryStore
    await store.aput(("user_123", "prefs"), "theme", {"value": "dark"})
    result = await store.aget(("user_123", "prefs"), "theme")
```

---

## Context Schema — Read-Only Runtime Context

For values that should be available to all nodes but are **not part of the state** (and not persisted), use `context_schema`:

```python
from pydantic import BaseModel

class AppContext(BaseModel):
    user_id: str
    tenant_id: str
    permissions: list[str]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Register context schema at graph build time
graph = StateGraph(State, context_schema=AppContext).compile()

# Pass context at invoke time
result = await graph.ainvoke(
    {"messages": [HumanMessage("Hello")]},
    context=AppContext(
        user_id="user-123",
        tenant_id="tenant-456",
        permissions=["read", "write"],
    )
)
```

Accessing context inside nodes:

```python
from langgraph.types import Runtime

async def authorized_node(state: State, runtime: Runtime) -> dict:
    ctx: AppContext = runtime.context  # Strongly typed!
    if "admin" not in ctx.permissions:
        return {"messages": [AIMessage("Permission denied")]}
    # ... proceed with admin operation
```

**Context vs Store:**
- Context = request-scoped, read-only, not persisted
- Store = cross-thread, mutable, persisted

---

## Long-Term Memory Architecture Pattern

Multi-tier memory for production agents:

```python
from langgraph.store.memory import InMemoryStore
from langgraph.types import Runtime

class MemoryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

async def load_memory(state: MemoryState, runtime: Runtime) -> dict:
    """Load relevant memories before processing."""
    namespace = (state["user_id"], "facts")
    last_message = state["messages"][-1].content

    # Semantic search for relevant context
    relevant = await runtime.store.asearch(namespace, query=last_message, limit=10)

    memory_text = "\n".join(f"- {m.value['text']}" for m in relevant)
    # Inject memory as a system message for the LLM
    return {"messages": [SystemMessage(f"User context:\n{memory_text}")]}

async def call_model(state: MemoryState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

async def extract_and_save_memory(state: MemoryState, runtime: Runtime) -> dict:
    """Extract facts from the conversation and save them."""
    last_exchange = state["messages"][-2:]  # Question + Answer
    extract_response = await llm.ainvoke(
        f"Extract any factual claims about the user from this conversation: {last_exchange}"
        f"\nReturn as JSON list of strings. Empty list if none."
    )
    facts = json.loads(extract_response.content)

    namespace = (state["user_id"], "facts")
    for fact in facts:
        key = hashlib.md5(fact.encode()).hexdigest()[:8]
        await runtime.store.aput(namespace, key, {"text": fact, "timestamp": time.time()})

    return {}  # No state change needed

memory_graph = (
    StateGraph(MemoryState)
    .add_node("load_memory", load_memory)
    .add_node("llm", call_model)
    .add_node("save_memory", extract_and_save_memory)
    .add_edge(START, "load_memory")
    .add_edge("load_memory", "llm")
    .add_edge("llm", "save_memory")
    .add_edge("save_memory", END)
    .compile(checkpointer=checkpointer, store=store)
)
```

---

## Namespace Design Patterns

Namespaces are tuples — design them by access pattern:

```python
# Per-user, per-category
("user_123", "memories")
("user_123", "preferences")
("user_123", "documents")

# Per-tenant, per-user
("tenant_456", "user_123", "memories")

# Shared (global, all users)
("global", "kb_articles")
("global", "system_prompts")

# Per-conversation snapshot
("user_123", "conversation_snapshots")
```

---

## Common Pitfalls

### 1. Confusing Store with Checkpointer
```python
# ❌ Using checkpoint state for cross-user data
# All data is scoped to thread_id — won't work across sessions

# ✅ Use store with user_id namespace for cross-thread persistence
await runtime.store.aput(("user_123", "prefs"), "key", value)
```

### 2. Not Using Async API
```python
# ❌ store.get() is synchronous — blocks the event loop in async contexts
item = store.get(namespace, key)

# ✅ Use async variants inside async nodes
item = await runtime.store.aget(namespace, key)
```

### 3. Missing Embedding Config for Semantic Search
```python
# ❌ Search without index returns lexical/exact only
store = InMemoryStore()
results = store.search(namespace, query="user preferences")  # No semantic match

# ✅ Configure embedding index for semantic search
store = InMemoryStore(index={"embed": init_embeddings("openai:text-embedding-3-small"), "dims": 1536})
```

---

## Related Topics
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Per-thread checkpointing
- [`07-multi-agent-systems.md`](./07-multi-agent-systems.md) — Shared memory across agents
- [`13-langgraph-platform.md`](./13-langgraph-platform.md) — Managed store in LangGraph Platform
