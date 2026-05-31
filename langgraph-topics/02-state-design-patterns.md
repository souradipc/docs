# State Design Patterns

## State Is Your Contract

The state schema is the **most important design decision** in a LangGraph application. It determines:
- What data flows between nodes
- How concurrent updates are merged
- What gets persisted in checkpoints
- What callers see as input/output

Getting state design right upfront prevents painful refactors later.

---

## State Schema Options

### TypedDict (Default, Recommended)

```python
from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Chat history — append-only
    messages: Annotated[List[BaseMessage], add_messages]

    # Accumulated results — append across parallel branches
    search_results: Annotated[List[str], operator.add]

    # Simple scalars — last-write-wins
    current_intent: str
    user_id: str
    iteration: int

    # Optional fields
    error: Optional[str]
    final_answer: Optional[str]
```

### Pydantic BaseModel (With Validation + Default Values)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PipelineState(BaseModel):
    # Required field
    query: str

    # Fields with defaults
    retrieved_docs: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    attempts: int = 0

    # Validators
    @field_validator("confidence")
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, v))

# Register with graph — Pydantic models support reducers too
from typing import Annotated
import operator

class PipelineStateWithReducer(BaseModel):
    query: str
    results: Annotated[List[str], operator.add]
```

### Dataclass State

```python
from dataclasses import dataclass, field
from typing import List, Annotated
import operator

@dataclass
class DataclassState:
    query: str
    results: Annotated[List[str], operator.add] = field(default_factory=list)
    done: bool = False
```

---

## Reducer Patterns in Depth

### `add_messages` — The Standard for Chat

```python
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

# add_messages behavior:
# 1. Appends new messages to the list
# 2. Deduplicates by message ID (same ID = update, not duplicate)
# 3. Handles RemoveMessage for targeted deletion

# Remove a specific message by ID
def trim_old_messages(state: State) -> dict:
    # Remove all but the last 4 messages
    messages_to_remove = [
        RemoveMessage(id=m.id) for m in state["messages"][:-4]
    ]
    return {"messages": messages_to_remove}
```

### `operator.add` — Parallel-Safe List Accumulation

```python
import operator

class State(TypedDict):
    # ✅ Multiple parallel branches can all append safely
    results: Annotated[list[str], operator.add]

# Node A returns {"results": ["result_a"]}
# Node B returns {"results": ["result_b"]}
# After fan-in: state["results"] == ["result_a", "result_b"]
```

### `operator.or_` — Set Union for Deduplication

```python
import operator

class State(TypedDict):
    # Set semantics — no duplicates
    seen_urls: Annotated[set[str], operator.or_]
```

### Custom Reducer — Merge Dicts, Max, Min, etc.

```python
def last_wins(existing, new):
    """Explicit last-write-wins (same as default, but documented)."""
    return new

def max_reducer(existing: float, new: float) -> float:
    """Keep highest confidence score."""
    return max(existing, new)

def merge_metadata(existing: dict, new: dict) -> dict:
    """Merge dicts — new values overwrite existing on conflict."""
    return {**existing, **new}

class State(TypedDict):
    best_score: Annotated[float, max_reducer]
    metadata: Annotated[dict, merge_metadata]
```

---

## Input/Output Schemas — Clean Public APIs

Use separate input and output schemas to expose only what callers need:

```python
from typing import TypedDict

class UserInput(TypedDict):
    """What the caller provides."""
    question: str
    user_id: str

class AssistantOutput(TypedDict):
    """What the graph returns."""
    answer: str
    sources: list[str]
    confidence: float

class InternalState(UserInput, AssistantOutput):
    """Full internal state — everything above + internals."""
    messages: Annotated[list, add_messages]
    retrieved_docs: list
    search_queries: list[str]
    llm_reasoning: str

# Graph uses internal state, but input/output are clean
builder = StateGraph(
    InternalState,
    input=UserInput,
    output=AssistantOutput,
)

# Caller interface is minimal:
result = graph.invoke({"question": "What is phishing?", "user_id": "alice"})
# result = {"answer": "...", "sources": [...], "confidence": 0.95}
# Internal fields never exposed
```

---

## Private State — Node-Local Data

Sometimes a node needs data that shouldn't persist in the main state. Pass it via `Command` or use a subgraph with its own state:

```python
from typing import TypedDict
from langgraph.types import Command

# Approach: pass data through Command routing without storing in state
def intermediate_processor(state: MainState) -> Command:
    # Compute something locally
    local_result = expensive_computation(state["input"])

    # Write only the final aggregated result to state
    return Command(
        update={"output": summarize(local_result)},
        goto="next_node",
    )
```

---

## State Versioning and Migration

When you change the state schema in production, existing checkpoints may be incompatible:

```python
# Strategy 1: Add fields with defaults (backward compatible)
class StateV2(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    # New field with default — old checkpoints still work
    locale: str  # Will be missing from old checkpoints → handle in nodes

# Strategy 2: Custom deserialization
def call_model(state: State) -> dict:
    # Handle missing field gracefully
    locale = state.get("locale", "en-US")
    return {"response": format_response(state["messages"], locale)}
```

---

## Designing State for Parallel Execution

When nodes run in parallel, they all write to the same state. Use reducers to prevent clobbering:

```python
import operator
from typing import Annotated, TypedDict

class ParallelState(TypedDict):
    topic: str

    # ❌ WRONG: all parallel nodes overwrite each other
    result: str

    # ✅ CORRECT: parallel nodes safely append
    results: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]

def search_node_1(state: ParallelState) -> dict:
    return {"results": ["Result from source 1"]}

def search_node_2(state: ParallelState) -> dict:
    return {"results": ["Result from source 2"]}

# Both run in parallel, both safely append to results
builder.add_edge(START, "search_node_1")
builder.add_edge(START, "search_node_2")
builder.add_edge("search_node_1", "aggregator")
builder.add_edge("search_node_2", "aggregator")
```

---

## State Size and Performance

Checkpoints serialize the full state at every step. Large state → large checkpoints → slow I/O.

| Field Type | Checkpoint Size | Recommendation |
|---|---|---|
| Messages (10 msgs) | ~5-10 KB | Keep history trimmed |
| Messages (100 msgs) | ~100 KB | Use summarization |
| Large documents | MBs | Store in external DB, put ID in state |
| Embeddings | MBs | Never store in state — use store/vectorDB |
| Binary data | Large | Use reference IDs only |

```python
# ❌ Store full document text in state
class BadState(TypedDict):
    documents: list[str]  # 50 docs × 2KB = 100KB per checkpoint!

# ✅ Store document IDs — fetch from DB when needed
class GoodState(TypedDict):
    document_ids: list[str]  # Small — just IDs
    # Nodes fetch actual content from DB using these IDs
```

---

## Context Schema — Runtime Configuration

Pass runtime configuration (user ID, tenant ID, feature flags) that isn't part of the workflow state:

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class AppContext:
    user_id: str
    tenant_id: str
    feature_flags: dict

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Access context inside nodes via Runtime
async def call_model(state: State, runtime: Runtime[AppContext]) -> dict:
    user_id = runtime.context.user_id
    memories = await runtime.store.asearch(
        (user_id, "memories"),
        query=state["messages"][-1].content,
    )
    # ...

builder = StateGraph(State, context_schema=AppContext)

# Inject at runtime
result = graph.invoke(
    {"messages": [("human", "Hello")]},
    {"configurable": {"thread_id": "t1"}},
    context=AppContext(user_id="alice", tenant_id="acme", feature_flags={}),
)
```

---

## Common Design Mistakes

### 1. Monolithic State — Too Many Fields
```python
# ❌ One giant state for a complex multi-step pipeline
class GodState(TypedDict):
    # 30+ fields from different pipeline stages
    ...

# ✅ Use subgraphs with focused state schemas per module
# Parent state: high-level fields
# Subgraph state: detailed fields for that subgraph only
```

### 2. Missing Reducers for Concurrent Writes
```python
# ❌ Parallel branches overwrite each other (race condition)
class State(TypedDict):
    results: list[str]  # No reducer! Parallel writes corrupt this.

# ✅ Always use operator.add for fields written by parallel nodes
class State(TypedDict):
    results: Annotated[list[str], operator.add]
```

### 3. Storing Ephemeral Data in Persistent State
```python
# ❌ Storing full document content in checkpoint state
class State(TypedDict):
    full_document_text: str  # 50KB — checkpointed every step!

# ✅ Store reference, fetch from external storage
class State(TypedDict):
    document_id: str  # 36 bytes (UUID)
```

---

## Related Topics
- [`01-state-graph-fundamentals.md`](./01-state-graph-fundamentals.md) — StateGraph basics
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — Node return types
- [`09-map-reduce-parallelism.md`](./09-map-reduce-parallelism.md) — Parallel state accumulation
