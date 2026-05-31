# StateGraph Fundamentals

## The Three Building Blocks

Every LangGraph application is composed of three things:

```
State ──────── What data is shared
Nodes ──────── What work is done
Edges ──────── What runs next
```

---

## Defining State

State is a `TypedDict` (or Pydantic model) that defines the shared data schema for the graph. Every node receives the full state as input and returns a partial update.

### TypedDict State (Most Common)

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    # Messages with add_messages reducer (appends, doesn't replace)
    messages: Annotated[list[BaseMessage], add_messages]

    # Simple values — replaced on each update
    user_id: str
    intent: Optional[str]
    error_count: int
    approved: bool
```

### Pydantic State (With Validation)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class WorkflowState(BaseModel):
    query: str
    results: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    attempts: int = 0
    final_answer: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
```

### MessagesState — Prebuilt for Chat

```python
from langgraph.graph import MessagesState

# Equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# Use this for any chat-based agent
```

---

## Reducers — Controlling State Updates

By default, node updates **replace** the state field. Reducers change that behavior.

### Default (Replace)
```python
class State(TypedDict):
    status: str  # Replaced each time — last write wins
```

### `add_messages` Reducer (Append)
```python
from langgraph.graph.message import add_messages
from typing import Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # - Appends new messages instead of replacing
    # - Handles deduplication by message ID
    # - Supports RemoveMessage for deleting specific messages
```

### `operator.add` Reducer (List Append)
```python
import operator

class State(TypedDict):
    results: Annotated[list[str], operator.add]
    # Each node can append to results without clobbering others
    # Critical for parallel fan-in: all parallel branches write safely
```

### Custom Reducer
```python
def merge_dicts(left: dict, right: dict) -> dict:
    """Deep merge two dicts."""
    return {**left, **right}

class State(TypedDict):
    metadata: Annotated[dict, merge_dicts]
```

---

## Defining Nodes

Nodes are plain Python functions. They receive the full state and return a dict with only the fields they want to update:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Basic node — returns partial state update
def call_llm(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages will append this

# Node that updates multiple fields
def analyze_intent(state: State) -> dict:
    last_msg = state["messages"][-1].content
    if "help" in last_msg.lower():
        return {"intent": "support", "confidence": 0.9}
    return {"intent": "general", "confidence": 0.6}

# Async node
async def async_node(state: State) -> dict:
    result = await some_async_operation(state["query"])
    return {"results": [result]}
```

### Node Naming Conventions

```python
# Method 1: Pass function (node name = function name)
builder.add_node(call_llm)          # Name: "call_llm"
builder.add_node(analyze_intent)    # Name: "analyze_intent"

# Method 2: Explicit name
builder.add_node("llm", call_llm)
builder.add_node("intent", analyze_intent)
```

---

## Building the Graph

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

# Add nodes
builder.add_node("analyze", analyze_intent)
builder.add_node("call_llm", call_llm)
builder.add_node("respond", respond)

# Add edges
builder.add_edge(START, "analyze")          # Entry point
builder.add_edge("analyze", "call_llm")    # Fixed transition
builder.add_edge("call_llm", "respond")
builder.add_edge("respond", END)            # Exit

# Compile
graph = builder.compile()
```

### Method Chaining (Fluent API)

```python
graph = (
    StateGraph(State)
    .add_node("analyze", analyze_intent)
    .add_node("call_llm", call_llm)
    .add_node("respond", respond)
    .add_edge(START, "analyze")
    .add_edge("analyze", "call_llm")
    .add_edge("call_llm", "respond")
    .add_edge("respond", END)
    .compile()
)
```

---

## Conditional Edges — Dynamic Routing

```python
from typing import Literal

def route_after_llm(state: State) -> Literal["tools", "end"]:
    """Decide what to do after the LLM responds."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

builder.add_conditional_edges(
    "call_llm",         # Source node
    route_after_llm,    # Router function — returns node name
    # Optional explicit path map (useful for type checking)
    {"tools": "execute_tools", "end": END},
)
```

---

## Compiling the Graph

Compilation validates the graph structure and returns an executable `CompiledStateGraph`:

```python
from langgraph.checkpoint.memory import InMemorySaver

# Basic compile
graph = builder.compile()

# With checkpointing (required for: memory, HITL, time travel)
graph = builder.compile(checkpointer=InMemorySaver())

# With interrupt points (HITL)
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["dangerous_node"],
    interrupt_after=["review_node"],
)
```

---

## Running the Graph

```python
# Stateless invocation (no memory between calls)
result = graph.invoke({"messages": [("human", "Hello")]})
print(result["messages"][-1].content)

# Stateful invocation (requires checkpointer)
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke({"messages": [("human", "Hello")]}, config=config)

# Async invocation
result = await graph.ainvoke({"messages": [("human", "Hello")]}, config=config)

# Streaming
for event in graph.stream({"messages": [("human", "Hello")]}, config=config):
    print(event)
```

---

## Input/Output Schemas

By default, input and output schemas match the state schema. Override for cleaner APIs:

```python
from typing import TypedDict

class InputSchema(TypedDict):
    user_message: str    # Only expose this to callers

class OutputSchema(TypedDict):
    reply: str           # Only return this from graph
    sources: list[str]

class InternalState(InputSchema, OutputSchema):
    # Internal-only fields
    retrieved_docs: list
    llm_response: str
    intent: str

builder = StateGraph(InternalState, input=InputSchema, output=OutputSchema)
```

---

## Visualizing the Graph

```python
# Mermaid diagram (requires graphviz or browser)
print(graph.get_graph().draw_mermaid())

# PNG image (requires graphviz)
img_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(img_bytes)

# In Jupyter notebooks
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))

# Include subgraphs
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```

---

## The `START` and `END` Sentinels

```python
from langgraph.graph import START, END

# START: virtual entry point — not a real node, just the input boundary
# END: virtual exit point — signals graph completion
# A node can have multiple edges to END (any branch can terminate the graph)

# Multiple entry paths (unusual but valid)
builder.add_edge(START, "node_a")
builder.add_edge(START, "node_b")  # Both start in parallel
```

---

## Full Working Example

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o").bind_tools([multiply])

def call_llm(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}

graph = (
    StateGraph(State)
    .add_node("llm", call_llm)
    .add_node("tools", ToolNode([multiply]))
    .add_edge(START, "llm")
    .add_conditional_edges("llm", tools_condition)
    .add_edge("tools", "llm")
    .compile()
)

result = graph.invoke({"messages": [("human", "What is 6 * 7?")]})
print(result["messages"][-1].content)  # "42"
```

---

## Common Pitfalls

### 1. Forgetting the `add_messages` Reducer
```python
# ❌ Messages get replaced → conversation history lost
class State(TypedDict):
    messages: list[BaseMessage]  # No reducer!

# ✅
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 2. Node Returns Full State Instead of Partial Update
```python
# ❌ This replaces the entire state with only {"result": "..."}
def my_node(state: State) -> State:
    return State(result="done")  # All other fields wiped!

# ✅ Return only the fields you're updating
def my_node(state: State) -> dict:
    return {"result": "done"}
```

### 3. Compiling Without Checkpointer When You Need Memory
```python
# ❌ No checkpointer → no state persistence → no multi-turn memory
graph = builder.compile()

# ✅ Always use checkpointer for stateful workflows
graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Related Topics
- [`02-state-design-patterns.md`](./02-state-design-patterns.md) — Advanced state schemas
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — Routing and Command
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Checkpointers
