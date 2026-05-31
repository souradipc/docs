# LangGraph — Stateful Agent Graphs

## What Is LangGraph?

LangGraph is a library for building **stateful, multi-step AI applications** as directed graphs. It extends LangChain with:

- **Explicit state management** — State is a typed schema, not implicit variables
- **Cycles and loops** — Agents can loop, retry, and branch dynamically
- **Persistence** — State is checkpointed at every step
- **Human-in-the-loop** — Pause execution and wait for human approval
- **Streaming** — Stream tokens and intermediate steps

LangGraph is the recommended way to build agents in LangChain ecosystem.

---

## Core Concepts

| Concept | Description |
|---|---|
| **State** | A `TypedDict` or Pydantic model defining all shared data |
| **Node** | A Python function or Runnable that transforms the state |
| **Edge** | A directed connection from one node to another |
| **Conditional Edge** | A function that chooses the next node based on state |
| **Graph** | `StateGraph` — the composition of nodes + edges |
| **Checkpointer** | Persists state at each step for recovery + human-in-loop |

---

## Minimal Graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

# 1. Define state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages is a reducer: appends instead of replacing

# 2. Create graph builder
builder = StateGraph(State)

# 3. Define nodes
llm = ChatOpenAI(model="gpt-4o")

def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Add nodes
builder.add_node("chatbot", chatbot)

# 5. Add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 6. Compile
graph = builder.compile()

# 7. Run
result = graph.invoke({"messages": [("human", "Hello!")]})
print(result["messages"][-1].content)
```

---

## State — The Central Data Structure

State is passed through every node and accumulated:

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Messages accumulate (not replaced) due to add_messages reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Simple values are replaced on each update
    current_intent: str
    confidence_score: float
    retrieved_docs: list[dict]
    error_count: int

    # Optional fields
    summary: Optional[str]
    user_id: Optional[str]
```

### Custom Reducers

```python
from typing import Annotated

def merge_lists(existing: list, new: list) -> list:
    """Merge two lists, deduplicating."""
    return list(set(existing) | set(new))

class State(TypedDict):
    # Use operator.add to concatenate lists
    import operator
    items: Annotated[list[str], operator.add]

    # Custom merge function
    unique_items: Annotated[list[str], merge_lists]
```

---

## Conditional Edges — Dynamic Routing

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Router function — returns node name as string
def route_after_llm(state: State) -> str:
    """Route based on whether the LLM wants to call tools."""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return END

# Build graph with conditional routing
builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("tools", ToolNode(tools))  # Prebuilt tool execution node

builder.add_edge(START, "llm")
builder.add_conditional_edges(
    "llm",           # Source node
    route_after_llm, # Routing function
    # Optional: map return values to node names
    # {"tools": "tools", END: END}
)
builder.add_edge("tools", "llm")  # After tools, go back to LLM

graph = builder.compile()
```

---

## Prebuilt Nodes and Components

```python
from langgraph.prebuilt import ToolNode, tools_condition

tools = [search_tool, calculator_tool]

# ToolNode executes all tool calls in the last AIMessage
tool_node = ToolNode(tools)

# tools_condition routes to "tools" if tool_calls, else END
builder.add_conditional_edges("llm", tools_condition)
```

---

## Full ReAct Agent from Scratch

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

tools = [get_weather]

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def call_llm(state: State) -> dict:
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(State)
builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")

graph = builder.compile()

result = graph.invoke({"messages": [("human", "What's the weather in Paris?")]})
print(result["messages"][-1].content)
```

---

## Subgraphs — Modular Composition

```python
from langgraph.graph import StateGraph

# Define sub-graph (inner state)
class InnerState(TypedDict):
    query: str
    results: list[str]

inner_builder = StateGraph(InnerState)
# ... add inner nodes and edges
inner_graph = inner_builder.compile()

# Outer graph uses the inner graph as a node
class OuterState(TypedDict):
    messages: Annotated[list, add_messages]
    inner_results: list[str]

outer_builder = StateGraph(OuterState)

def invoke_inner(state: OuterState) -> dict:
    inner_result = inner_graph.invoke({"query": state["messages"][-1].content})
    return {"inner_results": inner_result["results"]}

outer_builder.add_node("inner", invoke_inner)
outer_builder.add_edge(START, "inner")
outer_builder.add_edge("inner", END)

outer_graph = outer_builder.compile()
```

---

## Streaming Graph Execution

```python
# Stream mode "values" — full state at each step
for state in graph.stream(
    {"messages": [("human", "What's the weather?")]},
    stream_mode="values",
):
    last_message = state["messages"][-1]
    print(f"[{type(last_message).__name__}]: {last_message.content}")

# Stream mode "updates" — only the changes at each step
for node_name, updates in graph.stream(
    {"messages": [("human", "What's the weather?")]},
    stream_mode="updates",
):
    print(f"Node: {node_name}, Updates: {updates}")

# Stream tokens (LLM output word by word)
async for event in graph.astream_events(
    {"messages": [("human", "What's the weather?")]},
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

---

## State Snapshots and Visualization

```python
# Visualize the graph (requires graphviz)
from IPython.display import Image

img = graph.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(img)

# Get current state at any point
snapshot = graph.get_state(config)
print(snapshot.values)           # Current state
print(snapshot.next)             # Next nodes to execute
print(snapshot.created_at)       # When the snapshot was created

# List all checkpoints (history)
for checkpoint in graph.get_state_history(config):
    print(checkpoint.created_at, checkpoint.values["messages"][-1])
```

---

## Error Handling in Graphs

```python
# Retry logic in a node
import time

def resilient_node(state: State) -> dict:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff

# Error routing
def handle_error(state: State) -> str:
    if state.get("error_count", 0) > 3:
        return "give_up"
    return "retry"

builder.add_conditional_edges("risky_node", handle_error, {
    "retry": "risky_node",
    "give_up": "error_handler",
})
```

---

## Graph Compilation Options

```python
from langgraph.checkpoint.memory import MemorySaver

# With checkpointing
graph = builder.compile(checkpointer=MemorySaver())

# With interrupt points (human-in-the-loop)
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["dangerous_action"],  # Pause before this node
    interrupt_after=["review_step"],        # Pause after this node
)

# Recursion limit
result = graph.invoke(
    state,
    config={"recursion_limit": 50},  # Default is 25
)
```

---

## Common Pitfalls

### 1. Forgetting `add_messages` Reducer
```python
# ❌ Messages get replaced instead of accumulated
class State(TypedDict):
    messages: list[BaseMessage]  # No reducer!

# ✅ Use add_messages
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

### 2. Missing Edge Back to LLM from Tools
```python
# ❌ Tools runs but LLM never sees the result
builder.add_edge("tools", END)  # Agent stops after tool call!

# ✅ Route back to LLM
builder.add_edge("tools", "llm")  # LLM processes tool results
```

### 3. State Keys Not in TypedDict
```python
# ❌ Writing to undefined key → silently dropped
def my_node(state: State) -> dict:
    return {"undefined_key": "value"}  # Key not in State TypedDict → ignored!

# ✅ All state keys must be in the TypedDict definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    undefined_key: str  # Must be declared here
```

---

## Related Topics
- [`09-agents-tools.md`](./09-agents-tools.md) — Tools and agents basics
- [`12-langgraph-persistence-checkpointing.md`](./12-langgraph-persistence-checkpointing.md) — Persistence + human-in-the-loop
- [`13-callbacks-streaming.md`](./13-callbacks-streaming.md) — Streaming events from graphs
