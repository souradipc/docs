# Subgraphs

## What Are Subgraphs?

A subgraph is a compiled `StateGraph` used as a **node** inside a parent graph. Subgraphs enable:

- **Modularity** — encapsulate complex logic into reusable components
- **State isolation** — subgraph can have private state keys not visible to parent
- **Independent checkpointing** — subgraph can have its own checkpointer
- **Deep hierarchies** — parent → child → grandchild graphs
- **Reusability** — same subgraph used as multiple nodes in one parent graph

---

## Two Ways to Use Subgraphs as Nodes

### Option A: Direct Reference (Shared State Keys)

When the subgraph's state has at least one key in common with the parent state, use it directly:

```python
from langgraph.graph import StateGraph, MessagesState, START, END

# Subgraph has "messages" key (shared with parent)
class SubgraphState(MessagesState):
    # Extra private key only visible inside subgraph
    search_results: list[str]

sub_builder = StateGraph(SubgraphState)
# ... add nodes and edges
sub_compiled = sub_builder.compile()

# Parent graph also uses MessagesState — they share "messages"
parent_builder = StateGraph(MessagesState)
parent_builder.add_node("sub", sub_compiled)  # Direct reference!
parent_builder.add_edge(START, "sub")
parent_builder.add_edge("sub", END)
parent_graph = parent_builder.compile()
```

LangGraph automatically:
- Passes shared keys from parent state → subgraph input
- Propagates shared key updates from subgraph → parent state
- Private subgraph keys never appear in parent state

### Option B: Wrapper Function (Different State Schemas)

When parent and subgraph have different state schemas, use a wrapper function that transforms state in both directions:

```python
class ParentState(TypedDict):
    query: str
    final_report: str

class SubgraphState(TypedDict):
    # No keys shared with parent
    search_query: str
    raw_results: list[str]
    structured_output: str

# Build and compile subgraph
subgraph = build_research_subgraph()

# Wrapper function handles the state transformation
def call_research_subgraph(state: ParentState) -> dict:
    # Transform parent → subgraph input
    subgraph_result = subgraph.invoke({
        "search_query": state["query"],
        "raw_results": [],
        "structured_output": "",
    })
    # Transform subgraph output → parent state update
    return {"final_report": subgraph_result["structured_output"]}

parent_builder = StateGraph(ParentState)
parent_builder.add_node("research", call_research_subgraph)
```

---

## Full Example: Shared State Keys

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# Subgraph state — has shared key "foo" and private key "bar"
class SubgraphState(TypedDict):
    foo: str   # Shared with parent
    bar: str   # Private to subgraph

def subgraph_step_1(state: SubgraphState) -> dict:
    return {"bar": "internal_data"}

def subgraph_step_2(state: SubgraphState) -> dict:
    # Combines shared + private state
    return {"foo": state["foo"] + state["bar"]}

subgraph = (
    StateGraph(SubgraphState)
    .add_node("step_1", subgraph_step_1)
    .add_node("step_2", subgraph_step_2)
    .add_edge(START, "step_1")
    .add_edge("step_1", "step_2")
    .compile()
)

# Parent state — only has "foo"
class ParentState(TypedDict):
    foo: str

def parent_pre_node(state: ParentState) -> dict:
    return {"foo": "hello-" + state["foo"]}

parent = (
    StateGraph(ParentState)
    .add_node("pre", parent_pre_node)
    .add_node("sub", subgraph)          # Direct subgraph reference
    .add_edge(START, "pre")
    .add_edge("pre", "sub")
    .add_edge("sub", END)
    .compile()
)

result = parent.invoke({"foo": "world"})
print(result["foo"])  # "hello-worldinternal_data"
# Note: "bar" never appears in parent result
```

---

## Full Example: Different State Schemas

```python
class ResearchSubgraphState(TypedDict):
    topic: str
    sources: list[str]
    findings: str

def fetch_sources(state: ResearchSubgraphState) -> dict:
    return {"sources": web_search(state["topic"])}

def synthesize(state: ResearchSubgraphState) -> dict:
    findings = llm.invoke(f"Synthesize these sources: {state['sources']}")
    return {"findings": findings.content}

research_subgraph = (
    StateGraph(ResearchSubgraphState)
    .add_node("fetch", fetch_sources)
    .add_node("synthesize", synthesize)
    .add_edge(START, "fetch")
    .add_edge("fetch", "synthesize")
    .compile()
)

# Parent state
class ReportState(TypedDict):
    question: str
    sections: list[str]  # Accumulate with operator.add

import operator
from typing import Annotated

class ReportState(TypedDict):
    question: str
    sections: Annotated[list[str], operator.add]

def research_section(state: ReportState) -> dict:
    """Wrapper: transforms parent state → subgraph state → back."""
    result = research_subgraph.invoke({
        "topic": state["question"],
        "sources": [],
        "findings": "",
    })
    return {"sections": [result["findings"]]}

parent = (
    StateGraph(ReportState)
    .add_node("research", research_section)
    .add_edge(START, "research")
    .add_edge("research", END)
    .compile()
)
```

---

## Subgraph Checkpointing

### Option A: Subgraph Inherits Parent's Checkpointer (Default)

The subgraph uses the parent's checkpointer. State is checkpointed as a single unit. Time travel re-executes the entire subgraph from scratch:

```python
# No checkpointer on subgraph — inherits from parent
subgraph = subgraph_builder.compile()

parent = parent_builder.compile(checkpointer=InMemorySaver())
# Parent checkpoints include subgraph state seamlessly
```

### Option B: Subgraph Has Its Own Checkpointer (Independent)

The subgraph maintains its own checkpoint history. Time travel can navigate inside the subgraph:

```python
from langgraph.checkpoint.memory import InMemorySaver

# Subgraph has its own checkpointer
subgraph = subgraph_builder.compile(checkpointer=InMemorySaver())

# Parent also has its own checkpointer
parent = parent_builder.compile(checkpointer=InMemorySaver())
# Both maintain independent checkpoint histories
```

---

## Streaming Subgraph Events

By default, parent stream doesn't include subgraph events. Use `subgraphs=True`:

```python
for chunk in parent.stream(
    state,
    stream_mode="updates",
    subgraphs=True,     # Include subgraph events
    version="v2",
):
    if chunk["type"] == "updates":
        namespace = chunk.get("ns", [])
        node_data = chunk["data"]

        if len(namespace) == 0:
            print(f"[Parent] {list(node_data.keys())}")
        elif len(namespace) == 1:
            print(f"[Subgraph:{namespace[0]}] {list(node_data.keys())}")
```

---

## Deep Hierarchies — Three Levels

```python
# Grandchild graph
class GrandchildState(TypedDict):
    detail: str

grandchild = (
    StateGraph(GrandchildState)
    .add_node("process", process_detail)
    .add_edge(START, "process")
    .compile()
)

# Child graph — wraps grandchild
class ChildState(TypedDict):
    item: str

def call_grandchild(state: ChildState) -> dict:
    result = grandchild.invoke({"detail": state["item"]})
    return {"item": result["detail"]}

child = (
    StateGraph(ChildState)
    .add_node("grandchild", call_grandchild)
    .add_edge(START, "grandchild")
    .compile()
)

# Parent graph — wraps child
class ParentState(TypedDict):
    task: str

def call_child(state: ParentState) -> dict:
    result = child.invoke({"item": state["task"]})
    return {"task": result["item"]}

parent = (
    StateGraph(ParentState)
    .add_node("child", call_child)
    .add_edge(START, "child")
    .compile()
)
```

---

## When to Use Subgraphs

| Use Case | Subgraph? | Alternative |
|---|---|---|
| Reusable agent module | ✅ Yes | Inline all logic |
| Private state isolation | ✅ Yes | Prefix field names |
| Independent checkpointing | ✅ Yes | Single graph |
| Simple linear logic | ❌ Overkill | Regular nodes |
| One-time-use node | ❌ Overkill | Inline function |
| Multi-agent system | ✅ Yes | Tool wrapping |

---

## Common Pitfalls

### 1. Subgraph State Leaking to Parent
```python
# ❌ Shared key with unintended state pollution
class SubgraphState(TypedDict):
    messages: list  # Shared — ALL subgraph messages appear in parent!
    status: str     # Also shared!

# ✅ Be explicit about which keys to share
class SubgraphState(TypedDict):
    messages: list  # Intentionally shared
    _internal_status: str  # Naming convention for internal keys
    # (Still technically accessible but signals intent)
```

### 2. Forgetting to Transform State in Wrapper Functions
```python
# ❌ Passing wrong keys to subgraph
def bad_wrapper(state: ParentState) -> dict:
    result = subgraph.invoke(state)  # Passes parent keys that subgraph doesn't understand
    return result  # Returns subgraph keys that parent doesn't understand

# ✅ Explicitly transform both ways
def good_wrapper(state: ParentState) -> dict:
    subgraph_input = {"subgraph_key": state["parent_key"]}
    subgraph_result = subgraph.invoke(subgraph_input)
    return {"parent_key": subgraph_result["subgraph_key"]}
```

---

## Related Topics
- [`07-multi-agent-systems.md`](./07-multi-agent-systems.md) — Subgraphs in multi-agent context
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Subgraph checkpointing
- [`06-streaming.md`](./06-streaming.md) — Streaming subgraph events
