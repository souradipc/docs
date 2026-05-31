# Nodes, Edges & Routing

## Node Execution Contract

A node is any Python callable that:
- **Receives**: the full current state
- **Returns**: a partial state update dict OR a `Command` object
- **Side effects**: allowed (API calls, DB writes, tool invocations)

```python
def my_node(state: State) -> dict:
    # Can call anything: LLMs, tools, databases, APIs
    result = do_work(state)
    # Return ONLY the fields you want to update
    return {"output": result}

# Async node
async def async_node(state: State) -> dict:
    result = await async_work(state)
    return {"output": result}
```

---

## Node Registration

```python
builder = StateGraph(State)

# Method 1: Function name becomes node name
builder.add_node(call_llm)          # Node name: "call_llm"

# Method 2: Explicit name
builder.add_node("llm", call_llm)   # Node name: "llm"

# Method 3: Compiled subgraph as node (shared state keys)
builder.add_node("research", research_subgraph)

# Method 4: With retry and error handling
from langgraph.types import RetryPolicy
from langgraph.errors import NodeError

builder.add_node(
    "llm_call",
    call_llm,
    retry_policy=RetryPolicy(max_attempts=3, retry_on=Exception),
    error_handler=handle_llm_error,
)
```

---

## Fixed Edges

Unconditional transition from one node to another:

```python
builder.add_edge(START, "first_node")       # Graph entry
builder.add_edge("first_node", "second")    # Always goes A → B
builder.add_edge("second", END)             # Graph exit

# Multiple outgoing fixed edges = parallel execution
builder.add_edge(START, "fetch_data")
builder.add_edge(START, "load_config")      # Both start at the same time
# Both must complete before any node that depends on them
```

---

## Conditional Edges

Dynamic routing based on the current state:

```python
from typing import Literal

def route_after_analysis(state: State) -> Literal["approve", "reject", "review"]:
    score = state["confidence"]
    if score > 0.9:
        return "approve"
    elif score < 0.3:
        return "reject"
    else:
        return "review"

builder.add_conditional_edges(
    "analyze",                    # Source node
    route_after_analysis,         # Router function
    # Optional: explicit name mapping (for clarity + type safety)
    {
        "approve": "approve_node",
        "reject": "reject_node",
        "review": "human_review",
    },
)
```

### Router with Multiple Destinations

```python
def route_to_specialist(state: State) -> str:
    intent = state["intent"]
    routing_table = {
        "billing": "billing_agent",
        "technical": "tech_support",
        "complaint": "escalation",
    }
    return routing_table.get(intent, "general_agent")

builder.add_conditional_edges("classify", route_to_specialist)
```

### `tools_condition` — The Prebuilt Tool Router

```python
from langgraph.prebuilt import tools_condition

# Returns "tools" if last message has tool_calls, else END
builder.add_conditional_edges("llm", tools_condition)
```

---

## The `Command` Object — Routing + Update Combined

`Command` lets a node simultaneously update state AND declare the next node. This replaces conditional edges for complex routing logic:

```python
from langgraph.types import Command
from typing import Literal

def classify_and_route(state: State) -> Command[Literal["billing", "tech", "general"]]:
    """Classify intent and route to appropriate specialist."""
    intent = llm_classify(state["messages"][-1].content)
    confidence = compute_confidence(intent)

    # BOTH: update state AND route — in one return
    return Command(
        update={
            "intent": intent,
            "confidence": confidence,
        },
        goto=intent,  # "billing", "tech", or "general"
    )

# With Command, you don't need add_conditional_edges
builder.add_node("classify", classify_and_route)
builder.add_edge(START, "classify")
# The Command inside the node handles routing — no conditional edges needed
```

### `Command` vs Conditional Edges — When to Use Which

| Scenario | Conditional Edge | Command |
|---|---|---|
| Routing only (no state update) | ✅ Preferred | Works but verbose |
| Routing + state update atomically | ❌ Can't do both | ✅ Required |
| Human-in-the-loop resume routing | ❌ Not applicable | ✅ Required |
| Static, predictable routes | ✅ More readable | Works |
| Dynamic routes based on computed values | Works | ✅ More natural |

---

## Command for Multi-Target Routing

Send execution to multiple nodes simultaneously:

```python
from langgraph.types import Command, Send

def fan_out_node(state: State) -> Command:
    # Route to multiple nodes in parallel using Send
    return Command(
        goto=[
            Send("search_web", {"query": state["query"]}),
            Send("search_db", {"query": state["query"]}),
            Send("search_docs", {"query": state["query"]}),
        ]
    )
```

---

## Routing Patterns

### Pattern 1: ReAct Agent Loop

```python
def should_continue(state: State) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "__end__"

builder.add_node("llm", call_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue)
builder.add_edge("tools", "llm")  # Loop back
```

### Pattern 2: Retry Loop with Limit

```python
def should_retry(state: State) -> Literal["retry", "fail", "success"]:
    if state.get("error") is None:
        return "success"
    elif state.get("attempts", 0) < 3:
        return "retry"
    else:
        return "fail"

def call_with_tracking(state: State) -> dict:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"error": str(e), "attempts": state.get("attempts", 0) + 1}

builder.add_node("call", call_with_tracking)
builder.add_node("success_handler", handle_success)
builder.add_node("fail_handler", handle_failure)
builder.add_edge(START, "call")
builder.add_conditional_edges("call", should_retry, {
    "retry": "call",              # Loop back
    "fail": "fail_handler",
    "success": "success_handler",
})
```

### Pattern 3: Sequential Pipeline with Validation Gates

```python
def validate_output(state: State) -> Literal["pass", "regenerate"]:
    """Check if output meets quality threshold."""
    if state["quality_score"] >= 0.8:
        return "pass"
    elif state["attempts"] < 3:
        return "regenerate"
    return "pass"  # Accept after max retries

builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)
builder.add_node("finalize", finalize_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "validate")
builder.add_conditional_edges("validate", validate_output, {
    "pass": "finalize",
    "regenerate": "generate",  # Loop back to regenerate
})
builder.add_edge("finalize", END)
```

### Pattern 4: Parallel Fan-Out with Conditional Fan-In

```python
from typing import Literal

def start_parallel(state: State) -> Literal["branch_a", "branch_b"]:
    """Route to both branches."""
    pass  # This is actually done via multiple add_edge calls

# Fan-out: multiple edges from one node
builder.add_edge("start", "branch_a")
builder.add_edge("start", "branch_b")

# Fan-in: single target after all parallel branches
builder.add_edge("branch_a", "merge")
builder.add_edge("branch_b", "merge")
# LangGraph waits for ALL incoming edges before executing "merge"
```

---

## Node Return Values — Type Reference

| Return Type | Behavior |
|---|---|
| `dict` | Merges into state via reducers |
| `Command(update=..., goto=...)` | Updates state AND routes |
| `Command(goto=...)` | Routes without state update |
| `Command(resume=...)` | Used to resume from interrupt |
| `None` | No state update (node is a side-effect-only step) |

---

## Error Handling in Nodes

```python
from langgraph.types import RetryPolicy, Command
from langgraph.errors import NodeError

# Retry on specific exceptions
builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,   # Initial wait in seconds
        backoff_factor=2.0,     # Exponential backoff
        jitter=True,            # Add randomness to prevent thundering herd
        retry_on=(ConnectionError, TimeoutError),
    ),
)

# Error handler — runs after retries exhausted
def handle_api_error(state: State, error: NodeError) -> Command:
    return Command(
        update={"error": str(error), "status": "failed"},
        goto="error_recovery",
    )

builder.add_node(
    "api_call",
    call_external_api,
    retry_policy=RetryPolicy(max_attempts=3),
    error_handler=handle_api_error,
)
```

---

## Async Nodes and Concurrency

Async nodes are required when your graph uses `ainvoke` / `astream`:

```python
import asyncio

async def parallel_search(state: State) -> dict:
    # Run multiple async operations concurrently
    results = await asyncio.gather(
        search_web(state["query"]),
        search_database(state["query"]),
        search_knowledge_base(state["query"]),
    )
    return {"results": list(results)}
```

### Mix of Sync and Async Nodes

```python
# LangGraph handles mixed sync/async — just define each node appropriately
def sync_preprocess(state: State) -> dict:       # Sync OK
    return {"processed": preprocess(state["raw"])}

async def async_llm_call(state: State) -> dict:  # Async for I/O
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# Both work in the same graph
```

---

## Common Pitfalls

### 1. Conditional Edge Router Returns Wrong Type
```python
# ❌ Returning END constant directly doesn't work in conditional edges
def route(state: State):
    return END  # This won't work

# ✅ Return the string "__end__" or use END sentinel as a key
def route(state: State) -> str:
    return "__end__"  # String form of END

# OR map it in the routing dict
builder.add_conditional_edges("node", route, {"end": END})
```

### 2. Node Modifying State In-Place
```python
# ❌ Mutating state in-place — bypasses reducers, causes bugs
def bad_node(state: State) -> dict:
    state["messages"].append(new_msg)  # Mutation!
    return {}

# ✅ Return updates, let LangGraph apply them via reducers
def good_node(state: State) -> dict:
    return {"messages": [new_msg]}  # add_messages reducer handles append
```

---

## Related Topics
- [`01-state-graph-fundamentals.md`](./01-state-graph-fundamentals.md) — Building graphs
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — `interrupt()` inside nodes
- [`09-map-reduce-parallelism.md`](./09-map-reduce-parallelism.md) — Send API for dynamic fan-out
- [`12-production-error-handling.md`](./12-production-error-handling.md) — RetryPolicy in production
