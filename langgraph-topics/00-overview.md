# LangGraph — Overview & Architecture

## What Is LangGraph?

LangGraph is a **low-level orchestration framework and runtime** for building stateful, long-running AI agent workflows. It models agent logic as directed graphs where:

- **Nodes** = Python functions (agent logic, LLM calls, tool invocations)
- **Edges** = Control flow (fixed or conditional)
- **State** = Shared typed data structure passed through every node

LangGraph is not a higher-level abstraction on top of LangChain — it is a fundamentally different execution model focused on **durability, controllability, and observability** that LCEL chains cannot provide.

---

## Why LangGraph Exists — The Problem with Chains

LangChain's LCEL is excellent for **linear pipelines**. But real-world agent workflows are not linear:

| Challenge | LCEL Chain | LangGraph |
|---|---|---|
| Multi-turn conversation loops | ❌ No native state across turns | ✅ Built-in checkpointing |
| Conditional branching | Limited (RunnableBranch) | ✅ Full conditional edge routing |
| Cyclic execution (loops) | ❌ Not possible | ✅ First-class cycles |
| Human approval mid-workflow | ❌ Not possible | ✅ `interrupt()` + `Command(resume=)` |
| Resume after crash | ❌ Stateless | ✅ Checkpoint-based resumption |
| Parallel sub-tasks | Limited | ✅ Native fan-out/fan-in |
| Long-running async workflows | ❌ | ✅ Durable execution |
| Multi-agent orchestration | ❌ | ✅ Subgraphs + handoffs |
| Time travel / state replay | ❌ | ✅ Full state history |
| Streaming intermediate steps | Limited | ✅ Multiple stream modes |

---

## LangGraph vs LangChain LCEL — When to Use Which

```
Use LCEL when:                         Use LangGraph when:
  - Single-pass processing               - Multi-turn conversation agents
  - Simple linear pipelines              - Loops and conditional retries
  - RAG with no memory                   - Human-in-the-loop workflows
  - One-shot extraction/parsing          - Long-running background tasks
  - Stateless transformations            - Multi-agent systems
                                         - Any production agent needing durability
```

---

## Core Execution Model

LangGraph executes a compiled graph by:

1. **Initializing state** from the input
2. **Traversing nodes** in topological order, starting at `START`
3. **Applying state updates** — each node's return dict is merged into the state via reducers
4. **Evaluating edges** — determines the next node(s) to execute
5. **Checkpointing** — state is saved to persistent storage after every node (if checkpointer is set)
6. **Terminating** when execution reaches `END` or all active branches complete

Crucially, the graph is a **superstep model** — each "step" can consist of multiple parallel node executions. LangGraph waits for all parallel branches to complete before proceeding to their shared downstream node (fan-in).

---

## Key Primitives

### State
A `TypedDict` (or Pydantic model/dataclass) defining all shared data. Each field can have a **reducer** controlling how updates are merged.

### Node
A Python function `(state: State) -> dict` or `(state: State) -> Command`. Nodes are the unit of computation. They receive the full current state and return a partial update.

### Edge
A directed connection between nodes. Can be:
- **Fixed**: `add_edge("a", "b")` — always go from a to b
- **Conditional**: `add_conditional_edges("a", router_fn)` — dynamically pick next node

### Command
A return type that combines **state update + routing** in one step:
```python
return Command(update={"status": "done"}, goto="finalize")
```

### Checkpoint
A snapshot of the full graph state at a specific step. Enables resumption, time travel, and human-in-the-loop.

### Thread
A unique conversation session identified by `thread_id`. Each thread has its own independent state history in the checkpointer.

### interrupt()
A function called inside a node to pause execution and surface a value to the caller. Execution resumes when `Command(resume=value)` is provided.

---

## Package Structure

```
langgraph                    # Core graph execution engine
├── graph                    # StateGraph, START, END
├── checkpoint               # Checkpointer implementations
│   ├── memory               # InMemorySaver
│   ├── sqlite               # SqliteSaver / AsyncSqliteSaver
│   └── postgres             # PostgresSaver / AsyncPostgresSaver
├── prebuilt                 # create_react_agent, ToolNode, tools_condition
├── store                    # InMemoryStore, BaseStore (cross-thread memory)
├── func                     # Functional API (@entrypoint, @task)
├── types                    # Command, Send, interrupt, RetryPolicy
└── runtime                  # Runtime object for accessing store/context in nodes

langgraph-sdk                # Client SDK for LangGraph Platform
langgraph-cli                # CLI (langgraph dev, langgraph build)
```

---

## Request Lifecycle

```
User Input
    │
    ▼
graph.invoke(input, config)
    │
    ├─── Load checkpoint (if thread_id + checkpointer)
    │
    ├─── Merge input into state
    │
    ├─── Execute START → node_a
    │         │
    │         ├─── node_a(state) → update state
    │         ├─── Checkpoint saved (step N)
    │         │
    │         ├─── Conditional edge → node_b OR node_c
    │         │
    │         └─── node_b(state) → update state
    │                   │
    │                   └─── interrupt() called → PAUSE
    │                             │
    │                             ▼
    │                       Returns to caller (with interrupts)
    │
    ├─── graph.invoke(Command(resume="ok"), config)
    │         │
    │         └─── Resumes from interrupt inside node_b
    │                   │
    │                   └─── → END
    │
    └─── Return final state
```

---

## Installation

```bash
pip install langgraph

# Checkpointer backends
pip install langgraph-checkpoint-sqlite    # SQLite
pip install langgraph-checkpoint-postgres  # PostgreSQL (async)

# LangGraph Platform SDK
pip install langgraph-sdk

# CLI for local dev server
pip install langgraph-cli

# Verify
python -c "import langgraph; print(langgraph.__version__)"
```

---

## Quick Mental Model

Think of LangGraph as a **state machine with LLM-powered transitions**:

- **State** = the memory shared across all steps
- **Nodes** = the work units (call LLM, run tool, ask human, transform data)
- **Edges** = the transitions (always, or conditional on state)
- **Checkpointer** = the disk/database that makes it durable
- **interrupt()** = the pause button for human-in-the-loop
- **Store** = the long-term memory that persists across different threads

---

## Related Topics
- [`01-state-graph-fundamentals.md`](./01-state-graph-fundamentals.md) — Building your first graph
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Durable execution
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — Pause and resume workflows
