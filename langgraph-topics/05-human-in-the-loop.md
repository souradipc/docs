# Human-in-the-Loop Workflows

## What Is Human-in-the-Loop (HITL)?

Human-in-the-loop is the ability to **pause a running workflow**, surface information to a human, receive their input, and **resume execution** from exactly where it stopped.

LangGraph's HITL is fundamentally different from a simple "ask a question" pattern:
- The **entire agent state is persisted** while waiting for human input
- Waiting period can be **minutes, hours, or days** — no resource held
- Human can **modify the state** before resuming
- Multiple agents can be waiting simultaneously
- Supports **approval gates, form-filling, correction flows, and escalation**

---

## The `interrupt()` Function

`interrupt()` is called inside a node to pause execution. It:
1. Saves the current state to the checkpointer
2. Returns the provided value to the caller (as the `interrupts` field in the result)
3. **Pauses the node mid-execution** — the node function returns nothing yet
4. Waits for `Command(resume=value)` to be passed on the next invocation

```python
from langgraph.types import interrupt

def human_review_node(state: State) -> dict:
    # Pause here and surface the pending action to the human
    decision = interrupt(
        {
            "type": "approval_required",
            "action": state["planned_action"],
            "context": state["messages"][-1].content,
        }
    )
    # ↑ Execution PAUSES here. `decision` will contain whatever the human provides.

    # This code runs AFTER human resumes with Command(resume=...)
    if decision == "approve":
        return {"status": "approved", "action_taken": True}
    else:
        return {"status": "rejected", "action_taken": False}
```

---

## The `Command(resume=)` Primitive

Resume a paused graph by passing `Command(resume=value)` as the input:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "workflow-123"}}

# Step 1: Start workflow — hits interrupt and pauses
result = graph.invoke({"messages": [("human", "Deploy to production")]}, config=config)

# `result.interrupts` contains what interrupt() was called with
print(result.interrupts)
# (Interrupt(value={'type': 'approval_required', 'action': 'deploy'}),)

# Step 2: Human reviews, then resumes with decision
resumed = graph.invoke(Command(resume="approve"), config=config)
# The interrupt() call inside the node returns "approve"
# Execution continues from after the interrupt() call
```

---

## Full Approval Workflow

```python
from typing import Literal, Optional, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI

class WorkflowState(TypedDict):
    task: str
    plan: Optional[str]
    status: Optional[Literal["pending", "approved", "rejected", "done"]]

llm = ChatOpenAI(model="gpt-4o")

def plan_action(state: WorkflowState) -> dict:
    """LLM generates a plan."""
    response = llm.invoke(f"Create a step-by-step plan for: {state['task']}")
    return {"plan": response.content, "status": "pending"}

def request_approval(state: WorkflowState) -> Command[Literal["execute", "cancel"]]:
    """Pause for human approval."""
    decision = interrupt({
        "message": "Please review and approve or reject this plan:",
        "plan": state["plan"],
    })
    # decision is whatever the human passes in Command(resume=...)
    if decision == "approve":
        return Command(goto="execute", update={"status": "approved"})
    else:
        return Command(goto="cancel", update={"status": "rejected"})

def execute_plan(state: WorkflowState) -> dict:
    return {"status": "done"}

def cancel_plan(state: WorkflowState) -> dict:
    return {"status": "rejected"}

# Build graph
builder = StateGraph(WorkflowState)
builder.add_node("plan", plan_action)
builder.add_node("approval", request_approval)
builder.add_node("execute", execute_plan)
builder.add_node("cancel", cancel_plan)
builder.add_edge(START, "plan")
builder.add_edge("plan", "approval")
builder.add_edge("execute", END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Usage
config = {"configurable": {"thread_id": "deploy-task-1"}}

# Start workflow
result = graph.invoke({"task": "Deploy new model to production"}, config=config)
print(result.interrupts[0].value)  # Show plan to human in UI

# Human approves
final = graph.invoke(Command(resume="approve"), config=config)
print(final.value["status"])  # "done"
```

---

## `interrupt_before` / `interrupt_after` — Compile-Time Breakpoints

An alternative to the `interrupt()` function: declare breakpoints at compile time. The graph pauses **before or after** the specified nodes on every execution.

```python
graph = builder.compile(
    checkpointer=InMemorySaver(),
    interrupt_before=["dangerous_action"],   # Always pause before this node
    interrupt_after=["generate_plan"],        # Always pause after this node
)

config = {"configurable": {"thread_id": "task-1"}}

# Graph runs until it hits "generate_plan", then pauses
result = graph.invoke(input_state, config=config)

# Inspect the generated plan
state = graph.get_state(config)
plan = state.values.get("plan")
print(plan)

# Optionally modify the plan
if plan_needs_modification:
    graph.update_state(
        config,
        {"plan": "Modified plan here"},
        as_node="human_editor",
    )

# Resume execution
result = graph.invoke(None, config=config)
```

### `interrupt()` vs Compile-Time Breakpoints

| Feature | `interrupt()` in node | `interrupt_before/after` at compile |
|---|---|---|
| Conditional interruption | ✅ Can decide at runtime | ❌ Always interrupts |
| Pass value to caller | ✅ `interrupt(value)` | ❌ No value surfaced |
| Receive value from human | ✅ `answer = interrupt(...)` | ❌ Must call `update_state` |
| Code change required | ✅ Node code changes | ❌ Just compile config |
| Multiple interrupts in one node | ✅ Call `interrupt()` multiple times | ❌ Single break per node |

---

## Multiple Interrupts in a Single Node

A node can call `interrupt()` multiple times for sequential human interactions:

```python
def multi_step_review(state: State) -> dict:
    # First question
    name = interrupt("What is your name?")

    # Second question — only asked after first is answered
    age = interrupt(f"Hello {name}! How old are you?")

    return {"user_info": {"name": name, "age": age}}

# Usage
config = {"configurable": {"thread_id": "registration-1"}}

# First interrupt
graph.invoke({"messages": []}, config=config)
# → Pauses, asking "What is your name?"

# Resume with name
graph.invoke(Command(resume="Alice"), config=config)
# → Pauses, asking "Hello Alice! How old are you?"

# Resume with age
result = graph.invoke(Command(resume=30), config=config)
# → Completes, state has {"user_info": {"name": "Alice", "age": 30}}
```

---

## Modifying State Before Resuming

After a graph pauses, you can modify its state before resuming. The human can correct, enrich, or override:

```python
config = {"configurable": {"thread_id": "correction-flow-1"}}

# Graph pauses with a drafted email
result = graph.invoke({"task": "Draft email to client"}, config=config)

# Get current state
state = graph.get_state(config)
draft = state.values["draft_email"]
print(f"Draft:\n{draft}")

# Human edits the draft, inject corrected version
graph.update_state(
    config,
    {"draft_email": "Dear Client,\n[Human-edited content here]\n\nBest regards"},
    as_node="email_drafter",
)

# Resume — the node continues from after interrupt() with the corrected draft
result = graph.invoke(Command(resume="send"), config=config)
```

---

## HITL in Production — Architecture

```
Client (Browser/API)
    │
    ▼
POST /tasks/{task_id}/start
    │
    ├─── graph.invoke(input, config) → hits interrupt
    ├─── Store interrupt data in DB
    └─── Return 202 Accepted + interrupt_data to UI
         │
         ▼
    [User reviews in UI — seconds to days]
         │
POST /tasks/{task_id}/approve
    │
    ├─── graph.invoke(Command(resume=decision), config)
    └─── Return 200 with result
```

```python
from fastapi import FastAPI, BackgroundTasks
from langgraph.types import Command

app = FastAPI()

@app.post("/workflows")
async def start_workflow(request: WorkflowRequest):
    config = {"configurable": {"thread_id": request.workflow_id}}
    result = await graph.ainvoke(request.input, config=config)

    if result.interrupts:
        # Store interrupt for later resumption
        await db.save_pending_approval(
            workflow_id=request.workflow_id,
            interrupt_data=result.interrupts[0].value,
        )
        return {"status": "pending_approval", "data": result.interrupts[0].value}

    return {"status": "complete", "result": result.value}

@app.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str, decision: dict):
    config = {"configurable": {"thread_id": workflow_id}}
    result = await graph.ainvoke(
        Command(resume=decision["value"]),
        config=config,
    )
    return {"status": "complete", "result": result.value}
```

---

## Common Pitfalls

### 1. No Checkpointer = interrupt() Crashes
```python
# ❌ interrupt() requires a checkpointer — will raise GraphInterrupt
graph = builder.compile()  # No checkpointer!
graph.invoke(state)  # GraphInterrupt exception raised, can't resume

# ✅ Always attach a checkpointer when using interrupt()
graph = builder.compile(checkpointer=InMemorySaver())
```

### 2. Passing None Without a Paused Thread
```python
# ❌ graph.invoke(None, config) only works when graph is paused
config = {"configurable": {"thread_id": "fresh-thread"}}
graph.invoke(None, config)  # Error: no prior checkpoint

# ✅ Only invoke with None to resume a paused graph
state = graph.get_state(config)
if state.next:  # Check if graph is actually paused
    graph.invoke(None, config)
```

### 3. interrupt() in a Node Without Checkpointer = Lost State
```python
# ❌ Any crash between interrupt() and resumption loses state
# ✅ Use a persistent checkpointer (SQLite or PostgreSQL) in production
```

---

## Related Topics
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Checkpointers (required for HITL)
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — Command object
- [`13-langgraph-platform.md`](./13-langgraph-platform.md) — LangGraph Server for async HITL workflows
