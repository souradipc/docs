# Production Error Handling

## Error Handling Philosophy in LangGraph

LangGraph runs in an async, distributed-ish environment where failures are normal. Three layers of defense:

1. **RetryPolicy** — automatically retry transient failures at the node/task level
2. **error_handler** — intercept and route around node failures
3. **Saga/compensation** — undo already-completed work when downstream steps fail

---

## RetryPolicy on Nodes

Attach a `RetryPolicy` when registering a node:

```python
from langgraph.types import RetryPolicy

retry_policy = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,       # seconds before first retry
    backoff_factor=2.0,         # exponential: 1s, 2s, 4s
    max_interval=60.0,          # cap at 60s
    jitter=True,                # add randomness to prevent thundering herd
    retry_on=(                  # retry ONLY on these exception types
        ConnectionError,
        TimeoutError,
        httpx.HTTPStatusError,
    ),
)

builder = StateGraph(State)
builder.add_node("call_api", call_external_api, retry=retry_policy)
builder.add_node("call_llm", call_llm, retry=RetryPolicy(max_attempts=5))
```

**Default behavior without RetryPolicy**: Any exception in a node **immediately fails the entire graph run**.

---

## Error Handler — Route Around Failures

Use `error_handler` to catch a failed node and redirect to a recovery path:

```python
def api_call_node(state: State) -> dict:
    response = requests.get(state["url"], timeout=5)
    response.raise_for_status()
    return {"api_result": response.json()}

def handle_api_failure(state: State) -> dict:
    """Called when api_call_node fails after all retries."""
    error = state.get("error")  # NodeError is stored here
    return {
        "api_result": None,
        "fallback_used": True,
        "error_message": str(error),
    }

builder.add_node("call_api", api_call_node, retry=RetryPolicy(max_attempts=3))
builder.add_node("handle_api_failure", handle_api_failure)

# Route: on success → next_step; on failure → handle_api_failure
builder.add_edge("call_api", "next_step")  # Normal path
# error_handler is called when the node exhausts retries
builder.add_node(
    "call_api",
    api_call_node,
    retry=retry_policy,
    # error_handler is specified at add_node time
)
```

---

## NodeError — Inspecting Failures

When a node fails, LangGraph wraps the exception in `NodeError`:

```python
from langgraph.errors import NodeError

try:
    result = graph.invoke(state)
except NodeError as e:
    print(f"Node that failed: {e.node_name}")
    print(f"Underlying exception: {e.__cause__}")
    print(f"State at failure: {e.state}")
```

When catching errors inside the graph (e.g., in an error handler node):

```python
def error_handler_node(state: State) -> dict:
    # LangGraph sets a special key when routing through error handler
    node_error: NodeError = state.get("__error__")
    if node_error:
        logger.error(
            "Node %s failed: %s",
            node_error.node_name,
            str(node_error.__cause__),
        )
    return {"status": "error_handled"}
```

---

## Graceful Degradation Pattern

Use routing logic to degrade gracefully when a component fails:

```python
from typing import Literal

class State(TypedDict):
    query: str
    premium_result: str | None
    fallback_result: str | None
    final_answer: str

async def premium_api_call(state: State) -> dict:
    """Fast, expensive API."""
    try:
        result = await premium_service.search(state["query"])
        return {"premium_result": result}
    except Exception as e:
        logger.warning("Premium API failed: %s", e)
        return {"premium_result": None}  # Soft failure — don't raise

async def fallback_api_call(state: State) -> dict:
    """Slower, cheaper fallback."""
    result = await fallback_service.search(state["query"])
    return {"fallback_result": result}

def decide_path(state: State) -> Literal["synthesize", "fallback"]:
    if state["premium_result"] is not None:
        return "synthesize"
    return "fallback"

builder = (
    StateGraph(State)
    .add_node("premium", premium_api_call)
    .add_node("fallback", fallback_api_call)
    .add_node("synthesize", synthesize_answer)
    .add_edge(START, "premium")
    .add_conditional_edges("premium", decide_path, {
        "synthesize": "synthesize",
        "fallback": "fallback",
    })
    .add_edge("fallback", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

---

## Saga / Compensation Pattern

When multiple steps succeed in sequence but a later step fails, compensate by undoing earlier work:

```python
class SagaState(TypedDict):
    order_id: str
    payment_charged: bool
    inventory_reserved: bool
    notification_sent: bool
    compensation_needed: bool
    error: str | None

async def charge_payment(state: SagaState) -> dict:
    await payment_service.charge(state["order_id"])
    return {"payment_charged": True}

async def reserve_inventory(state: SagaState) -> dict:
    try:
        await inventory_service.reserve(state["order_id"])
        return {"inventory_reserved": True}
    except Exception as e:
        # Failed AFTER payment was charged — need compensation
        return {"inventory_reserved": False, "compensation_needed": True, "error": str(e)}

async def compensate_payment(state: SagaState) -> dict:
    """Rollback: refund the payment."""
    logger.warning("Compensating: refunding payment for order %s", state["order_id"])
    await payment_service.refund(state["order_id"])
    return {"payment_charged": False}

def should_compensate(state: SagaState) -> Literal["compensate", "continue"]:
    if state.get("compensation_needed"):
        return "compensate"
    return "continue"

saga_graph = (
    StateGraph(SagaState)
    .add_node("charge", charge_payment)
    .add_node("reserve", reserve_inventory)
    .add_node("notify", send_notification)
    .add_node("compensate_payment", compensate_payment)
    .add_node("saga_failed", handle_saga_failure)
    .add_edge(START, "charge")
    .add_edge("charge", "reserve")
    .add_conditional_edges("reserve", should_compensate, {
        "compensate": "compensate_payment",
        "continue": "notify",
    })
    .add_edge("compensate_payment", "saga_failed")
    .add_edge("notify", END)
    .add_edge("saga_failed", END)
    .compile()
)
```

---

## Circuit Breaker Pattern

Track failure rates and stop calling a broken service:

```python
import time
from collections import deque

class CircuitBreakerState(TypedDict):
    query: str
    circuit_open: bool
    failure_times: list[float]
    result: str | None

FAILURE_THRESHOLD = 5
WINDOW_SECONDS = 60

def check_circuit(state: CircuitBreakerState) -> Literal["open", "closed"]:
    now = time.time()
    recent_failures = [t for t in state.get("failure_times", []) if now - t < WINDOW_SECONDS]
    if len(recent_failures) >= FAILURE_THRESHOLD:
        return "open"
    return "closed"

async def call_service(state: CircuitBreakerState) -> dict:
    try:
        result = await flaky_service.call(state["query"])
        return {"result": result}
    except Exception as e:
        now = time.time()
        return {
            "result": None,
            "failure_times": state.get("failure_times", []) + [now],
        }

async def return_cached_fallback(state: CircuitBreakerState) -> dict:
    return {"result": get_cached_result(state["query"])}

circuit_graph = (
    StateGraph(CircuitBreakerState)
    .add_node("service_call", call_service)
    .add_node("cached_fallback", return_cached_fallback)
    .add_conditional_edges(START, check_circuit, {
        "open": "cached_fallback",
        "closed": "service_call",
    })
    .add_edge("service_call", END)
    .add_edge("cached_fallback", END)
    .compile()
)
```

---

## Monitoring Failed Runs

With a checkpointer, you can inspect and recover failed runs:

```python
# Get the state at the point of failure
config = {"configurable": {"thread_id": "run-xyz"}}
state = graph.get_state(config)
print(state.values)         # State at time of failure
print(state.next)           # Which nodes were pending
print(state.metadata)       # Includes "error" key if run failed

# Inspect history to find the failure point
for checkpoint in graph.get_state_history(config):
    if checkpoint.metadata.get("error"):
        print(f"Failed at step: {checkpoint.metadata['step']}")
        print(f"Error: {checkpoint.metadata['error']}")
        break

# Patch state and retry from failure point
graph.update_state(config, {"api_result": cached_fallback_result})
graph.invoke(None, config=config)  # Resume from last checkpoint
```

---

## Error Handling Checklist for Production

```
□ All external API calls wrapped in RetryPolicy (max_attempts=3, backoff)
□ RetryPolicy retry_on specifies only transient errors (ConnectionError, TimeoutError)
□ Non-transient errors (AuthError, NotFoundError) NOT retried
□ Graceful degradation routing for non-critical services
□ Saga compensation defined for all multi-step mutating workflows
□ NodeError logged with node_name and state snapshot
□ Failed runs visible via get_state_history inspection
□ Async nodes use async retry-compatible HTTP clients (httpx, aiohttp)
□ LLM calls retried on RateLimitError with exponential backoff
```

---

## Related Topics
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — Node registration and RetryPolicy
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Inspecting failed runs
- [`11-functional-api.md`](./11-functional-api.md) — RetryPolicy on `@task`
