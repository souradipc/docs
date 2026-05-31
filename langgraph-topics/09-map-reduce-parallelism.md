# Map-Reduce & Parallelism

## Why Parallelism in LangGraph?

LLM calls are slow (1-10 seconds each). Sequential execution compounds that latency:

- 5 search queries × 2s each = **10 seconds sequential**
- 5 search queries in parallel = **2 seconds** (limited by slowest)

LangGraph offers two mechanisms for parallelism:

1. **Static fan-out** — multiple fixed edges from one node
2. **Dynamic fan-out with `Send`** — spawn N workers at runtime based on data

---

## Static Parallelism — Multiple Fixed Edges

When you know at graph-build time which nodes should run in parallel, add multiple edges from the same source node:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined: str

# Three nodes run in parallel after START
def write_joke(state: State) -> dict:
    response = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": response.content}

def write_story(state: State) -> dict:
    response = llm.invoke(f"Write a short story about {state['topic']}")
    return {"story": response.content}

def write_poem(state: State) -> dict:
    response = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": response.content}

def combine(state: State) -> dict:
    return {"combined": f"{state['joke']}\n\n{state['story']}\n\n{state['poem']}"}

graph = (
    StateGraph(State)
    .add_node("joke", write_joke)
    .add_node("story", write_story)
    .add_node("poem", write_poem)
    .add_node("combine", combine)
    # Fan-out: START → three parallel nodes
    .add_edge(START, "joke")
    .add_edge(START, "story")
    .add_edge(START, "poem")
    # Fan-in: all three → combine (LangGraph waits for all to complete)
    .add_edge("joke", "combine")
    .add_edge("story", "combine")
    .add_edge("poem", "combine")
    .add_edge("combine", END)
    .compile()
)

result = graph.invoke({"topic": "cybersecurity"})
# joke, story, and poem all generated concurrently
```

**Fan-in rule**: A node with multiple incoming edges waits for **all** incoming branches to complete before executing.

---

## Dynamic Fan-Out — The `Send` API

When you don't know at build time how many parallel workers you need, use `Send` to dynamically spawn them:

```python
from langgraph.types import Send
from typing import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list[str]                           # Input list to parallelize over
    jokes: Annotated[list[str], operator.add]     # Accumulated results (reducer needed!)
    best_joke: str

# Step 1: Generate the list of things to parallelize
def generate_subjects(state: OverallState) -> dict:
    response = llm.invoke(f"List 5 subtopics of {state['topic']}. JSON list only.")
    subjects = json.loads(response.content)
    return {"subjects": subjects}

# Step 2: This node runs once PER subject (spawned dynamically)
def write_joke_for_subject(state: OverallState) -> dict:
    # Note: state["subject"] is injected by Send() — it's not in OverallState
    subject = state["subject"]  # type: ignore
    response = llm.invoke(f"Write a funny joke about {subject}")
    return {"jokes": [response.content]}  # operator.add will accumulate these

# Step 3: Fan-in — runs after ALL joke workers complete
def pick_best(state: OverallState) -> dict:
    response = llm.invoke(f"Pick the funniest joke:\n{chr(10).join(state['jokes'])}")
    return {"best_joke": response.content}

# The conditional edge function returns a list of Send() objects
def spawn_joke_workers(state: OverallState) -> list[Send]:
    return [
        Send("write_joke", {"subject": subject})
        for subject in state["subjects"]
    ]

graph = (
    StateGraph(OverallState)
    .add_node("generate_subjects", generate_subjects)
    .add_node("write_joke", write_joke_for_subject)
    .add_node("pick_best", pick_best)
    .add_edge(START, "generate_subjects")
    # Dynamic fan-out
    .add_conditional_edges("generate_subjects", spawn_joke_workers, ["write_joke"])
    # Fan-in: all write_joke instances → pick_best
    .add_edge("write_joke", "pick_best")
    .add_edge("pick_best", END)
    .compile()
)

result = graph.invoke({"topic": "cybersecurity"})
print(result["best_joke"])
```

---

## Send API Deep Dive

`Send(node_name, state_update)` spawns a new execution of `node_name` with an augmented state:

```python
from langgraph.types import Send

# Send accepts:
# 1. A node name (string)
# 2. A dict to merge into the current state for that worker

# The worker node receives the normal state PLUS the injected fields
def spawn_workers(state: MainState) -> list[Send]:
    return [
        Send("process_item", {
            "item": item,          # Injected for this worker only
            "item_index": i,       # Additional context
        })
        for i, item in enumerate(state["items"])
    ]

# The process_item node can access "item" and "item_index"
# even though they're not in MainState's TypedDict definition
def process_item(state: MainState) -> dict:
    item = state["item"]          # Injected by Send
    idx = state["item_index"]     # Injected by Send
    result = transform(item)
    return {"results": [result]}  # operator.add reducer accumulates
```

---

## Parallel Research Pipeline

Real-world example: Parallel search + LLM synthesis:

```python
from typing import TypedDict, Annotated, List
import operator

class ResearchState(TypedDict):
    question: str
    search_queries: List[str]               # Expanded queries
    search_results: Annotated[List[dict], operator.add]  # Parallel results
    final_answer: str

def expand_queries(state: ResearchState) -> dict:
    """Generate multiple search queries for the question."""
    response = llm.invoke(f"""Generate 4 diverse search queries for: {state['question']}
    Return as JSON list of strings.""")
    queries = json.loads(response.content)
    return {"search_queries": queries}

def execute_search(state: ResearchState) -> dict:
    """Execute one search query (called in parallel)."""
    query = state["search_query"]  # Injected by Send
    results = search_api(query)
    return {"search_results": [{"query": query, "results": results}]}

def synthesize_answers(state: ResearchState) -> dict:
    """Synthesize all search results into a final answer."""
    context = "\n\n".join(
        f"Query: {r['query']}\nResults: {r['results']}"
        for r in state["search_results"]
    )
    response = llm.invoke(f"Answer based on these search results:\n{context}\n\nQuestion: {state['question']}")
    return {"final_answer": response.content}

def fan_out_searches(state: ResearchState) -> list[Send]:
    return [
        Send("execute_search", {"search_query": q})
        for q in state["search_queries"]
    ]

research_graph = (
    StateGraph(ResearchState)
    .add_node("expand", expand_queries)
    .add_node("execute_search", execute_search)
    .add_node("synthesize", synthesize_answers)
    .add_edge(START, "expand")
    .add_conditional_edges("expand", fan_out_searches, ["execute_search"])
    .add_edge("execute_search", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

---

## Parallelism with Subgraphs

Each `Send` can spawn a subgraph, enabling complex parallel workflows:

```python
# A full subgraph to process each section of a document
section_processor = (
    StateGraph(SectionState)
    .add_node("extract", extract_entities)
    .add_node("classify", classify_content)
    .add_node("summarize", summarize_section)
    .add_edge(START, "extract")
    .add_edge("extract", "classify")
    .add_edge("classify", "summarize")
    .compile()
)

def dispatch_section_processors(state: DocumentState) -> list[Send]:
    return [
        Send("process_section", {"section_text": section, "section_id": i})
        for i, section in enumerate(state["sections"])
    ]

def process_section(state: DocumentState) -> dict:
    """Runs the section_processor subgraph for each section."""
    result = section_processor.invoke({
        "section_text": state["section_text"],   # Injected by Send
        "section_id": state["section_id"],
    })
    return {"processed_sections": [result["summary"]]}
```

---

## Concurrency Control

LangGraph runs parallel nodes concurrently. For I/O-bound tasks, use async:

```python
import asyncio

async def parallel_async_node(state: State) -> dict:
    """Run multiple async operations concurrently within a node."""
    results = await asyncio.gather(
        async_search_1(state["query"]),
        async_search_2(state["query"]),
        async_db_lookup(state["query"]),
    )
    return {"results": list(results)}

# When using async nodes, invoke the graph with ainvoke
result = await graph.ainvoke(state)
```

---

## Parallel Execution vs Sequential — Tradeoffs

| Aspect | Sequential | Parallel (Static) | Dynamic (Send) |
|---|---|---|---|
| Total latency | Additive | Max of branches | Max of workers |
| Implementation | Simple | Medium | Complex |
| API cost | Same | Same | Same |
| Error isolation | Any failure stops all | One failure stops graph | Worker failures isolated |
| Result ordering | Deterministic | Non-deterministic | Non-deterministic |
| Memory | Low | Medium (all in-flight) | High (N workers in-flight) |

---

## Common Pitfalls

### 1. Missing `operator.add` Reducer for Parallel Writes
```python
# ❌ All parallel workers overwrite each other
class State(TypedDict):
    results: list[str]  # No reducer! Only one result survives

# ✅ Use operator.add so all workers contribute
class State(TypedDict):
    results: Annotated[list[str], operator.add]
```

### 2. Using `Send` Without Specifying the Target in `add_conditional_edges`
```python
# ❌ Misses the target node declaration
builder.add_conditional_edges("fan_out", spawn_workers)

# ✅ Declare which nodes Send can target
builder.add_conditional_edges("fan_out", spawn_workers, ["worker_node"])
```

### 3. Unbounded Parallelism
```python
# ❌ Spawning 1000 workers can overwhelm the API rate limits
def spawn_all(state):
    return [Send("worker", {"item": x}) for x in state["huge_list"]]  # 1000 workers!

# ✅ Batch into manageable chunks
def spawn_batched(state, batch_size=10):
    chunks = [state["items"][i:i+batch_size] for i in range(0, len(state["items"]), batch_size)]
    return [Send("batch_worker", {"batch": chunk}) for chunk in chunks]
```

---

## Related Topics
- [`02-state-design-patterns.md`](./02-state-design-patterns.md) — Reducers for parallel state
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — `Command` with `Send`
- [`07-multi-agent-systems.md`](./07-multi-agent-systems.md) — Parallel agent invocations
