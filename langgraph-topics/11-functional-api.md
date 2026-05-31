# Functional API — @entrypoint & @task

## Why the Functional API?

The `StateGraph` API requires you to think in terms of graphs: nodes, edges, and reducers. For many workflows this is natural, but sometimes a sequential Python function is clearer.

The **Functional API** (`@entrypoint`, `@task`) provides:
- Familiar Python function syntax
- **Durable execution**: if a task completes, it won't re-run on retry — LangGraph replays from checkpoints
- First-class async support
- Same streaming, HITL, and persistence as StateGraph

| Feature | StateGraph | Functional API |
|---|---|---|
| Explicit state schema | Required | Optional |
| Parallelism | Via Send / parallel edges | Via `asyncio.gather` over tasks |
| Routing | Conditional edges | Regular `if/else` |
| Durable execution | Full | Full |
| Learning curve | Higher | Lower |
| Best for | Complex branching workflows | Sequential pipelines, ETL |

---

## Basic Usage

```python
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter

@task
async def slow_step_1(input_data: str) -> str:
    """This is a 'task' — a durable unit of work."""
    result = await some_slow_api_call(input_data)
    return result

@task
async def slow_step_2(input_data: str) -> str:
    result = await another_slow_api_call(input_data)
    return result

@entrypoint(checkpointer=checkpointer)
async def my_workflow(input_data: str) -> str:
    """The entrypoint is the orchestrator — must use await on tasks."""
    result_1 = await slow_step_1(input_data)
    result_2 = await slow_step_2(result_1)
    return result_2
```

---

## Durable Execution — The Key Benefit

Tasks that complete are **not re-run** if the entrypoint is invoked again on the same thread:

```python
@task
async def fetch_from_api(item_id: str) -> dict:
    """Expensive API call — should only run once per item."""
    await asyncio.sleep(5)  # Simulate slow call
    return {"id": item_id, "data": "..."}

@entrypoint(checkpointer=MemorySaver())
async def process_items(item_ids: list[str]) -> list[dict]:
    # On FIRST invocation: all three tasks run
    results = await asyncio.gather(*[fetch_from_api(item_id) for item_id in item_ids])
    return list(results)

config = {"configurable": {"thread_id": "job-abc"}}

# First call — all tasks execute
await process_items.ainvoke(["item-1", "item-2", "item-3"], config=config)

# If this crashes mid-way and you retry with the SAME thread_id,
# completed tasks are replayed from checkpoint — NOT re-executed
await process_items.ainvoke(["item-1", "item-2", "item-3"], config=config)
```

This makes the functional API ideal for **idempotent batch jobs** and **multi-step ETL pipelines**.

---

## RetryPolicy on Tasks

```python
from langgraph.types import RetryPolicy

@task(
    retry=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,          # 1s, 2s, 4s
        jitter=True,
        retry_on=(ConnectionError, TimeoutError),
    )
)
async def call_external_api(query: str) -> dict:
    response = await httpx.AsyncClient().get(f"/api?q={query}")
    response.raise_for_status()
    return response.json()
```

---

## TimeoutPolicy on Tasks

```python
from langgraph.types import TimeoutPolicy

@task(
    timeout=TimeoutPolicy(
        idle_timeout=30,   # seconds — fail if no progress for 30s
    ),
    retry=RetryPolicy(max_attempts=2, retry_on=TimeoutError)
)
async def slow_llm_call(prompt: str) -> str:
    return await llm.ainvoke(prompt)
```

---

## Streaming in Functional API

Use `StreamWriter` to emit custom events during execution:

```python
from langgraph.types import StreamWriter

@entrypoint(checkpointer=checkpointer)
async def streaming_workflow(input: str, writer: StreamWriter) -> str:
    writer({"status": "started", "input": input})

    result_1 = await step_one(input)
    writer({"status": "step_1_done", "result": result_1})

    result_2 = await step_two(result_1)
    writer({"status": "step_2_done", "result": result_2})

    return result_2

# Client consumes stream
async for event in streaming_workflow.astream(
    "input data",
    config=config,
    stream_mode="custom",
):
    print(event)  # {"status": "started", ...}, {"status": "step_1_done", ...}, etc.
```

---

## Human-in-the-Loop in Functional API

`interrupt()` works the same way:

```python
from langgraph.types import interrupt, Command

@entrypoint(checkpointer=checkpointer)
async def approval_workflow(data: dict) -> dict:
    # Process data
    result = await process(data)

    # Pause and wait for human review
    decision = interrupt({
        "message": "Please review the result",
        "result": result,
    })

    if decision == "approve":
        await publish(result)
        return {"status": "published", "result": result}
    else:
        return {"status": "rejected"}

# Start the workflow
await approval_workflow.ainvoke(data, config=config)
# Returns when interrupted — check state

# Resume after human approves
await approval_workflow.ainvoke(
    Command(resume="approve"),
    config=config,
)
```

---

## Parallel Tasks in Functional API

Use `asyncio.gather` for concurrent execution:

```python
@task
async def search_source_a(query: str) -> list[str]:
    return await search_api_a(query)

@task
async def search_source_b(query: str) -> list[str]:
    return await search_api_b(query)

@task
async def synthesize(results_a: list[str], results_b: list[str]) -> str:
    combined = results_a + results_b
    return await llm.ainvoke(f"Synthesize: {combined}")

@entrypoint(checkpointer=checkpointer)
async def parallel_search(query: str) -> str:
    # Both searches run in parallel
    results_a, results_b = await asyncio.gather(
        search_source_a(query),
        search_source_b(query),
    )
    return await synthesize(results_a, results_b)
```

Each `@task` is individually checkpointed — if `synthesize` fails, re-running won't re-execute `search_source_a` or `search_source_b`.

---

## Full Production Example — Document Processing Pipeline

```python
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy, StreamWriter, interrupt

@task(retry=RetryPolicy(max_attempts=3, retry_on=Exception))
async def extract_text(doc_url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(doc_url)
        return await pdf_parser.extract(response.content)

@task
async def chunk_text(text: str) -> list[str]:
    return text_splitter.split_text(text)

@task(retry=RetryPolicy(max_attempts=2))
async def embed_chunk(chunk: str) -> list[float]:
    return await embedding_model.aembed(chunk)

@task
async def store_embeddings(doc_id: str, chunks: list[str], embeddings: list[list[float]]) -> int:
    return await vector_db.upsert_many(doc_id, chunks, embeddings)

@entrypoint(checkpointer=checkpointer)
async def ingest_document(doc: dict, writer: StreamWriter) -> dict:
    doc_url = doc["url"]
    doc_id = doc["id"]

    writer({"phase": "extracting", "doc_id": doc_id})
    text = await extract_text(doc_url)

    writer({"phase": "chunking"})
    chunks = await chunk_text(text)

    writer({"phase": "embedding", "chunk_count": len(chunks)})
    embeddings = await asyncio.gather(*[embed_chunk(c) for c in chunks])

    # Require human approval for large documents
    if len(chunks) > 100:
        decision = interrupt({
            "message": f"Document has {len(chunks)} chunks — approve ingestion?",
            "doc_id": doc_id,
        })
        if decision != "approve":
            return {"status": "rejected", "doc_id": doc_id}

    writer({"phase": "storing"})
    count = await store_embeddings(doc_id, chunks, embeddings)

    return {"status": "complete", "doc_id": doc_id, "chunks_stored": count}
```

---

## When to Use Functional API vs StateGraph

**Use Functional API when:**
- Workflow is essentially sequential with occasional parallel steps
- You want idiomatic Python with `if/else` routing
- ETL pipelines, document processing, batch jobs
- Quick prototyping

**Use StateGraph when:**
- Complex routing logic with many branches
- Multiple agents interacting through shared state
- Cycles (loops until a condition is met)
- Need explicit state schema for validation
- Building multi-agent supervisors

---

## Related Topics
- [`01-state-graph-fundamentals.md`](./01-state-graph-fundamentals.md) — StateGraph alternative
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — `interrupt()` mechanics
- [`06-streaming.md`](./06-streaming.md) — `StreamWriter` patterns
- [`12-production-error-handling.md`](./12-production-error-handling.md) — `RetryPolicy` details
