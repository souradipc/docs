# LangGraph Platform & Deployment

## What Is LangGraph Platform?

LangGraph Platform wraps your LangGraph application in a production-ready server with:
- **Persistent storage** (Postgres-backed checkpointer + store)
- **REST API** auto-generated from your graph
- **Background runs** (fire-and-forget)
- **Cron jobs** (scheduled graph runs)
- **Double-texting handling** (what to do when user sends while run is in progress)
- **Horizontal scaling** with built-in task queue

| Deployment Option | Description |
|---|---|
| `langgraph dev` | Local development server (SQLite, hot reload) |
| LangGraph Cloud | Fully managed (SaaS) |
| Self-hosted (Docker) | Docker Compose / Kubernetes, your infra |

---

## `langgraph.json` — Configuration File

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./app/agents/my_agent.py:graph",
    "research": "./app/agents/research.py:graph"
  },
  "env": ".env",
  "python_version": "3.11",
  "dockerfile_lines": [
    "RUN apt-get install -y tesseract-ocr"
  ]
}
```

- `graphs` maps an **assistant name** to a Python module path and variable name
- Multiple graphs can be served from one server
- `env` loads environment variables

---

## Local Development

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Start local dev server (auto-reload on file changes)
langgraph dev

# Serves on http://localhost:2024
# Swagger UI at http://localhost:2024/docs
```

The dev server uses SQLite for persistence. Full server features available locally.

---

## REST API — Core Endpoints

LangGraph Platform exposes a standard REST API:

### Assistants (Graph Versions)
```
GET  /assistants                         # List all assistants
GET  /assistants/{assistant_id}          # Get assistant
POST /assistants/search                  # Search assistants
```

### Threads (Conversation State)
```
POST /threads                            # Create thread
GET  /threads/{thread_id}/state          # Get current state
POST /threads/{thread_id}/state          # Update state
GET  /threads/{thread_id}/history        # Get state history
```

### Runs (Graph Executions)
```
POST /threads/{thread_id}/runs           # Background run (fire-and-forget)
POST /threads/{thread_id}/runs/stream    # Streaming run
POST /threads/{thread_id}/runs/wait      # Synchronous run (wait for result)
GET  /threads/{thread_id}/runs           # List all runs on thread
POST /runs/stream                        # Stateless streaming run
```

### Crons
```
POST /threads/{thread_id}/runs/crons     # Create scheduled run
DELETE /runs/crons/{cron_id}             # Cancel cron
```

---

## Python SDK

```python
from langgraph_sdk import get_sync_client, get_client

# Sync client
client = get_sync_client(url="http://localhost:2024", api_key="...")

# Async client
client = get_client(url="http://localhost:2024", api_key="...")

# Create a thread
thread = client.threads.create()
thread_id = thread["thread_id"]

# Streaming run
async for chunk in client.runs.stream(
    thread_id=thread_id,
    assistant_id="agent",        # Must match key in langgraph.json
    input={"messages": [{"role": "user", "content": "Hello!"}]},
    stream_mode="messages",      # values | updates | messages | debug
):
    if chunk.event == "messages/partial":
        print(chunk.data["content"], end="", flush=True)

# Wait run (blocking)
result = client.runs.wait(
    thread_id=thread_id,
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]},
)
print(result["messages"][-1]["content"])
```

---

## Streaming from Local Server (Development)

When using `langgraph dev`, use `stream_mode="messages-tuple"` for message streaming:

```python
async for chunk in client.runs.stream(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    stream_mode="messages-tuple",   # Use this for local server
):
    if chunk.event == "messages":
        message_type, message_data = chunk.data
        if message_type == "AIMessageChunk":
            print(message_data["content"], end="", flush=True)
```

---

## Background Runs — Fire-and-Forget

```python
# POST /threads/{thread_id}/runs — returns immediately
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input={"document_url": "https://example.com/large-doc.pdf"},
    webhook="https://your-service.com/webhook/run-complete",  # Called on completion
)

run_id = run["run_id"]
status = run["status"]  # "pending"

# Check status later
run_status = client.runs.get(thread_id, run_id)
print(run_status["status"])  # pending | running | success | error
```

---

## Cron Jobs — Scheduled Runs

```python
# Run every hour
cron = client.runs.create_cron(
    assistant_id="agent",
    schedule="0 * * * *",           # Standard cron syntax
    input={"task": "hourly_report"},
    # No thread_id → creates a new thread each run
)

# Run every day at 9am UTC on a specific thread
cron = client.runs.create_cron(
    thread_id=thread_id,
    assistant_id="agent",
    schedule="0 9 * * *",
    input={"task": "daily_summary"},
)

# Cancel
client.runs.delete_cron(cron["cron_id"])
```

---

## Double-Texting — Handling Concurrent Requests

**Problem**: User sends a message while a run is still in progress on the same thread.

LangGraph Platform offers 4 strategies (set via `multitask_strategy`):

```python
# Strategy 1: reject — return 409 error immediately
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    multitask_strategy="reject",
)

# Strategy 2: interrupt — cancel running run, start new one
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    multitask_strategy="interrupt",
)

# Strategy 3: rollback — cancel running run + rollback its state changes
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    multitask_strategy="rollback",
)

# Strategy 4: enqueue — queue new run after current completes
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    multitask_strategy="enqueue",
)
```

| Strategy | Use Case |
|---|---|
| `reject` | Strict ordering; user must wait |
| `interrupt` | Streaming chat (cancel stale response) |
| `rollback` | Need clean state before new run |
| `enqueue` | Batch processing, ordered queue |

---

## Human-in-the-Loop via Platform API

```python
# 1. Start run — will interrupt when it hits interrupt()
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input={"messages": [...]},
)

# 2. Poll or use webhook to detect interruption
state = client.threads.get_state(thread_id)
pending_interrupt = state["next"]    # Non-empty when interrupted

# 3. Resume with Command(resume=...)
client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=Command(resume={"decision": "approve"}),  # Pass Command as input
    multitask_strategy="enqueue",
)
```

---

## Webhooks

```python
# Webhook called when run completes (success or error)
run = client.runs.create(
    thread_id=thread_id,
    assistant_id="agent",
    input=input_data,
    webhook="https://your-backend.com/api/run-complete",
)

# Webhook payload example:
# {
#   "run_id": "...",
#   "thread_id": "...",
#   "status": "success",
#   "result": {...}
# }
```

---

## Production Deployment — Self-Hosted

```bash
# Build Docker image
langgraph build -t my-agent:latest

# Run with Docker Compose (includes Redis queue + Postgres)
# Uses official LangGraph Platform docker-compose template
docker compose up
```

Required services:
- **Redis** — task queue for background runs
- **Postgres** — persistent checkpointer + store

Set in environment:
```
LANGSMITH_API_KEY=...
REDIS_URI=redis://localhost:6379
POSTGRES_URI=postgresql://user:pass@localhost/db
```

---

## Related Topics
- [`04-persistence-checkpointing.md`](./04-persistence-checkpointing.md) — Checkpointing concepts
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — HITL with platform
- [`06-streaming.md`](./06-streaming.md) — Stream modes
- [`10-memory-store.md`](./10-memory-store.md) — Cross-thread store in production
