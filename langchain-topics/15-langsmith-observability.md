# LangSmith — Observability & Tracing

## What Is LangSmith?

LangSmith is the observability and evaluation platform for LangChain applications. It provides:

- **Tracing** — Full visibility into every LLM call, tool invocation, chain step
- **Debugging** — Inspect inputs/outputs at every level of a complex chain
- **Datasets** — Build golden datasets for systematic evaluation
- **Evaluation** — Run automated experiments against datasets
- **Monitoring** — Track latency, token usage, cost, error rates in production
- **Prompt Playground** — Test and iterate on prompts interactively

LangSmith is a SaaS platform (cloud-hosted or self-hosted).

---

## Setup

```bash
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls_...
export LANGCHAIN_PROJECT=my-project   # Optional: group traces by project
```

Once tracing is enabled, **all LangChain operations are automatically traced** — no code changes needed.

---

## Automatic Tracing

With the environment variables set, every chain, model call, and tool invocation is traced automatically:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# All of this is automatically traced to LangSmith
model = ChatOpenAI(model="gpt-4o")
chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | model
)
result = chain.invoke({"question": "What is phishing?"})

# In LangSmith, you'll see:
# - Chain run: "RunnableSequence"
#   - Step 1: "ChatPromptTemplate" (input → formatted messages)
#   - Step 2: "ChatOpenAI" (messages → response)
#     - Input tokens: 25, Output tokens: 150, Latency: 1.2s
```

---

## `@traceable` — Trace Any Python Function

```python
from langsmith import traceable

@traceable
def run_threat_analysis(ioc: str) -> dict:
    """This function will appear in LangSmith as a named run."""
    intel = get_threat_intel.invoke(ioc)
    analysis = model.invoke(f"Analyze this threat: {intel}")
    return {
        "ioc": ioc,
        "intel": intel,
        "analysis": analysis.content,
    }

# Call it normally — trace appears in LangSmith
result = run_threat_analysis("192.168.1.100")
```

### `@traceable` with Metadata

```python
@traceable(
    name="threat_pipeline",           # Custom name in LangSmith
    tags=["production", "v2"],        # Tags for filtering
    metadata={"version": "2.0"},      # Extra metadata
    run_type="chain",                 # "chain" | "llm" | "tool" | "retriever"
)
def threat_pipeline(ioc: str) -> str:
    return analyze(ioc)
```

---

## `wrap_openai` — Trace Direct OpenAI Calls

If you use the OpenAI SDK directly (not through LangChain), wrap it:

```python
from langsmith import wrap_openai
from openai import OpenAI

client = wrap_openai(OpenAI())

# This call now appears in LangSmith traces
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Run Context and Metadata

```python
from langsmith.run_helpers import get_current_run_tree

@traceable
def my_pipeline(input: str) -> str:
    # Get current trace context
    run_tree = get_current_run_tree()
    if run_tree:
        run_id = run_tree.id
        # Store run_id to attach feedback later

    return model.invoke(input).content

# Tag runs with user metadata
result = chain.invoke(
    {"question": "What is malware?"},
    config={
        "run_name": "threat-q-answer",
        "tags": ["user-facing", "production"],
        "metadata": {
            "user_id": "alice123",
            "session_id": "sess-456",
            "environment": "production",
        },
    },
)
```

---

## Attaching Feedback

Capture user ratings and attach them to traces for online evaluation:

```python
from langsmith import Client

client = Client()

@app.post("/feedback")
async def submit_feedback(run_id: str, is_helpful: bool, comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=1.0 if is_helpful else 0.0,
        comment=comment,
        source_info={"user_agent": "webapp"},
    )
```

---

## Projects and Environments

Organize traces by project and environment:

```python
import os

# Different projects for different use cases
os.environ["LANGCHAIN_PROJECT"] = "production-agent"   # Production
os.environ["LANGCHAIN_PROJECT"] = "staging-agent"      # Staging
os.environ["LANGCHAIN_PROJECT"] = "dev-alice"          # Developer sandbox

# Or set per-invocation
result = chain.invoke(
    {"question": "Hello"},
    config={"tags": ["staging"]},
)
```

---

## Datasets in LangSmith

Build golden evaluation datasets from:

### From Code

```python
from langsmith import Client

client = Client()
dataset = client.create_dataset("security-qa-eval")

client.create_examples(
    inputs=[
        {"question": "What is a zero-day exploit?"},
        {"question": "How does ransomware work?"},
    ],
    outputs=[
        {"answer": "A zero-day exploit targets an unpatched vulnerability..."},
        {"answer": "Ransomware encrypts victim files and demands payment..."},
    ],
    dataset_id=dataset.id,
)
```

### From Production Traces

```python
# Capture interesting/problematic runs from production
client.create_examples_from_runs(
    run_ids=["run-id-1", "run-id-2"],  # IDs from LangSmith
    dataset_id=dataset.id,
)
```

---

## Running Experiments

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

def my_app(inputs: dict) -> dict:
    return {"answer": chain.invoke(inputs["question"])}

results = evaluate(
    my_app,
    data="security-qa-eval",
    evaluators=[
        LangChainStringEvaluator("qa"),           # Answer correctness
        LangChainStringEvaluator("conciseness"),  # Response conciseness
    ],
    experiment_prefix="chain-v2",
    metadata={"model": "gpt-4o", "prompt_version": "v2"},
)

# View in LangSmith UI: compare experiments side by side
print(results.to_pandas())
```

---

## Monitoring in Production

Key metrics LangSmith tracks automatically:

| Metric | Description |
|---|---|
| **Latency (p50/p95/p99)** | Response time distribution |
| **Token Usage** | Input + output tokens per run |
| **Cost** | Estimated API cost per run |
| **Error Rate** | % of runs that errored |
| **Feedback Score** | Average user rating |
| **Trace Volume** | Number of runs over time |

```python
# Query metrics via API
from langsmith import Client

client = Client()

# Get runs from the last 24 hours
runs = list(client.list_runs(
    project_name="production-agent",
    start_time=datetime.now() - timedelta(hours=24),
    error=False,  # Only successful runs
    run_type="chain",
))

# Compute average latency
latencies = [
    (r.end_time - r.start_time).total_seconds()
    for r in runs
    if r.end_time and r.start_time
]
print(f"Mean latency: {sum(latencies)/len(latencies):.2f}s")
```

---

## Prompt Playground

In the LangSmith UI:
1. Open any run → click "Open in Playground"
2. Modify the prompt or model parameters
3. Run again to compare outputs side by side
4. Save the improved prompt as a hub template

---

## Self-Hosted LangSmith

For air-gapped or compliance-sensitive environments:

```bash
# Docker Compose deployment
docker compose up langsmith

# Point to self-hosted
export LANGCHAIN_ENDPOINT=https://your-langsmith.internal.com
export LANGCHAIN_API_KEY=your-key
```

---

## Common Pitfalls

### 1. Tracing in Test Environments
```python
# ❌ Tests send traces to production project
os.environ["LANGCHAIN_PROJECT"] = "production"

# ✅ Use separate project for CI/dev or disable tracing
os.environ["LANGCHAIN_PROJECT"] = "ci-tests"
# OR disable entirely in unit tests:
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

### 2. Leaking Sensitive Data in Traces
```python
# ❌ Full user messages including PII traced
chain.invoke({"user_message": "My SSN is 123-45-6789"})
# → Visible in LangSmith!

# ✅ Sanitize before tracing, or configure what to hide
# LangSmith supports metadata-only tracing (hide inputs/outputs)
chain.invoke(
    {"user_message": sanitize(user_message)},
    config={"metadata": {"has_pii": False}},
)
```

### 3. High Volume Sampling
```python
# ❌ Tracing every request at 10K req/s → high cost
# ✅ Sample traces in production
import random

def should_trace() -> bool:
    return random.random() < 0.1  # 10% sampling

if should_trace():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

---

## Related Topics
- [`13-callbacks-streaming.md`](./13-callbacks-streaming.md) — Callbacks and observability hooks
- [`14-evaluation-testing.md`](./14-evaluation-testing.md) — Evaluation using LangSmith datasets
- [`16-production-deployment.md`](./16-production-deployment.md) — Monitoring in production
