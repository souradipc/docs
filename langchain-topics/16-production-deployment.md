# Production Deployment

## Production Readiness Checklist

Before deploying a LangChain application:

- [ ] Model API keys stored in secrets manager (not env files in prod)
- [ ] LangSmith tracing configured for the production project
- [ ] Rate limiting and retry logic configured
- [ ] Async patterns throughout (no blocking calls in async paths)
- [ ] Error handling for API failures and model errors
- [ ] Token budget + cost controls set
- [ ] Context window limits respected
- [ ] Prompt injection defenses in place
- [ ] Sensitive data not logged in traces
- [ ] Health check endpoint deployed

---

## LangServe — Serving Chains as APIs

LangServe wraps any LangChain chain as a REST API automatically:

```python
# pip install "langserve[all]"
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="My LangChain API")

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

# Automatically creates: /chain/invoke, /chain/stream, /chain/batch
add_routes(app, chain, path="/chain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

LangServe auto-generates:
- `POST /chain/invoke` — single invocation
- `POST /chain/stream` — SSE streaming
- `POST /chain/batch` — batch processing
- `GET /chain/playground` — interactive UI
- `GET /chain/openapi.json` — OpenAPI schema

---

## FastAPI Integration (Without LangServe)

For more control, integrate LangChain directly into FastAPI:

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
import json

app = FastAPI()

# Initialize chain on startup (not per-request)
chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = await chain.ainvoke({"question": request.question})
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream(request: QueryRequest):
    async def generate():
        try:
            async for token in chain.astream({"question": request.question}):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gpt-4o"}
```

---

## Async — Critical for Production

```python
# ❌ Blocking call inside async handler — blocks the event loop
@app.post("/query")
async def query(request: QueryRequest):
    result = chain.invoke({"question": request.question})  # BLOCKING
    return {"answer": result}

# ✅ Always use async variants in async handlers
@app.post("/query")
async def query(request: QueryRequest):
    result = await chain.ainvoke({"question": request.question})  # Non-blocking
    return {"answer": result}

# ✅ Concurrent model calls
import asyncio

async def batch_analyze(questions: list[str]) -> list[str]:
    tasks = [chain.ainvoke({"question": q}) for q in questions]
    results = await asyncio.gather(*tasks)  # All in parallel
    return results
```

---

## Rate Limiting

```python
# Dependency-based rate limiting
from fastapi import Request, HTTPException
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str):
        now = time.time()
        # Clean old requests
        self.requests[client_id] = [
            r for r in self.requests[client_id]
            if r > now - self.window
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        self.requests[client_id].append(now)

limiter = RateLimiter(max_requests=10, window_seconds=60)

@app.post("/query")
async def query(request: QueryRequest, client_request: Request):
    client_id = client_request.client.host
    limiter.check(client_id)
    return {"answer": await chain.ainvoke({"question": request.question})}
```

---

## Caching Model Responses

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis

# Production: Redis cache shared across all workers
redis_client = redis.from_url("redis://localhost:6379")
set_llm_cache(RedisCache(redis_client))

# Or SQLite for single-instance production
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path="/app/data/llm_cache.db"))
```

---

## Error Handling

```python
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException
from openai import RateLimitError, APIConnectionError
import logging

logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4o", max_retries=3)

# Fallback chain
primary_model = ChatOpenAI(model="gpt-4o")
fallback_model = ChatOpenAI(model="gpt-4o-mini")
resilient_model = primary_model.with_fallbacks([fallback_model])

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = await resilient_model.ainvoke(
            [("human", request.question)]
        )
        return {"answer": result.content}
    except RateLimitError:
        raise HTTPException(status_code=429, detail="AI service at capacity")
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    except LangChainException as e:
        logger.error(f"LangChain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Processing error")
```

---

## Multi-Tenancy

Isolate different tenants (users/orgs) in the same deployment:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

class AgentService:
    def __init__(self, checkpointer):
        self.agent = create_react_agent(model, tools, checkpointer=checkpointer)

    def get_config(self, tenant_id: str, session_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": f"tenant:{tenant_id}:session:{session_id}",
            },
            "tags": [f"tenant:{tenant_id}"],
            "metadata": {"tenant_id": tenant_id},
        }

    async def query(self, tenant_id: str, session_id: str, message: str) -> str:
        config = self.get_config(tenant_id, session_id)
        result = await self.agent.ainvoke(
            {"messages": [("human", message)]},
            config=config,
        )
        return result["messages"][-1].content
```

---

## Token Budget and Cost Controls

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenBudgetCallback(BaseCallbackHandler):
    """Enforce token budget per request."""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        if self.total_tokens > self.max_tokens:
            raise ValueError(f"Token budget exceeded: {self.total_tokens} > {self.max_tokens}")

@app.post("/query")
async def query(request: QueryRequest):
    budget_callback = TokenBudgetCallback(max_tokens=2000)
    try:
        result = await chain.ainvoke(
            {"question": request.question},
            config={"callbacks": [budget_callback]},
        )
        return {"answer": result, "tokens_used": budget_callback.total_tokens}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## Deployment Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=production

# Optional tuning
LANGCHAIN_VERBOSE=false        # Disable verbose logging in prod
OPENAI_TIMEOUT=60              # Request timeout
OPENAI_MAX_RETRIES=3
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY app/ ./app/
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Health Checks

```python
@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}

@app.get("/health/deep")
async def deep_health():
    """Check AI service connectivity."""
    try:
        model = ChatOpenAI(model="gpt-4o-mini")
        await model.ainvoke([("human", "ping")])
        return {"status": "ok", "ai": "reachable"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service unreachable: {e}")
```

---

## Common Pitfalls

### 1. Initializing Clients Per Request
```python
# ❌ Creates new HTTP connections on every request
@app.post("/query")
async def query(request: QueryRequest):
    model = ChatOpenAI(model="gpt-4o")  # Initialized per request!
    return {"answer": await model.ainvoke(...)}

# ✅ Initialize once at startup
model = ChatOpenAI(model="gpt-4o")  # Module level or lifespan

@app.post("/query")
async def query(request: QueryRequest):
    return {"answer": await model.ainvoke(...)}
```

### 2. No Timeout Configuration
```python
# ❌ Requests can hang indefinitely
model = ChatOpenAI(model="gpt-4o")

# ✅ Set timeouts
model = ChatOpenAI(
    model="gpt-4o",
    timeout=30,         # 30 second timeout
    max_retries=2,
)
```

---

## Related Topics
- [`13-callbacks-streaming.md`](./13-callbacks-streaming.md) — Streaming in FastAPI
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — Production monitoring
- [`17-cost-optimization.md`](./17-cost-optimization.md) — Managing token costs
