# FastAPI Overview — Architecture, Philosophy & Core Concepts

## What is FastAPI?

FastAPI is a modern, high-performance Python web framework for building APIs. It is built on top of **Starlette** (the ASGI web layer) and **Pydantic** (for data validation and serialization). It was designed from the ground up to be:

- **Fast to run**: Performance comparable to NodeJS and Go (powered by Starlette + uvicorn)
- **Fast to develop**: Fewer bugs due to type safety; editor autocompletion everywhere
- **Standards-based**: Fully compliant with OpenAPI and JSON Schema
- **Production-ready**: Battle-tested in large-scale systems at companies like Microsoft, Uber, Netflix

---

## Why FastAPI Exists

Before FastAPI, Python developers had two main choices:

| Framework | Pros | Cons |
|---|---|---|
| Django REST Framework | Full-featured, mature | Verbose, sync-first, heavy |
| Flask | Lightweight, flexible | No async, no validation, no docs |

FastAPI fills a gap: **lightweight + async + automatic validation + automatic docs**.

It was created by **Sebastián Ramírez (tiangolo)** to address the reality that modern APIs need:
1. **Async support** — to handle thousands of concurrent requests efficiently
2. **Type-safe request/response handling** — avoiding runtime errors
3. **Automatic documentation** — not a nice-to-have but a requirement in professional settings
4. **Dependency injection** — for clean, testable code

---

## Core Architecture

```
Client Request
      │
      ▼
   uvicorn / hypercorn (ASGI Server)
      │
      ▼
   Starlette (Routing, Middleware, WebSockets, Static Files)
      │
      ▼
   FastAPI (Path Operations, Dependency Injection, OpenAPI generation)
      │
      ▼
   Pydantic (Validation, Serialization, Schema generation)
      │
      ▼
   Your Business Logic
```

### Key Components

| Layer | Library | Responsibility |
|---|---|---|
| ASGI server | uvicorn / hypercorn | Process HTTP/WebSocket, call ASGI app |
| Web layer | Starlette | Routing, middleware, requests, responses |
| API layer | FastAPI | Decorators, DI, OpenAPI, type injection |
| Validation | Pydantic v2 | Schema validation, serialization, config |

---

## ASGI vs WSGI

FastAPI is **ASGI-first** (Asynchronous Server Gateway Interface). This is fundamentally different from WSGI (used by Django and Flask).

| | WSGI | ASGI |
|---|---|---|
| Concurrency model | Synchronous, one request at a time per worker | Asynchronous, many concurrent requests per worker |
| WebSocket support | Not native | First-class citizen |
| Performance | Limited by I/O blocking | Can handle 10,000+ concurrent connections |
| Python requirement | Any Python | Python 3.7+ |

> **Key insight**: With ASGI, a single process can handle many simultaneous requests during I/O wait (e.g., DB queries, HTTP calls). You don't need 100 workers to handle 100 concurrent requests.

---

## OpenAPI Integration

FastAPI **automatically generates an OpenAPI schema** from your code — no manual YAML or JSON needed.

It does this by introspecting:
- Path operation decorators (`@app.get`, `@app.post`, etc.)
- Python type hints on parameters
- Pydantic models used as request/response types
- Docstrings

This schema is served at `/openapi.json` and powers:
- **Swagger UI** → `/docs`
- **ReDoc** → `/redoc`

```python
from fastapi import FastAPI

app = FastAPI(
    title="My Production API",
    description="Manages widgets and sprockets",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
```

---

## Pydantic Integration

Pydantic v2 is deeply integrated into FastAPI. It handles:

1. **Request validation** — invalid inputs return a 422 Unprocessable Entity automatically
2. **Response serialization** — your data is serialized per the `response_model`
3. **Settings management** — via `pydantic-settings`
4. **Schema generation** — feeds the OpenAPI spec

```python
from pydantic import BaseModel, Field
from typing import Optional

class Item(BaseModel):
    name: str
    price: float = Field(gt=0, description="Must be positive")
    description: Optional[str] = None
    tax: float = 0.0
```

> **Performance note**: Pydantic v2 is written in Rust and is 5–50x faster than v1 for validation and serialization.

---

## Type Hints as the Central Mechanism

FastAPI leverages Python's type hint system (`typing` module, PEP 484, 526) as its primary source of truth. Every parameter, return type, and model field is declared with types.

This single decision gives FastAPI:
- **Validation** — Pydantic enforces types
- **Documentation** — OpenAPI schema from type info
- **Editor support** — Mypy/Pyright check your code
- **IDE autocompletion** — Your IDE knows the exact shape of every object

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None) -> dict:
    ...
```

FastAPI reads this and knows:
- `item_id` is a **path parameter** (because it's in the path) and must be an `int`
- `q` is an optional **query parameter** (singular type, not in path, has default `None`)
- The response is a dict (used in OpenAPI docs)

---

## The Request Lifecycle

```
1. HTTP request arrives at uvicorn
2. Starlette parses headers, path, query string, body
3. FastAPI matches the path to a route
4. FastAPI resolves all dependencies (Depends)
5. FastAPI extracts and validates path/query/header/body params (Pydantic)
6. Your path operation function executes
7. Return value is serialized through response_model (Pydantic)
8. HTTP response is sent back
9. (Optional) Background tasks run after response is sent
```

---

## When to Use FastAPI

✅ **Use FastAPI when:**
- Building REST APIs or GraphQL APIs (with Strawberry)
- You need async I/O performance (database queries, external API calls)
- You want auto-generated interactive docs
- You value type safety and IDE support
- Microservices or cloud-native backends

❌ **Don't use FastAPI when:**
- Building full-stack web apps with server-side HTML rendering (use Django)
- You need a fully-featured admin panel out of the box (use Django)
- Your team is unfamiliar with async Python (can cause subtle bugs)
- You need complex ORM relationships with Django-style migrations natively

---

## Production Readiness Checklist (Overview)

- [ ] Use `pydantic-settings` for environment-based config
- [ ] Add health check endpoint (`/health`)
- [ ] Use structured logging (JSON format for log aggregators)
- [ ] Configure CORS for your actual origins — never `*` in production
- [ ] Add rate limiting (via middleware or reverse proxy)
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Use lifespan context manager for startup/shutdown
- [ ] Run behind a reverse proxy (nginx, Caddy, Traefik)
- [ ] Use Gunicorn + Uvicorn workers for multi-core deployments

---

## Key Concepts Map

```
FastAPI
├── Routing           → Path operations, APIRouter, include_router
├── Parameters        → Path, Query, Header, Cookie, Body
├── Validation        → Pydantic models, Field, validators
├── Dependency Injection → Depends(), yield dependencies
├── Security          → OAuth2, JWT, API keys, HTTP Basic
├── Middleware        → CORS, logging, auth, tracing
├── Error Handling    → HTTPException, custom handlers
├── Async             → async/await, concurrency, threadpool
├── Database          → SQLModel, SQLAlchemy, async sessions
├── WebSockets        → Real-time bidirectional communication
├── Background Tasks  → Post-response async work
├── Lifespan          → Startup/shutdown resource management
├── Testing           → TestClient, AsyncClient, dependency overrides
├── Settings          → pydantic-settings, .env files
└── Deployment        → Docker, uvicorn, gunicorn, scaling
```

---

## Related Topics
- [`01-routing-path-operations.md`](./01-routing-path-operations.md) — HTTP methods, routing rules
- [`05-dependency-injection.md`](./05-dependency-injection.md) — FastAPI's killer feature
- [`09-async-concurrency.md`](./09-async-concurrency.md) — When and how to use async
- [`16-deployment-production.md`](./16-deployment-production.md) — Running FastAPI in production
