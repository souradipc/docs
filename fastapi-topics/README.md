# FastAPI Knowledge Base

A comprehensive, production-focused reference for FastAPI — organized by topic. Each document covers the **what**, **why**, **how**, **when**, **pitfalls**, and **best practices** of every major concept.

---

## Document Index

| # | Topic | Key Concepts |
|---|---|---|
| [00](./00-overview.md) | **Overview & Architecture** | ASGI, Starlette, Pydantic, request lifecycle, OpenAPI |
| [01](./01-routing-path-operations.md) | **Routing & Path Operations** | HTTP methods, path params, ordering, APIRouter, status codes |
| [02](./02-parameters.md) | **Parameters** | Path, Query, Header, Cookie, Annotated, validation constraints |
| [03](./03-request-body-pydantic.md) | **Request Body & Pydantic** | BaseModel, Field, nested models, validators, extra="forbid" |
| [04](./04-response-models.md) | **Response Models** | response_model, filtering, serialization, ORJSONResponse |
| [05](./05-dependency-injection.md) | **Dependency Injection** | Depends, yield, caching, scoped deps, class-based deps |
| [06](./06-security-authentication.md) | **Security & Authentication** | OAuth2, JWT, RBAC, API keys, password hashing |
| [07](./07-middleware-cors.md) | **Middleware & CORS** | Custom middleware, CORS config, GZip, TrustedHost |
| [08](./08-error-handling.md) | **Error Handling** | HTTPException, custom handlers, validation errors, RFC 7807 |
| [09](./09-async-concurrency.md) | **Async & Concurrency** | async/await, event loop, gather, anyio, threadpool |
| [10](./10-database-integration.md) | **Database Integration** | SQLModel, SQLAlchemy, async DB, CRUD, Alembic |
| [11](./11-websockets-streaming.md) | **WebSockets & Streaming** | WebSocket, StreamingResponse, SSE, LLM streaming |
| [12](./12-background-tasks-lifespan.md) | **Background Tasks & Lifespan** | BackgroundTasks, lifespan context, startup/shutdown |
| [13](./13-app-structure-routers.md) | **App Structure & Routers** | APIRouter, project layout, service layer, versioning |
| [14](./14-settings-configuration.md) | **Settings & Configuration** | pydantic-settings, .env, secrets, multi-environment |
| [15](./15-testing.md) | **Testing** | TestClient, AsyncClient, dependency_overrides, conftest |
| [16](./16-deployment-production.md) | **Deployment & Production** | Docker, Gunicorn, Uvicorn workers, nginx, Kubernetes |
| [17](./17-openapi-docs.md) | **OpenAPI & Documentation** | Schema customization, examples, tags, Swagger UI |

---

## Quick Reference: Common Patterns

### Minimal Production App

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
)

app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS, ...)
app.include_router(api_router, prefix=settings.API_V1_STR)
```

### Standard Endpoint

```python
@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    responses={404: {"model": NotFoundError}},
    summary="Get item by ID",
)
async def get_item(
    item_id: int,
    db: SessionDep,
    current_user: CurrentUser,
) -> ItemResponse:
    """Retrieve a single item. Returns 404 if not found."""
    item = await item_service.get(db, item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item
```

### Clean Dependency Stack

```python
# deps.py
SessionDep     = Annotated[Session, Depends(get_session)]
CurrentUser    = Annotated[User, Depends(get_current_active_user)]
AdminUser      = Annotated[User, Depends(require_admin)]
SettingsDep    = Annotated[Settings, Depends(get_settings)]
```

---

## Architecture Decision Quick Guide

| Question | Answer | Doc |
|---|---|---|
| `async def` or `def`? | async for async I/O, def for sync libs | [09](./09-async-concurrency.md) |
| SQLModel or SQLAlchemy? | SQLModel for FastAPI-native apps | [10](./10-database-integration.md) |
| Auth: JWT or sessions? | JWT for stateless APIs, sessions for browser apps | [06](./06-security-authentication.md) |
| BackgroundTasks or Celery? | BackgroundTasks for best-effort, Celery for reliable | [12](./12-background-tasks-lifespan.md) |
| 1 worker or many? | Always multiple workers in production | [16](./16-deployment-production.md) |
| `response_model` or return type? | Both; `response_model` for full control | [04](./04-response-models.md) |
| One big file or modules? | APIRouter modules always | [13](./13-app-structure-routers.md) |

---

## Key Libraries Ecosystem

| Purpose | Library |
|---|---|
| Core framework | `fastapi` |
| ASGI server | `uvicorn[standard]` |
| Process manager | `gunicorn` |
| Data validation | `pydantic` (v2) |
| ORM | `sqlmodel` or `sqlalchemy` |
| Async DB driver | `asyncpg` (PostgreSQL) |
| Settings | `pydantic-settings` |
| JWT | `python-jose` or `pyjwt` |
| Password hashing | `passlib[bcrypt]` |
| Async HTTP | `httpx` |
| Testing | `pytest`, `pytest-asyncio` |
| Migrations | `alembic` |
| Async file I/O | `aiofiles` |
| Redis | `redis[hiredis]` |
| Observability | `opentelemetry-instrumentation-fastapi` |

---

*Generated from official FastAPI documentation — https://fastapi.tiangolo.com*
