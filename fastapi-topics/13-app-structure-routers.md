# App Structure & APIRouter

## The Problem with a Single File

As an application grows, a monolithic `main.py` becomes unmanageable:

```python
# ❌ Anti-pattern: everything in one file
app = FastAPI()

@app.get("/users/")
@app.post("/users/")
@app.get("/items/")
@app.post("/items/")
@app.get("/orders/")
# ... 200 more endpoints
```

FastAPI's `APIRouter` solves this by allowing you to split routes across files and modules while composing them in `main.py`.

---

## Recommended Project Structure

```
app/
├── main.py                    # FastAPI app instance + router registration
├── core/
│   ├── config.py              # Settings (pydantic-settings)
│   ├── database.py            # Engine, session factory
│   ├── security.py            # Auth utilities
│   └── logging.py             # Log configuration
├── models/                    # SQLAlchemy/SQLModel table models
│   ├── user.py
│   ├── item.py
│   └── __init__.py
├── schemas/                   # Pydantic request/response schemas
│   ├── user.py
│   ├── item.py
│   └── __init__.py
├── api/
│   ├── deps.py                # Shared dependencies (get_db, get_current_user)
│   └── v1/
│       ├── api.py             # v1 router that includes sub-routers
│       ├── users.py
│       ├── items.py
│       └── auth.py
├── services/                  # Business logic layer
│   ├── user_service.py
│   └── item_service.py
└── tests/
    ├── conftest.py
    └── api/
        ├── test_users.py
        └── test_items.py
```

---

## `APIRouter` Setup

### Router Definition (per module)

```python
# app/api/v1/items.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Annotated
from app.api.deps import SessionDep, CurrentUser
from app.schemas.item import ItemCreate, ItemUpdate, ItemResponse
from app.services import item_service

router = APIRouter(
    prefix="/items",            # All routes prefixed with /items
    tags=["items"],             # Swagger UI grouping
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    }
)

@router.get("/", response_model=list[ItemResponse])
async def list_items(
    session: SessionDep,
    skip: int = 0,
    limit: Annotated[int, Query(le=100)] = 20,
):
    return await item_service.list(session, skip=skip, limit=limit)

@router.post("/", response_model=ItemResponse, status_code=201)
async def create_item(
    item: ItemCreate,
    session: SessionDep,
    current_user: CurrentUser,
):
    return await item_service.create(session, item, owner_id=current_user.id)

@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, session: SessionDep):
    item = await item_service.get(session, item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item

@router.patch("/{item_id}", response_model=ItemResponse)
async def update_item(
    item_id: int,
    item_update: ItemUpdate,
    session: SessionDep,
    current_user: CurrentUser,
):
    return await item_service.update(session, item_id, item_update, current_user)

@router.delete("/{item_id}", status_code=204)
async def delete_item(
    item_id: int,
    session: SessionDep,
    current_user: CurrentUser,
):
    await item_service.delete(session, item_id, current_user)
```

---

### Aggregating Routers

```python
# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1 import users, items, auth

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
```

---

### Main App Assembly

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import create_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes — all prefixed with /api/v1
app.include_router(api_router, prefix=settings.API_V1_STR)
```

---

## Shared Dependencies Module

```python
# app/api/deps.py
from typing import Annotated
from fastapi import Depends, HTTPException, status
from sqlmodel import Session

from app.core.database import get_session
from app.core.security import decode_token
from app.models.user import User

# Session dependency
SessionDep = Annotated[Session, Depends(get_session)]

# Auth dependencies
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: Session = Depends(get_session),
) -> User:
    payload = decode_token(token)
    user = session.get(User, payload["sub"])
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user

async def get_current_active_user(
    user: User = Depends(get_current_user)
) -> User:
    if not user.is_active:
        raise HTTPException(400, "Inactive user")
    return user

async def get_admin_user(
    user: User = Depends(get_current_active_user)
) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admin access required")
    return user

# Type aliases for clean endpoint signatures
CurrentUser = Annotated[User, Depends(get_current_active_user)]
AdminUser = Annotated[User, Depends(get_admin_user)]
```

---

## API Versioning Patterns

### Pattern 1: URL Path Versioning (Most Common)

```python
# /api/v1/items, /api/v2/items
app.include_router(v1_router, prefix="/api/v1")
app.include_router(v2_router, prefix="/api/v2")
```

### Pattern 2: Header Versioning

```python
from fastapi import Header, APIRouter

def require_api_version(accept_version: str = Header(default="1")):
    if accept_version not in ["1", "2"]:
        raise HTTPException(400, f"Unsupported API version: {accept_version}")
    return accept_version

@router.get("/items/", dependencies=[Depends(require_api_version)])
async def list_items(version: str = Depends(require_api_version)):
    if version == "2":
        return items_service_v2.list()
    return items_service_v1.list()
```

### Pattern 3: Multiple `include_router` with Overrides

```python
# Reuse v1 router for v2 with different dependencies
app.include_router(
    items_v1_router,
    prefix="/api/v2/items",
    tags=["items-v2"],
    dependencies=[Depends(new_auth_scheme)],  # Different auth
)
```

---

## Router-Level Configuration

### Apply Dependencies to All Routes in a Router

```python
# All routes in admin_router require admin auth
admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_admin_user)],
)
```

### Apply Custom Responses to All Routes

```python
router = APIRouter(
    responses={
        404: {"description": "Not found", "model": NotFoundError},
        500: {"description": "Internal server error"},
    }
)
```

---

## Service Layer Pattern

Keep path operations thin — move business logic to services:

```python
# app/services/item_service.py
from sqlmodel import Session, select
from app.models.item import Item
from app.schemas.item import ItemCreate, ItemUpdate
from app.exceptions import NotFoundError, ForbiddenError

async def create(session: Session, data: ItemCreate, owner_id: int) -> Item:
    item = Item(**data.model_dump(), owner_id=owner_id)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item

async def get(session: Session, item_id: int) -> Item | None:
    return session.get(Item, item_id)

async def update(
    session: Session,
    item_id: int,
    data: ItemUpdate,
    user: User,
) -> Item:
    item = session.get(Item, item_id)
    if not item:
        raise NotFoundError("Item", item_id)
    if item.owner_id != user.id and not user.is_admin:
        raise ForbiddenError("Cannot update item owned by another user")

    update_data = data.model_dump(exclude_unset=True)
    item.sqlmodel_update(update_data)
    session.add(item)
    session.commit()
    session.refresh(item)
    return item
```

Benefits of this pattern:
- Services have **no FastAPI imports** — fully testable in isolation
- Path operations are **thin wrappers** that handle HTTP concerns only
- Business logic can be **reused** across different endpoints or transports

---

## `include_router` Options Reference

```python
app.include_router(
    router,
    prefix="/v2/items",              # Override or add prefix
    tags=["items-v2"],               # Override tags
    dependencies=[Depends(auth)],    # Add to all routes
    responses={418: {...}},          # Add to all routes' response docs
    include_in_schema=True,          # Show/hide in OpenAPI docs
    default_response_class=ORJSONResponse,  # Default response class
)
```

---

## Common Pitfalls

### 1. Circular Imports
```python
# ❌ Circular: main.py imports router.py, router.py imports main.py
# main.py
from app.api.items import router
app.include_router(router)

# items.py
from app.main import app  # ← circular!
```
**Solution**: Move shared state to `app.state` accessed via `request.app.state`.

### 2. Forgetting Prefix When Including
```python
# ❌ Missing prefix — all item routes end up at root
app.include_router(items.router)  # GET / instead of GET /items/

# ✅
app.include_router(items.router)  # items.router already has prefix="/items"
# OR
app.include_router(items.router, prefix="/items")  # Add prefix here
```

### 3. Tag Duplication
```python
# ❌ Tags set in both router and include_router — results in duplicate tags
router = APIRouter(tags=["items"])
app.include_router(router, tags=["items"])  # "items" appears twice in docs

# ✅ Set tags in only one place
router = APIRouter()
app.include_router(router, prefix="/items", tags=["items"])
```

---

## Best Practices

1. **One router per domain** — users, items, orders, auth each in their own file
2. **Shared deps in `deps.py`** — don't repeat dependency definitions
3. **Service layer** — keep business logic out of path operation functions
4. **Version via prefix** — `/api/v1/` for all v1 endpoints
5. **Router-level deps** — apply auth at router level, not per-endpoint
6. **Keep `main.py` minimal** — just app creation, middleware, router registration

---

## Related Topics
- [`05-dependency-injection.md`](./05-dependency-injection.md) — Shared dependencies
- [`14-settings-configuration.md`](./14-settings-configuration.md) — Config per environment
- [`15-testing.md`](./15-testing.md) — Testing modular apps
