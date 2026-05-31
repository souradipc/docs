# Routing & Path Operations

## What Are Path Operations?

A **path operation** is a Python function decorated with an HTTP method decorator that maps a URL path to a handler. It is the fundamental unit of a FastAPI application.

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")   # ← path operation decorator
async def read_item(item_id: int):  # ← path operation function
    return {"item_id": item_id}
```

The decorator `@app.get(...)` tells FastAPI:
1. The **path**: `/items/{item_id}`
2. The **HTTP method**: `GET`
3. The **handler function**: `read_item`

---

## HTTP Method Decorators

FastAPI supports all standard HTTP methods:

```python
@app.get("/items")        # Retrieve resources
@app.post("/items")       # Create resources
@app.put("/items/{id}")   # Full update (replace)
@app.patch("/items/{id}") # Partial update
@app.delete("/items/{id}")# Delete resources
@app.head("/items")       # Like GET but no body
@app.options("/items")    # Describe communication options
@app.trace("/items")      # Diagnostic loop-back
```

### REST Semantics Best Practices

| Method | Idempotent | Side Effects | Typical Use |
|---|---|---|---|
| GET | ✅ | None | Fetch resource |
| POST | ❌ | Creates resource | Submit data |
| PUT | ✅ | Replaces resource | Full update |
| PATCH | ❌ | Modifies resource | Partial update |
| DELETE | ✅ | Removes resource | Delete |

> **Production rule**: Always follow REST semantics. GET requests must never have side effects — this breaks caching layers and monitoring tools.

---

## Path Parameters

Path parameters are declared using `{variable_name}` in the path and a matching function argument:

```python
@app.get("/users/{user_id}/posts/{post_id}")
async def read_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}
```

### Type Validation

FastAPI uses Pydantic to validate path parameters. Invalid types return `422 Unprocessable Entity`:

```
GET /users/abc/posts/1
→ 422: {"detail": [{"type": "int_parsing", "loc": ["path", "user_id"], ...}]}
```

### Enum Path Parameters

Restrict allowed values using Python `Enum`:

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model": model_name, "message": "Deep Learning FTW!"}
    return {"model": model_name}
```

---

## Path Operation Order Matters

FastAPI evaluates routes **in the order they are defined**. More specific paths must come before generic ones:

```python
# ✅ CORRECT: specific route defined first
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

```python
# ❌ WRONG: /users/me would match /users/{user_id} with user_id="me"
@app.get("/users/{user_id}")
async def read_user(user_id: str): ...

@app.get("/users/me")  # This will NEVER be reached
async def read_user_me(): ...
```

---

## Path Operation Configuration

The decorator accepts many optional parameters to control behavior and documentation:

```python
@app.post(
    "/items/",
    response_model=ItemResponse,
    status_code=201,
    tags=["items"],
    summary="Create a new item",
    description="Creates a new item with the provided data. Returns the created item.",
    response_description="The newly created item",
    deprecated=False,
    operation_id="create_item_v1",
    responses={
        409: {"description": "Item already exists"},
        422: {"description": "Validation Error"},
    }
)
async def create_item(item: ItemCreate):
    ...
```

### `tags` — API Organization

Tags group endpoints in the Swagger UI. Use them to organize by domain:

```python
@app.get("/users/", tags=["users"])
@app.get("/items/", tags=["items"])
@app.post("/payments/", tags=["payments", "billing"])
```

### `status_code` — Response Code

Default is `200`. Change it for semantic correctness:

```python
@app.post("/items/", status_code=201)      # Created
@app.delete("/items/{id}", status_code=204) # No Content
```

You can also use `fastapi.status` constants:

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
```

### `deprecated` — Mark Old Endpoints

```python
@app.get("/old-endpoint", deprecated=True)
async def old_endpoint():
    return {"message": "Use /new-endpoint instead"}
```

---

## APIRouter — Modular Routing

For large applications, use `APIRouter` to split routes across files:

```python
# app/routers/items.py
from fastapi import APIRouter

router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def list_items():
    return []

@router.get("/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id}

@router.post("/", status_code=201)
async def create_item():
    return {}
```

```python
# app/main.py
from fastapi import FastAPI
from app.routers import items, users

app = FastAPI()

app.include_router(items.router)
app.include_router(users.router, prefix="/api/v1")
```

### `include_router` Override Options

```python
app.include_router(
    items.router,
    prefix="/v2/items",         # Override prefix
    tags=["items-v2"],          # Override tags
    dependencies=[Depends(verify_token)],  # Apply to all routes
    responses={418: {"description": "I'm a teapot"}},
)
```

---

## Path Priority in `include_router`

When multiple routers have overlapping paths, the first included router wins. Be deliberate about registration order.

---

## Path Converters

FastAPI (via Starlette) supports path converters for more control:

```python
# Default: matches any string except "/"
@app.get("/items/{item_id}")

# Match any path including slashes
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
# GET /files/home/user/documents/report.pdf
# → {"file_path": "home/user/documents/report.pdf"}
```

---

## Returning Responses

Path operations can return several types:

```python
# 1. Dict — auto-serialized to JSON
@app.get("/dict")
async def return_dict():
    return {"key": "value"}

# 2. Pydantic model — serialized + validated
@app.get("/model", response_model=ItemResponse)
async def return_model():
    return ItemResponse(name="foo", price=10.0)

# 3. Explicit Response object — full control
from fastapi.responses import JSONResponse, PlainTextResponse
@app.get("/custom")
async def return_custom():
    return JSONResponse(content={"msg": "ok"}, status_code=200)

# 4. None / 204 No Content
@app.delete("/items/{id}", status_code=204)
async def delete_item(id: int):
    return None
```

---

## Common Pitfalls

### 1. Route Shadowing
```python
# Bug: /items/count matches /items/{item_id} with item_id="count"
@app.get("/items/{item_id}")  # defined first
@app.get("/items/count")      # never reached
```

### 2. Missing `async` for blocking code
```python
# This blocks the event loop — BAD
@app.get("/slow")
async def slow_endpoint():
    time.sleep(5)  # blocks all concurrent requests!
```
Use `def` (runs in threadpool) or proper `await`:
```python
@app.get("/slow")
def slow_endpoint():  # FastAPI runs this in a threadpool
    time.sleep(5)     # safe — doesn't block event loop
```

### 3. Mutable Default Arguments
```python
# BUG: shared mutable default
@app.get("/items")
async def get_items(filters: dict = {}):  # same dict reused
    ...

# CORRECT: use None and create in function body
@app.get("/items")
async def get_items(filters: dict | None = None):
    if filters is None:
        filters = {}
```

---

## Best Practices

1. **Group routes by resource domain** using `APIRouter` with `prefix`
2. **Always declare `status_code`** explicitly — don't rely on the default 200
3. **Use `tags`** to organize endpoints in Swagger UI
4. **Add docstrings** to path operation functions — they become API descriptions
5. **Keep path operation functions thin** — delegate to service layer
6. **Use `response_model`** to explicitly define and filter response shapes
7. **Version your API** via path prefix (`/v1/`, `/v2/`) or custom headers

```python
# Production-style path operation
@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=ItemResponse,
    responses={409: {"model": ErrorResponse}},
    summary="Create item",
)
async def create_item(
    item: ItemCreate,
    db: SessionDep,
    current_user: CurrentUserDep,
) -> ItemResponse:
    """
    Create a new item with the given data.

    Requires authentication. Returns the created item with generated ID.
    """
    return await item_service.create(db, item, owner=current_user)
```

---

## Related Topics
- [`02-parameters.md`](./02-parameters.md) — Path, query, header params
- [`04-response-models.md`](./04-response-models.md) — response_model, serialization
- [`13-app-structure-routers.md`](./13-app-structure-routers.md) — APIRouter architecture patterns
