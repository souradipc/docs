# Response Models & Serialization

## What is `response_model`?

`response_model` tells FastAPI the **exact shape** of the data to return from an endpoint. It serves three purposes:

1. **Filtering** — strips fields not in the model (e.g., hides `hashed_password`)
2. **Validation** — ensures the returned data conforms to the schema
3. **Documentation** — generates the response schema in OpenAPI / Swagger UI

```python
from pydantic import BaseModel
from fastapi import FastAPI

class UserCreate(BaseModel):
    username: str
    password: str         # sensitive — should NOT be returned

class UserResponse(BaseModel):
    id: int
    username: str         # safe to return

app = FastAPI()

@app.post("/users/", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    # Even if this dict contains "password", it won't appear in the response
    return {"id": 1, "username": user.username, "password": user.password}
```

The `password` field is silently excluded because it's not in `UserResponse`.

---

## Declaring Response Model

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 0.0
    tags: list[str] = []

# The response type annotation also works (recommended in modern FastAPI):
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int) -> Item:
    return Item(name="Widget", price=9.99)
```

---

## Return Type Annotation vs `response_model`

FastAPI supports declaring the response type via Python's return annotation:

```python
# Via return annotation — cleaner, type-checker friendly
@app.get("/items/{item_id}")
async def read_item(item_id: int) -> Item:
    ...

# Via response_model parameter — more flexible (allows non-Pydantic returns)
@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    ...
```

**Key difference**: Use `response_model=` when the return type annotation can't express the full response type (e.g., you return `Any` internally but want a specific schema in the docs).

---

## Filtering Response Fields

### `response_model_exclude_unset`

Returns only the fields that were explicitly set, ignoring defaults:

```python
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 10.5
    tags: list[str] = []

@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: int):
    # Simulate a DB object with only some fields set
    return {"name": "Widget", "price": 9.99}
    # Response: {"name": "Widget", "price": 9.99}
    # NOT: {"name": "Widget", "description": null, "price": 9.99, "tax": 10.5, "tags": []}
```

> **Use case**: Storing sparse data in DB and returning only the stored fields without padding defaults.

### `response_model_exclude_none`

Exclude fields whose value is `None`:

```python
@app.get("/items/{item_id}", response_model=Item, response_model_exclude_none=True)
async def read_item(item_id: int):
    return Item(name="Widget", price=9.99, description=None)
    # Response: {"name": "Widget", "price": 9.99}
    # NOT: {"name": "Widget", "description": null, ...}
```

### `response_model_include` and `response_model_exclude`

```python
@app.get("/items/{item_id}", response_model=Item, response_model_include={"name", "price"})
async def read_item(item_id: int):
    ...
# Only returns: {"name": ..., "price": ...}

@app.get("/items/{item_id}", response_model=Item, response_model_exclude={"tax", "tags"})
async def read_item(item_id: int):
    ...
# Returns everything except "tax" and "tags"
```

---

## Response with Multiple Types

Use a Union type for endpoints that can return different response shapes:

```python
from typing import Union

class Cat(BaseModel):
    name: str
    meows: bool

class Dog(BaseModel):
    name: str
    barks: bool

@app.get("/animals/{animal_id}", response_model=Union[Cat, Dog])
async def get_animal(animal_id: int):
    if animal_id == 1:
        return Cat(name="Kitty", meows=True)
    return Dog(name="Rex", barks=True)
```

> **OpenAPI note**: Union types in responses are rendered as `anyOf` in the schema.

---

## List Responses

```python
@app.get("/items/", response_model=list[Item])
async def list_items():
    return [{"name": "Widget", "price": 9.99}, {"name": "Gadget", "price": 19.99}]
```

---

## HTTP Status Codes

```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
@app.delete("/items/{id}", status_code=status.HTTP_204_NO_CONTENT)
@app.put("/items/{id}", status_code=status.HTTP_200_OK)
```

### Dynamically Changing Status Code

Return a `Response` object directly to override the declared status:

```python
from fastapi import Response, status

@app.put("/items/{item_id}", status_code=200, response_model=Item)
async def upsert_item(item_id: int, item: ItemCreate, response: Response):
    existing = db.get(item_id)
    if not existing:
        response.status_code = status.HTTP_201_CREATED
        return db.create(item_id, item)
    return db.update(item_id, item)
```

---

## `Response` Object in Path Operations

Inject `Response` to modify headers, cookies, or status without losing `response_model` benefits:

```python
from fastapi import Response

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int, response: Response):
    item = db.get(item_id)
    response.headers["X-Cache"] = "HIT"
    response.headers["X-Item-ID"] = str(item_id)
    return item
```

---

## Custom Response Classes

By default, FastAPI returns `JSONResponse`. You can change this:

```python
from fastapi.responses import (
    JSONResponse,       # Default — JSON with Content-Type: application/json
    HTMLResponse,       # HTML content
    PlainTextResponse,  # Plain text
    RedirectResponse,   # HTTP redirect
    FileResponse,       # File download
    StreamingResponse,  # Stream large content
    ORJSONResponse,     # Faster JSON using orjson
    UJSONResponse,      # Faster JSON using ujson
)

# Set globally for an endpoint
@app.get("/html", response_class=HTMLResponse)
async def read_html():
    return "<html><body>Hello</body></html>"

# Return directly (bypasses response_model)
@app.get("/redirect")
async def redirect():
    return RedirectResponse(url="/new-location", status_code=301)
```

### Performance: Use `ORJSONResponse` for High-Throughput APIs

```python
from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse)  # global default

# Or per endpoint
@app.get("/items/", response_class=ORJSONResponse)
async def list_items():
    return [{"name": "Widget"}]
```

> **Note**: `ORJSONResponse` requires installing `orjson`. It's significantly faster for large payloads.

---

## Additional Responses Documentation

Document non-200 responses in OpenAPI:

```python
@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        404: {"description": "Item not found", "model": ErrorResponse},
        403: {"description": "Access denied"},
        500: {"description": "Internal server error"},
    }
)
async def read_item(item_id: int):
    ...
```

---

## `response_model=None` — Disable Validation

When you need full control over the response:

```python
@app.get("/raw", response_model=None)
async def raw_response():
    return JSONResponse(content={"anything": "goes"}, status_code=200)
```

---

## OpenAPI Response Examples

Add example values to your response schema:

```python
class Item(BaseModel):
    name: str
    price: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"name": "Widget", "price": 9.99},
                {"name": "Gadget", "price": 19.99},
            ]
        }
    }
```

---

## Common Pitfalls

### 1. Returning ORM Objects Without `from_attributes`
```python
# ❌ SQLAlchemy ORM object returned without ORM mode
class ItemResponse(BaseModel):
    name: str
    price: float

@app.get("/items/{id}", response_model=ItemResponse)
def get_item(id: int):
    return db.query(Item).get(id)  # SQLAlchemy object — fails without ORM mode
```

```python
# ✅ Enable ORM mode
class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    price: float
```

### 2. Including Sensitive Fields
```python
# ❌ NEVER do this
class UserResponse(BaseModel):
    username: str
    hashed_password: str   # Exposed to client!
```

### 3. Using `dict` as `response_model`
```python
# ❌ No filtering, no docs, no validation
@app.get("/items/", response_model=dict)

# ✅ Use a proper Pydantic model
@app.get("/items/", response_model=ItemResponse)
```

---

## Best Practices

1. **Always declare `response_model`** — never rely on raw dict returns in production
2. **Separate schemas** for input (`ItemCreate`) and output (`ItemResponse`)
3. **Use `response_model_exclude_unset=True`** for PATCH endpoints
4. **Add `response_model_exclude_none=True`** for cleaner JSON payloads
5. **Use `ORJSONResponse`** for performance-critical endpoints
6. **Document error responses** with `responses={}` for complete API contracts
7. **Never expose** internal fields, secrets, or database internals in response schemas

```python
# Production-grade response configuration
@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    response_model_exclude_none=True,
    responses={
        404: {"model": NotFoundError, "description": "Item not found"},
        403: {"model": ForbiddenError, "description": "Access denied"},
    },
    summary="Get item by ID",
)
async def get_item(item_id: int, db: SessionDep) -> ItemResponse:
    item = await db.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
```

---

## Related Topics
- [`03-request-body-pydantic.md`](./03-request-body-pydantic.md) — Input schema design
- [`08-error-handling.md`](./08-error-handling.md) — Error response models
- [`01-routing-path-operations.md`](./01-routing-path-operations.md) — Path operation configuration
