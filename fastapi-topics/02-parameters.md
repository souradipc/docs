# Parameters — Path, Query, Header, Cookie

## How FastAPI Determines Parameter Types

FastAPI uses a simple decision tree to classify function parameters:

```
Is the name in the path template?  → Path parameter
Is it a Pydantic model or Body()?  → Request body
Otherwise?                         → Query parameter
```

```python
@app.get("/items/{item_id}")
async def read_item(
    item_id: int,          # → path param (in path template)
    q: str | None = None,  # → query param (simple type, not in path)
    skip: int = 0,         # → query param
    limit: int = 100,      # → query param
):
    ...
```

---

## Path Parameters

### Basic Declaration

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}
```

### Path with Validation

Use `Path()` from `fastapi` to add constraints and metadata:

```python
from typing import Annotated
from fastapi import FastAPI, Path

@app.get("/items/{item_id}")
async def get_item(
    item_id: Annotated[int, Path(
        title="The ID of the item",
        description="Must be between 1 and 999",
        ge=1,    # greater than or equal
        le=999,  # less than or equal
    )]
):
    return {"item_id": item_id}
```

### Validation Constraints (shared with Query and Body)

| Constraint | Applies to | Description |
|---|---|---|
| `ge` | int/float | Greater than or equal |
| `gt` | int/float | Greater than (exclusive) |
| `le` | int/float | Less than or equal |
| `lt` | int/float | Less than (exclusive) |
| `min_length` | str | Minimum string length |
| `max_length` | str | Maximum string length |
| `pattern` | str | Regex pattern match |

---

## Query Parameters

Any simple-typed parameter not in the path is a query parameter:

```python
# GET /items?skip=0&limit=10&active=true
@app.get("/items")
async def list_items(skip: int = 0, limit: int = 100, active: bool = True):
    return {"skip": skip, "limit": limit, "active": active}
```

### Required vs Optional Query Parameters

```python
# Required (no default)
async def search(q: str): ...
# GET /search → 422 (q is required)
# GET /search?q=foo → works

# Optional (has default)
async def search(q: str | None = None): ...
# GET /search → q is None
# GET /search?q=foo → q is "foo"

# Optional with default value
async def list_items(limit: int = 100): ...
```

### Boolean Query Parameters

FastAPI accepts these string values as `True`: `1`, `true`, `on`, `yes`  
And these as `False`: `0`, `false`, `off`, `no`

```python
@app.get("/items")
async def list_items(active: bool = True):
    ...
# GET /items?active=false → active = False
# GET /items?active=0     → active = False
# GET /items?active=yes   → active = True
```

### Query Parameter with Validation

```python
from fastapi import Query

@app.get("/items")
async def list_items(
    q: Annotated[str | None, Query(
        min_length=3,
        max_length=50,
        pattern="^[a-z]+$",
        title="Search query",
        description="Only lowercase letters",
        alias="search-query",  # Accept as ?search-query=foo in URL
    )] = None
):
    ...
```

### Alias — When Python Name Doesn't Match URL

```python
@app.get("/items")
async def get_items(
    item_query: Annotated[str | None, Query(alias="item-query")] = None
):
    # URL: GET /items?item-query=foo
    # Function receives: item_query="foo"
    ...
```

### Deprecating a Query Parameter

```python
q: Annotated[str | None, Query(deprecated=True)] = None
```

### List / Multiple Values

```python
# GET /items?tags=foo&tags=bar&tags=baz
@app.get("/items")
async def list_items(tags: Annotated[list[str], Query()] = []):
    return {"tags": tags}
```

---

## Header Parameters

```python
from fastapi import Header

@app.get("/items")
async def read_items(
    x_token: Annotated[str | None, Header()] = None,
    user_agent: Annotated[str | None, Header()] = None,
):
    return {"X-Token": x_token, "User-Agent": user_agent}
```

> **Auto-conversion**: FastAPI converts underscores (`_`) to hyphens (`-`) in header names. `x_token` reads the `X-Token` header.

### Disable Underscore-to-Hyphen Conversion

```python
x_token: Annotated[str, Header(convert_underscores=False)]
```

### Duplicate Headers (list)

```python
x_token: Annotated[list[str] | None, Header()] = None
# Reads multiple X-Token headers into a list
```

---

## Cookie Parameters

```python
from fastapi import Cookie

@app.get("/items")
async def read_items(
    session_id: Annotated[str | None, Cookie()] = None,
    ads_id: Annotated[str | None, Cookie()] = None,
):
    return {"session_id": session_id}
```

---

## Using `Annotated` vs Old Syntax

FastAPI supports two syntaxes. **`Annotated` is the recommended modern approach:**

```python
# Modern (recommended) — Annotated
from typing import Annotated
from fastapi import Query

async def search(q: Annotated[str, Query(min_length=3)] = None): ...

# Old syntax (still works)
async def search(q: str = Query(default=None, min_length=3)): ...
```

Benefits of `Annotated`:
- Metadata is **separate** from the default value
- Works better with type checkers (Mypy, Pyright)
- Reusable type aliases

```python
# Reusable annotated alias
SearchQuery = Annotated[str | None, Query(min_length=3, max_length=50)]

@app.get("/items")
async def search_items(q: SearchQuery = None): ...

@app.get("/users")
async def search_users(q: SearchQuery = None): ...
```

---

## Excluding Parameters from OpenAPI

```python
from fastapi import Query

# Hidden from Swagger UI — useful for internal params
async def get_item(
    internal_key: Annotated[str, Query(include_in_schema=False)] = ""
): ...
```

---

## Common Pitfalls

### 1. Path Parameter Type Mismatch
```python
@app.get("/items/{item_id}")
async def get_item(item_id: int):  # expects int

# GET /items/abc → 422 error (not int)
# This is actually correct behavior — don't suppress it
```

### 2. Query Parameter Boolean Coercion
```python
# This WILL NOT work as expected
async def get_items(active: bool = True):
    ...
# GET /items?active=True → True  ✅
# GET /items?active=False → False ✅  
# GET /items?active=FALSE → False ✅ (case-insensitive)
# GET /items?active=nope → 422 ❌ (unrecognized)
```

### 3. Missing `Query()` for List Parameters
```python
# ❌ Will NOT work as list — treated as str
async def get_items(tags: list[str] = []):

# ✅ Correct
async def get_items(tags: Annotated[list[str], Query()] = []):
```

### 4. Using Mutable Defaults
```python
# ❌ BUG: mutable default shared across calls
async def get_items(tags: list[str] = []):

# ✅ SAFE: list is wrapped by Query() — new list per call
async def get_items(tags: Annotated[list[str], Query()] = []):
```

---

## Best Practices

1. **Always add `title` and `description`** to Query/Path — they appear in Swagger UI
2. **Use `Annotated`** for clean, reusable type definitions
3. **Validate early**: add `min_length`, `max_length`, `ge`, `le` constraints rather than validating inside the function
4. **Use aliases** when your URL convention uses hyphens but Python uses underscores
5. **Mark deprecated params** instead of removing them immediately — gives clients time to migrate
6. **Never use headers for sensitive data** without HTTPS

```python
# Production-quality parameter declarations
@app.get("/users")
async def list_users(
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max records to return")] = 100,
    search: Annotated[str | None, Query(min_length=2, max_length=100)] = None,
    active: Annotated[bool, Query(description="Filter by active status")] = True,
    sort_by: Annotated[str, Query(pattern="^(name|email|created_at)$")] = "created_at",
):
    ...
```

---

## Related Topics
- [`03-request-body-pydantic.md`](./03-request-body-pydantic.md) — Body parameters
- [`05-dependency-injection.md`](./05-dependency-injection.md) — Reusable parameter sets via Depends
- [`08-error-handling.md`](./08-error-handling.md) — Handling 422 validation errors
