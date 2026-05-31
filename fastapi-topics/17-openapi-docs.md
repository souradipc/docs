# OpenAPI & Documentation

## What FastAPI Generates Automatically

FastAPI automatically generates a complete **OpenAPI 3.0** schema from your code and serves:

| URL | Interface |
|---|---|
| `/openapi.json` | Raw JSON schema |
| `/docs` | Swagger UI (interactive) |
| `/redoc` | ReDoc (read-only, clean) |

No additional configuration needed — just define your endpoints with type hints and Pydantic models.

---

## Customizing the OpenAPI Schema

```python
from fastapi import FastAPI

app = FastAPI(
    title="My Production API",
    description="""
## Overview

This API manages **widgets** and **sprockets** for the Acme Corporation backend.

### Authentication
Use Bearer token authentication via `Authorization: Bearer <token>` header.

### Rate Limiting
- Standard: 100 requests/minute
- Premium: 1000 requests/minute
    """,
    version="2.1.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "api@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_url="/api/v1/openapi.json",  # Custom schema URL
    docs_url="/api/v1/docs",             # Custom Swagger URL
    redoc_url="/api/v1/redoc",           # Custom ReDoc URL
)
```

---

## Disabling Documentation in Production

For security or performance, you may want to disable docs in production:

```python
from app.core.config import settings

app = FastAPI(
    docs_url="/docs" if settings.ENV != "production" else None,
    redoc_url="/redoc" if settings.ENV != "production" else None,
    openapi_url="/openapi.json" if settings.ENV != "production" else None,
)
```

Or protect docs behind authentication:

```python
from fastapi import FastAPI, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

app = FastAPI(docs_url=None, redoc_url=None)  # Disable default docs

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui(username: str = Depends(verify_internal_token)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")
```

---

## Endpoint Documentation

### Docstrings as Descriptions

```python
@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """
    Retrieve a single item by its unique identifier.

    - **item_id**: The integer ID of the item to retrieve.
    - Returns the full item object including all public fields.
    - Returns 404 if item does not exist.
    """
    ...
```

The docstring is rendered as **Markdown** in Swagger UI.

### Explicit `summary` and `description`

```python
@app.post(
    "/items/",
    summary="Create Item",
    description="Creates a new item with the provided data. Requires authentication.",
    response_description="The newly created item with its assigned ID.",
    status_code=201,
)
async def create_item(item: ItemCreate):
    ...
```

### Tags for Grouping

```python
@app.get("/users/", tags=["users"])
@app.post("/users/", tags=["users"])
@app.get("/items/", tags=["items"])

# Tags can have metadata
app = FastAPI(
    openapi_tags=[
        {
            "name": "users",
            "description": "User management operations. Requires authentication.",
            "externalDocs": {
                "description": "User guide",
                "url": "https://docs.example.com/users",
            },
        },
        {
            "name": "items",
            "description": "Item CRUD operations.",
        },
    ]
)
```

---

## Request/Response Examples

### Via `Field()` in Pydantic Models

```python
from pydantic import BaseModel, Field

class ItemCreate(BaseModel):
    name: str = Field(
        ...,
        examples=["Widget Pro", "Gadget Max"],
        description="The name of the item"
    )
    price: float = Field(
        ...,
        examples=[9.99, 49.99],
        gt=0,
    )
```

### Via `model_config`

```python
class Item(BaseModel):
    name: str
    price: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Widget Pro",
                    "price": 9.99,
                    "description": "The finest widget",
                }
            ]
        }
    }
```

### Via `openapi_examples` on Body()

```python
from fastapi import Body
from typing import Annotated

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Annotated[
        Item,
        Body(
            openapi_examples={
                "normal": {
                    "summary": "A normal item update",
                    "value": {"name": "Widget", "price": 9.99},
                },
                "discounted": {
                    "summary": "Item with discount",
                    "value": {"name": "Widget", "price": 5.00, "discount": 4.99},
                },
            }
        ),
    ],
):
    ...
```

---

## `include_in_schema` — Hiding Endpoints

```python
# Hide from OpenAPI docs but still functional
@app.get("/internal/health", include_in_schema=False)
async def internal_health():
    return {"status": "ok"}
```

---

## Custom OpenAPI Schema

For advanced customization, override the schema generation:

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Custom Schema API",
        version="2.0.0",
        description="Custom description",
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }

    # Apply security globally
    openapi_schema["security"] = [{"BearerAuth": []}]

    # Add server list
    openapi_schema["servers"] = [
        {"url": "https://api.example.com", "description": "Production"},
        {"url": "https://staging-api.example.com", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Local"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## Documenting Error Responses

```python
from pydantic import BaseModel

class HTTPErrorResponse(BaseModel):
    detail: str

class ValidationErrorDetail(BaseModel):
    loc: list[str | int]
    msg: str
    type: str

class ValidationErrorResponse(BaseModel):
    detail: list[ValidationErrorDetail]

@app.get(
    "/items/{item_id}",
    response_model=ItemResponse,
    responses={
        200: {"description": "Item found"},
        404: {
            "description": "Item not found",
            "model": HTTPErrorResponse,
            "content": {
                "application/json": {
                    "example": {"detail": "Item with id=42 not found"}
                }
            }
        },
        401: {"description": "Not authenticated", "model": HTTPErrorResponse},
        403: {"description": "Access forbidden", "model": HTTPErrorResponse},
        422: {"description": "Validation error", "model": ValidationErrorResponse},
    }
)
async def get_item(item_id: int):
    ...
```

---

## Swagger UI Customization

```python
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI

app = FastAPI(docs_url=None)  # Disable default

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="My API - Docs",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,    # Hide models section
            "persistAuthorization": True,       # Remember auth between reloads
            "displayRequestDuration": True,     # Show request time
            "filter": True,                     # Enable search filter
            "tryItOutEnabled": True,            # Auto-enable "try it out"
        }
    )
```

---

## OpenAPI Extensions (x-*)

Add custom extensions to the schema:

```python
def custom_openapi():
    schema = get_openapi(title="My API", version="1.0", routes=app.routes)

    # Add custom extension to an operation
    for path_data in schema["paths"].values():
        for operation in path_data.values():
            operation["x-internal-use-only"] = False
            operation["x-rate-limit"] = "100/minute"

    return schema
```

---

## Best Practices

1. **Always add `summary`** to every endpoint — shows in Swagger sidebar
2. **Write meaningful docstrings** — they become the operation description
3. **Add `response_description`** for the success response
4. **Document all error codes** in `responses={}` for client developers
5. **Add request body examples** — they appear in Swagger "Try it out"
6. **Use `tags`** consistently — organize by domain/resource
7. **Add tag metadata** in `openapi_tags` for descriptions
8. **Disable docs in production** or protect behind auth for sensitive APIs
9. **Use `include_in_schema=False`** for internal/health endpoints

---

## Related Topics
- [`01-routing-path-operations.md`](./01-routing-path-operations.md) — Path operation configuration
- [`03-request-body-pydantic.md`](./03-request-body-pydantic.md) — Schema from Pydantic models
- [`04-response-models.md`](./04-response-models.md) — Response documentation
- [`06-security-authentication.md`](./06-security-authentication.md) — Security schemes in docs
