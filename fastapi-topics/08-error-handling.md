# Error Handling

## How FastAPI Handles Errors

FastAPI has a layered error handling system:

1. **Pydantic validation errors** → automatically returns `422 Unprocessable Entity`
2. **`HTTPException`** → returns the specified HTTP status code
3. **Custom exception handlers** → catch any exception type and return a custom response
4. **Unhandled exceptions** → Starlette returns `500 Internal Server Error`

---

## `HTTPException`

The primary tool for returning error responses:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "ItemNotFound"},  # optional custom headers
        )
    return {"item": items[item_id]}
```

### `detail` Can Be Any JSON-serializable Value

```python
raise HTTPException(
    status_code=422,
    detail=[
        {
            "loc": ["body", "price"],
            "msg": "Price must be positive",
            "type": "value_error",
        }
    ]
)
```

---

## HTTP Status Code Reference

```python
from fastapi import status

# 2xx — Success
status.HTTP_200_OK
status.HTTP_201_CREATED
status.HTTP_204_NO_CONTENT

# 3xx — Redirect
status.HTTP_301_MOVED_PERMANENTLY
status.HTTP_302_FOUND
status.HTTP_304_NOT_MODIFIED

# 4xx — Client Error
status.HTTP_400_BAD_REQUEST       # Generic bad input
status.HTTP_401_UNAUTHORIZED      # Not authenticated
status.HTTP_403_FORBIDDEN         # Authenticated but not authorized
status.HTTP_404_NOT_FOUND         # Resource not found
status.HTTP_405_METHOD_NOT_ALLOWED
status.HTTP_409_CONFLICT          # State conflict (duplicate, etc.)
status.HTTP_410_GONE              # Resource permanently deleted
status.HTTP_422_UNPROCESSABLE_ENTITY  # Validation error
status.HTTP_429_TOO_MANY_REQUESTS  # Rate limiting

# 5xx — Server Error
status.HTTP_500_INTERNAL_SERVER_ERROR
status.HTTP_502_BAD_GATEWAY
status.HTTP_503_SERVICE_UNAVAILABLE
status.HTTP_504_GATEWAY_TIMEOUT
```

---

## Custom Exception Classes

Define domain-specific exceptions:

```python
class AppError(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)

class NotFoundError(AppError):
    def __init__(self, resource: str, id: int | str):
        super().__init__(f"{resource} with id={id} not found", "NOT_FOUND")
        self.resource = resource
        self.id = id

class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(message, "CONFLICT")

class AuthorizationError(AppError):
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, "FORBIDDEN")
```

---

## Custom Exception Handlers

Register handlers to convert exceptions to HTTP responses:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=404,
        content={
            "error": exc.code,
            "message": exc.message,
            "resource": exc.resource,
            "id": str(exc.id),
        }
    )

@app.exception_handler(ConflictError)
async def conflict_handler(request: Request, exc: ConflictError):
    return JSONResponse(
        status_code=409,
        content={"error": exc.code, "message": exc.message}
    )

@app.exception_handler(AuthorizationError)
async def auth_handler(request: Request, exc: AuthorizationError):
    return JSONResponse(
        status_code=403,
        content={"error": exc.code, "message": exc.message}
    )
```

Now in your service layer, you can raise these exceptions directly:

```python
# service/items.py
async def get_item(item_id: int, db: Session) -> Item:
    item = db.get(Item, item_id)
    if not item:
        raise NotFoundError("Item", item_id)
    return item
```

---

## Overriding Built-in Exception Handlers

### Override `HTTPException` Handler

```python
from fastapi.exceptions import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url),
        }
    )
```

### Override `RequestValidationError` Handler

Customize how Pydantic validation errors are returned:

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " → ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Input validation failed",
            "details": errors,
        }
    )
```

---

## Error Response Schema

Define consistent error response models:

```python
from pydantic import BaseModel
from typing import Any

class ErrorDetail(BaseModel):
    field: str
    message: str
    type: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[ErrorDetail] | None = None
    request_id: str | None = None

class NotFoundResponse(BaseModel):
    error: str = "NOT_FOUND"
    message: str
    resource: str

# Use in response documentation
@app.get(
    "/items/{item_id}",
    response_model=Item,
    responses={
        404: {"model": NotFoundResponse},
        422: {"model": ErrorResponse},
    }
)
async def get_item(item_id: int):
    ...
```

---

## Re-raising with Built-in Handler

Sometimes you want to add logic but still use the default FastAPI handler:

```python
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # Log the error
    logger.warning(f"Validation error on {request.url}: {exc.errors()}")
    # Delegate to default handler
    return await request_validation_exception_handler(request, exc)
```

---

## Global Error Catching

Catch all unhandled exceptions (last resort):

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log to your error tracking service
    logger.exception(f"Unhandled error: {exc}", extra={"url": str(request.url)})
    sentry_sdk.capture_exception(exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Our team has been notified.",
            "request_id": request.state.request_id,
        }
    )
```

---

## Error Handling in Dependencies

Dependencies can also raise exceptions:

```python
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
```

---

## Raising HTTP Errors from Service Layer

**Pattern**: Service layer raises domain exceptions, API layer converts them:

```python
# Layer 1: Service raises domain exceptions (no FastAPI imports)
class ItemService:
    async def get_item(self, item_id: int) -> Item:
        item = await self.repo.find(item_id)
        if not item:
            raise NotFoundError("Item", item_id)
        return item

# Layer 2: API layer handles domain exceptions → HTTP responses
@app.get("/items/{item_id}")
async def get_item(item_id: int, service: ItemServiceDep):
    # The exception handler registered above will catch NotFoundError
    return await service.get_item(item_id)
```

This pattern keeps your service layer **framework-agnostic** and testable.

---

## Common Pitfalls

### 1. Exposing Internal Error Details
```python
# ❌ Leaks internal implementation details
except Exception as e:
    raise HTTPException(500, detail=str(e))  # Exposes stack traces, DB errors, etc.

# ✅ Generic message for clients, detailed logs internally
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(500, detail="Internal server error")
```

### 2. Wrong Status Codes
```python
# ❌ Using 400 for "not found" (wrong semantics)
raise HTTPException(400, "User not found")

# ✅ Correct status code
raise HTTPException(404, "User not found")

# ❌ Using 403 for "unauthenticated" (403 = authenticated but forbidden)
raise HTTPException(403, "Not logged in")

# ✅ Use 401 for missing/invalid authentication
raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
```

### 3. Not Handling Exceptions in Yield Dependencies
```python
# ❌ DB not rolled back on exception
def get_db():
    db = Session(engine)
    yield db
    db.close()  # Won't be called if exception raised

# ✅ Use try/finally
def get_db():
    db = Session(engine)
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
```

---

## Best Practices

1. **Use domain-specific exceptions** in service layer — keep it framework-free
2. **Register exception handlers** to convert domain exceptions to HTTP responses
3. **Never expose** raw exception messages, stack traces, or DB errors to clients
4. **Log all 5xx errors** with full context (request ID, user, path, exception)
5. **Use consistent error response shape** across all endpoints
6. **Document error responses** in `responses={}` for every endpoint
7. **Return `request_id`** in error responses so users can report issues

```python
# Production error response pattern
class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs"""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str | None = None  # The request URL
    request_id: str | None = None

@app.exception_handler(HTTPException)
async def problem_detail_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ProblemDetail(
            title=HTTP_STATUS_CODES.get(exc.status_code, "Error"),
            status=exc.status_code,
            detail=str(exc.detail),
            instance=str(request.url),
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(exclude_none=True),
        headers=exc.headers,
    )
```

---

## Related Topics
- [`05-dependency-injection.md`](./05-dependency-injection.md) — Error handling in yield deps
- [`07-middleware-cors.md`](./07-middleware-cors.md) — Middleware-level error catching
- [`03-request-body-pydantic.md`](./03-request-body-pydantic.md) — Validation error structure
