# Middleware & CORS

## What is Middleware?

Middleware is code that runs **before every request** reaches your path operation and/or **after every response** leaves it. It wraps the entire request-response cycle.

```
Client Request
     ↓
[ Middleware 1 ]  ← runs first (request phase)
     ↓
[ Middleware 2 ]
     ↓
[ Path Operation ] ← your handler
     ↓
[ Middleware 2 ]  ← runs after (response phase)
     ↓
[ Middleware 1 ]  ← runs last
     ↓
Client Response
```

Use cases:
- Request logging
- CORS headers
- Authentication token extraction
- Rate limiting
- Request ID injection
- Timing/observability
- Gzip compression
- Trusted host checking

---

## Creating Custom Middleware

### `@app.middleware("http")`

```python
import time
import uuid
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response
```

### Starlette `BaseHTTPMiddleware` Class

For more complex middleware with state and configuration:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger=None):
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next) -> Response:
        self.logger.info(f"→ {request.method} {request.url}")
        try:
            response = await call_next(request)
            self.logger.info(f"← {response.status_code}")
            return response
        except Exception as exc:
            self.logger.error(f"Error: {exc}")
            raise

app.add_middleware(LoggingMiddleware, logger=my_logger)
```

### Pure ASGI Middleware (Advanced, Maximum Performance)

```python
class RawMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Modify scope, receive, or send here
            pass
        await self.app(scope, receive, send)

app.add_middleware(RawMiddleware)
```

---

## CORS — Cross-Origin Resource Sharing

**CORS** controls whether browsers allow JavaScript from one origin (domain) to make requests to your API at a different origin.

### Why CORS Exists

Without CORS, a malicious website `evil.com` could make API requests to `yourbank.com` using your logged-in credentials (cookies/session). CORS is the browser's security mechanism to prevent this.

### Adding CORS Middleware

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com", "https://admin.yourapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Total-Count", "X-Request-ID"],
    max_age=600,  # Preflight cache duration in seconds
)
```

### CORS Parameter Reference

| Parameter | Description | Production Value |
|---|---|---|
| `allow_origins` | Allowed origin URLs | Explicit list — never `["*"]` with credentials |
| `allow_methods` | Allowed HTTP methods | Specific methods you support |
| `allow_headers` | Allowed request headers | Headers your clients send |
| `allow_credentials` | Allow cookies/auth headers | `True` only if needed |
| `expose_headers` | Headers visible to JS | Pagination, request ID headers |
| `max_age` | Preflight cache (seconds) | `600` (10 min) |
| `allow_origin_regex` | Regex pattern for origins | For dynamic subdomains |

### Development vs Production CORS

```python
# Development — permissive
if settings.ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Production — explicit
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,  # ["https://app.example.com"]
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
```

> ⚠️ **Security warning**: `allow_origins=["*"]` with `allow_credentials=True` is **not allowed** by browsers and will be rejected. If you need credentials, list origins explicitly.

### Dynamic Origins (Subdomain Pattern)

```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.yourcompany\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Built-in Starlette Middleware

FastAPI (via Starlette) ships with several ready-to-use middleware:

### GZip Compression

```python
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)  # compress responses > 1KB
```

### Trusted Hosts

Prevent HTTP Host header attacks:

```python
from starlette.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourapp.com", "*.yourapp.com", "localhost"]
)
```

### HTTPS Redirect

```python
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
```

### Session Middleware

```python
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SESSION_SECRET_KEY,
    session_cookie="session",
    max_age=3600,
    same_site="lax",
    https_only=True,
)
```

---

## Middleware Registration Order

**Middleware is applied in reverse registration order** (last added = first to wrap):

```python
app.add_middleware(MiddlewareC)  # 3rd added
app.add_middleware(MiddlewareB)  # 2nd added
app.add_middleware(MiddlewareA)  # 1st added

# Execution order for request: A → B → C → handler
# Execution order for response: handler → C → B → A
```

**Recommended order** (register in this order, innermost first):

```python
app.add_middleware(CORSMiddleware, ...)         # Outermost — handles OPTIONS preflight first
app.add_middleware(TrustedHostMiddleware, ...)  # Security check early
app.add_middleware(GZipMiddleware, ...)         # Compress responses
app.add_middleware(LoggingMiddleware)           # Log after all other middleware
app.add_middleware(TracingMiddleware)           # Tracing wraps everything
```

---

## Accessing Request State

Middleware can pass data to path operations via `request.state`:

```python
@app.middleware("http")
async def inject_request_context(request: Request, call_next):
    request.state.user_ip = request.client.host
    request.state.request_id = str(uuid.uuid4())
    return await call_next(request)

@app.get("/items/")
async def list_items(request: Request):
    # Access values set by middleware
    print(request.state.request_id)
    print(request.state.user_ip)
    return []
```

---

## Exception Handling in Middleware

```python
@app.middleware("http")
async def error_tracking_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        # Log to Sentry, Datadog, etc.
        sentry_sdk.capture_exception(exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
```

> **Note**: FastAPI's built-in exception handlers (`HTTPException`, `RequestValidationError`) run inside the middleware stack, so they appear as normal responses to outer middleware. Unhandled exceptions bubble up.

---

## Performance Monitoring Middleware

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# OpenTelemetry auto-instrumentation (recommended)
FastAPIInstrumentor.instrument_app(app)

# Custom timing middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    # Push to Prometheus, StatsD, etc.
    metrics.histogram(
        "http_request_duration_seconds",
        duration,
        tags={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
        }
    )
    return response
```

---

## Common Pitfalls

### 1. Reading the Request Body in Middleware
```python
# ❌ This consumes the body — handler gets empty body
@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()  # consumed!
    # handler receives empty body
    return await call_next(request)

# ✅ Cache the body and make it re-readable
@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    # Re-inject by creating a new receive callable
    async def receive():
        return {"type": "http.request", "body": body}
    request._receive = receive  # patch the receive
    return await call_next(request)
```

### 2. Heavy Middleware on Every Request
```python
# ❌ DB query in middleware for every single request
@app.middleware("http")
async def attach_config(request: Request, call_next):
    request.state.config = await db.get_config()  # N+1 problem!
    return await call_next(request)

# ✅ Cache config at startup
config_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_cache.update(await db.get_config())
    yield
```

---

## Best Practices

1. **Be explicit with CORS origins** — never use `["*"]` in production
2. **Register CORSMiddleware first** so preflight OPTIONS requests are handled before auth middleware
3. **Keep middleware fast** — it runs on every request
4. **Use `request.state`** to share data between middleware and handlers
5. **Add `X-Request-ID`** headers for distributed tracing correlation
6. **Log middleware errors separately** from application errors
7. **Avoid blocking operations** in middleware — always `await`

---

## Related Topics
- [`08-error-handling.md`](./08-error-handling.md) — Exception handlers
- [`06-security-authentication.md`](./06-security-authentication.md) — Auth middleware patterns
- [`16-deployment-production.md`](./16-deployment-production.md) — Reverse proxy CORS handling
