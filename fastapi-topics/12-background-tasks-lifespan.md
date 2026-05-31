# Background Tasks & Lifespan

## Background Tasks

### What Are Background Tasks?

Background tasks run **after the response is sent** to the client. They're useful for operations that shouldn't delay the response:

- Sending email notifications
- Writing audit logs
- Updating caches
- Triggering webhooks
- Cleaning up temporary files

```
Client Request
     ↓
Path Operation (fast work)
     ↓
HTTP Response sent to client ← client is unblocked here
     ↓
Background task runs (email, logging, etc.)
```

---

### Basic Usage

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def write_notification(email: str, message: str):
    with open("log.txt", "a") as log:
        log.write(f"Notification sent to {email}: {message}\n")

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, "You have a new message!")
    return {"message": "Notification will be sent in the background"}
```

`BackgroundTasks.add_task(func, *args, **kwargs)`:
- `func` — sync or async callable
- `*args`, `**kwargs` — passed to the function

---

### Async Background Tasks

```python
import asyncio
import httpx

async def send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)

@app.post("/items/")
async def create_item(item: ItemCreate, background_tasks: BackgroundTasks):
    new_item = await item_service.create(item)
    background_tasks.add_task(
        send_webhook,
        "https://hooks.example.com/item-created",
        {"item_id": new_item.id, "name": new_item.name}
    )
    return new_item
```

---

### Background Tasks in Dependencies

```python
from fastapi import Depends, BackgroundTasks

async def log_operation(
    background_tasks: BackgroundTasks,
    operation: str,
    user_id: int,
):
    background_tasks.add_task(audit_log.write, operation, user_id)

@app.post("/items/")
async def create_item(
    item: ItemCreate,
    background_tasks: BackgroundTasks,
    user: CurrentUser,
    _: None = Depends(lambda bt=BackgroundTasks(): log_operation(bt, "create_item", user.id)),
):
    ...
```

A cleaner pattern:

```python
async def audit_dependency(
    background_tasks: BackgroundTasks,
    request: Request,
    user: CurrentUser,
):
    yield  # Path operation runs here
    background_tasks.add_task(
        audit_log.write,
        path=request.url.path,
        method=request.method,
        user_id=user.id,
    )

@app.post("/items/", dependencies=[Depends(audit_dependency)])
async def create_item(item: ItemCreate):
    ...
```

---

### Background Tasks Limitations

Background tasks run **in the same process** as the web server. This means:

| Consideration | Detail |
|---|---|
| **No retry logic** | If the task fails, it's silently lost |
| **Not durable** | Server restart kills queued tasks |
| **Single process** | Can't distribute across workers |
| **Blocks worker** | Long tasks reduce server throughput |

**For production workloads, use Celery or ARQ instead:**

| Use Case | Tool |
|---|---|
| Quick, best-effort tasks | `BackgroundTasks` |
| Reliable, retryable jobs | Celery, ARQ, Dramatiq |
| Scheduled jobs (cron) | Celery Beat, APScheduler |
| Long-running jobs | Celery, distributed workers |

---

## Lifespan — Startup & Shutdown

### What is Lifespan?

The `lifespan` context manager controls what happens when the application **starts** and **stops**. It's the correct place to:
- Initialize database connection pools
- Load ML models into memory
- Connect to external services (Redis, message queues)
- Warm up caches
- Clean up resources on shutdown

---

### Modern Lifespan (Recommended)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Shared state dictionary (or use app.state)
app_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ─── STARTUP ──────────────────────────────────────
    print("Starting up...")

    # Initialize DB connection pool
    app_state["db_engine"] = create_engine(settings.DATABASE_URL)

    # Load ML model once (not per-request)
    app_state["ml_model"] = load_model("./models/classifier.pkl")

    # Connect to Redis
    app_state["redis"] = await aioredis.from_url(settings.REDIS_URL)

    # Warm up cache
    await warm_up_cache(app_state["redis"])

    print("Startup complete!")

    yield   # ← Application runs here

    # ─── SHUTDOWN ─────────────────────────────────────
    print("Shutting down...")

    # Dispose DB connection pool
    app_state["db_engine"].dispose()

    # Close Redis connection
    await app_state["redis"].close()

    print("Shutdown complete!")


app = FastAPI(lifespan=lifespan)
```

---

### Accessing Lifespan State in Routes

Use `app.state` (the recommended approach) to share state across the app:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ml_model = load_model("./model.pkl")
    app.state.redis = await aioredis.from_url(settings.REDIS_URL)
    yield
    await app.state.redis.close()

app = FastAPI(lifespan=lifespan)

# In a route — access via request.app.state
@app.post("/predict")
async def predict(request: Request, data: PredictRequest):
    model = request.app.state.ml_model
    result = model.predict(data.features)
    return {"prediction": result}

# Or via dependency
def get_redis(request: Request) -> Redis:
    return request.app.state.redis

RedisDep = Annotated[Redis, Depends(get_redis)]

@app.get("/cache/{key}")
async def get_cache(key: str, redis: RedisDep):
    value = await redis.get(key)
    return {"value": value}
```

---

### Multiple Lifespan Contexts (Composition)

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def db_lifespan(app: FastAPI):
    app.state.db = create_engine(settings.DATABASE_URL)
    yield
    app.state.db.dispose()

@asynccontextmanager
async def redis_lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url(settings.REDIS_URL)
    yield
    await app.state.redis.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with db_lifespan(app):
        async with redis_lifespan(app):
            yield

app = FastAPI(lifespan=lifespan)
```

---

### Old-Style Events (Deprecated)

```python
# ⚠️ Deprecated — use lifespan context manager instead
@app.on_event("startup")
async def startup():
    app.state.redis = await aioredis.from_url(settings.REDIS_URL)

@app.on_event("shutdown")
async def shutdown():
    await app.state.redis.close()
```

> FastAPI's documentation now recommends `lifespan` over `on_event` for all new code.

---

## Combining Background Tasks + Lifespan

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks

email_queue = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start email worker at startup
    global email_queue
    email_queue = asyncio.Queue()
    worker_task = asyncio.create_task(email_worker(email_queue))
    yield
    # Stop worker at shutdown
    await email_queue.put(None)  # Poison pill
    await worker_task

async def email_worker(queue: asyncio.Queue):
    while True:
        job = await queue.get()
        if job is None:
            break
        await send_email(**job)

@app.post("/register")
async def register(user: UserCreate, background_tasks: BackgroundTasks):
    new_user = await create_user(user)
    # Add to persistent queue (not just BackgroundTask)
    await email_queue.put({"to": new_user.email, "template": "welcome"})
    return new_user
```

---

## Testing Lifespan

```python
from fastapi.testclient import TestClient

def test_app_with_lifespan():
    # TestClient as context manager triggers lifespan
    with TestClient(app) as client:
        # Lifespan startup has run
        response = client.get("/health")
        assert response.status_code == 200
    # Lifespan shutdown has run
```

---

## Common Pitfalls

### 1. Heavy Work per Request Instead of Startup
```python
# ❌ Model loaded on every request — 10 second cold start per request
@app.post("/predict")
async def predict(data: PredictRequest):
    model = load_model("./model.pkl")  # Called thousands of times
    return model.predict(data.features)

# ✅ Load once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model("./model.pkl")  # Once
    yield
```

### 2. Not Disposing Resources on Shutdown
```python
# ❌ Connections leak on server shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = create_engine(url)
    yield
    # Missing: app.state.db.dispose()

# ✅ Always clean up
async def lifespan(app: FastAPI):
    app.state.db = create_engine(url)
    yield
    app.state.db.dispose()
```

### 3. Background Tasks for Critical Work
```python
# ❌ Critical payment confirmation via BackgroundTask — can be lost
@app.post("/payment")
async def process_payment(payment: PaymentRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(confirm_payment_to_bank, payment)  # Not guaranteed!
    return {"status": "processing"}

# ✅ Use a proper job queue for critical tasks
@app.post("/payment")
async def process_payment(payment: PaymentRequest):
    job = await celery_app.send_task("confirm_payment", args=[payment.dict()])
    return {"status": "processing", "job_id": job.id}
```

---

## Best Practices

1. **Use `lifespan`** context manager for all startup/shutdown logic
2. **Load expensive resources once** at startup (DB pools, ML models, HTTP clients)
3. **Always dispose/close** resources in the `yield` cleanup section
4. **Use `app.state`** to share resources between lifespan and route handlers
5. **Reserve `BackgroundTasks`** for non-critical, best-effort operations
6. **Use Celery/ARQ** for durable, retryable background jobs
7. **Test lifespan** with `TestClient` as context manager

---

## Related Topics
- [`09-async-concurrency.md`](./09-async-concurrency.md) — Async patterns
- [`10-database-integration.md`](./10-database-integration.md) — DB connection pool lifecycle
- [`15-testing.md`](./15-testing.md) — Testing lifespan events
