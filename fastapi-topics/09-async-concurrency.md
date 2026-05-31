# Async & Concurrency

## The Core Question: `async def` or `def`?

FastAPI supports **both** async and sync path operation functions. The choice affects how FastAPI runs your code:

| Function type | How FastAPI runs it | Use when |
|---|---|---|
| `async def` | Directly on the event loop | Calling `await`-able async libraries |
| `def` | In a **threadpool** (concurrent) | Calling blocking/sync libraries |

```python
# Async — event loop runs this directly
@app.get("/items")
async def list_items():
    items = await db.execute(select(Item))  # Non-blocking DB call
    return items.all()

# Sync — FastAPI moves this to a threadpool worker
@app.get("/orders")
def list_orders():
    orders = sync_db.query(Order).all()  # Blocking DB call — safe in threadpool
    return orders
```

> **Key insight**: With `def`, FastAPI runs your function in a separate thread using `anyio`'s threadpool, so it does NOT block the event loop. You can safely mix `async def` and `def` in the same app.

---

## How Python's Event Loop Works

The event loop is a single-threaded scheduler that manages coroutines (async functions):

```
Event Loop
├── Coroutine A: waiting for DB response → suspend A
├── Coroutine B: waiting for HTTP request → suspend B
├── Coroutine C: CPU computation → run C until complete
├── DB responds → resume A
└── HTTP arrives → resume B
```

While one coroutine waits for I/O, the event loop runs other coroutines. This is called **cooperative multitasking**.

**The cardinal rule**: Never block the event loop from an `async def` function.

---

## What Blocks the Event Loop

These operations in `async def` **WILL block** all concurrent requests:

```python
# ❌ time.sleep() blocks the event loop
@app.get("/slow")
async def slow():
    time.sleep(5)  # ALL other requests frozen for 5 seconds!

# ❌ Synchronous DB calls (psycopg2, pymysql, etc.)
@app.get("/items")
async def list_items():
    items = sync_db_conn.execute("SELECT * FROM items")  # blocks event loop

# ❌ requests library (synchronous HTTP)
@app.get("/proxy")
async def proxy():
    response = requests.get("https://api.example.com")  # blocks!

# ❌ Heavy CPU computation
@app.get("/hash")
async def compute():
    result = bcrypt.hashpw(b"password", bcrypt.gensalt(rounds=14))  # CPU-bound, blocks
```

---

## Async Alternatives

```python
# ✅ asyncio.sleep() — non-blocking
import asyncio

async def slow():
    await asyncio.sleep(5)  # Other requests run while waiting

# ✅ Async DB (asyncpg, SQLAlchemy async, Tortoise ORM)
async def list_items():
    async with AsyncSession(engine) as session:
        result = await session.execute(select(Item))
        return result.scalars().all()

# ✅ httpx for async HTTP requests
import httpx

async def proxy():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
    return response.json()
```

---

## CPU-Bound Tasks — Use ProcessPool or Background Tasks

For CPU-intensive work (image processing, ML inference, encryption), use:

### Option 1: `run_in_executor` with ProcessPool

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

def cpu_intensive_work(data: bytes) -> bytes:
    """This runs in a separate process — doesn't block event loop"""
    return process_image(data)

@app.post("/process")
async def process(file: UploadFile):
    data = await file.read()
    loop = asyncio.get_running_loop()
    # Offload to process pool
    result = await loop.run_in_executor(executor, cpu_intensive_work, data)
    return {"size": len(result)}
```

### Option 2: `def` Route (ThreadPool)

```python
# FastAPI runs this in a threadpool automatically
@app.post("/hash")
def hash_password(password: str = Body(...)):
    """OK for I/O-bound but NOT for CPU-bound — threads share GIL"""
    return {"hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()}
```

> **GIL note**: Python's Global Interpreter Lock (GIL) means threads can't truly parallelize CPU-bound Python code. For real CPU parallelism, use `ProcessPoolExecutor` or Celery workers.

---

## `asyncio.gather` — Concurrent Async Operations

Run multiple async operations **simultaneously**:

```python
import asyncio
import httpx

@app.get("/dashboard")
async def get_dashboard(user_id: int):
    async with httpx.AsyncClient() as client:
        # Run all three API calls concurrently — total time = max(t1, t2, t3)
        # NOT t1 + t2 + t3
        profile, orders, recommendations = await asyncio.gather(
            client.get(f"/users/{user_id}"),
            client.get(f"/users/{user_id}/orders"),
            client.get(f"/users/{user_id}/recommendations"),
        )

    return {
        "profile": profile.json(),
        "orders": orders.json(),
        "recommendations": recommendations.json(),
    }
```

### Handling Partial Failures in gather

```python
results = await asyncio.gather(
    fetch_profile(user_id),
    fetch_orders(user_id),
    return_exceptions=True,  # Don't fail all if one fails
)

profile = results[0] if not isinstance(results[0], Exception) else None
orders = results[1] if not isinstance(results[1], Exception) else []
```

---

## `asyncio.TaskGroup` (Python 3.11+)

Modern structured concurrency:

```python
async def get_dashboard(user_id: int):
    async with asyncio.TaskGroup() as tg:
        profile_task = tg.create_task(fetch_profile(user_id))
        orders_task = tg.create_task(fetch_orders(user_id))
        recs_task = tg.create_task(fetch_recommendations(user_id))
    # All tasks complete here (or all are cancelled on first failure)
    return {
        "profile": profile_task.result(),
        "orders": orders_task.result(),
        "recommendations": recs_task.result(),
    }
```

---

## `anyio` — FastAPI's Async Backend

FastAPI uses `anyio` (which supports both `asyncio` and `trio`) as its async abstraction layer:

```python
import anyio

# Run blocking code in a threadpool from async context
async def safe_blocking_call():
    result = await anyio.to_thread.run_sync(blocking_function, arg1, arg2)
    return result

# Limit threadpool concurrency
limiter = anyio.CapacityLimiter(10)  # max 10 concurrent threads

async def rate_limited_blocking_call():
    async with limiter:
        result = await anyio.to_thread.run_sync(blocking_function)
    return result
```

---

## Connection Pool Management

Async DB connections must be managed at the app level, not per-request:

```python
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# ✅ Created once at startup — shared pool
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=10,         # Base pool connections
    max_overflow=20,      # Extra connections allowed
    pool_timeout=30,      # Wait up to 30s for connection
    pool_recycle=1800,    # Recycle connections every 30 min
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Engine auto-creates pool on first use
    await engine.dispose()  # Close all connections on shutdown

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

---

## Async Context Managers

```python
# Using async resources safely
@app.get("/process")
async def process_data():
    async with aiofiles.open("data.txt", mode="r") as f:
        content = await f.read()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("/external", json={"data": content})

    return response.json()
```

---

## Performance: async vs sync Benchmark Context

| Scenario | Best choice |
|---|---|
| Fast sync operations (in-memory, CPU < 10ms) | `def` (no async overhead) |
| Async-native DB (asyncpg, motor) | `async def` |
| Legacy sync DB (psycopg2, pymysql) | `def` (threadpool) |
| External HTTP calls | `async def` + `httpx` |
| File I/O | `async def` + `aiofiles` |
| CPU-intensive work | `def` or ProcessPool |
| Mixed DB + HTTP | `async def` with `asyncio.gather` |

---

## Common Pitfalls

### 1. Mixing sync and async incorrectly
```python
# ❌ Calling async function without await (runs nothing!)
@app.get("/items")
async def list_items():
    result = fetch_items()  # forgot await — returns coroutine object, not result
    return result

# ✅
async def list_items():
    result = await fetch_items()
    return result
```

### 2. Creating event loop inside async context
```python
# ❌ Don't do this inside async code
asyncio.run(some_async_func())  # Creates new event loop — error if one already running

# ✅ Just await
await some_async_func()
```

### 3. Blocking async DB inside sync handler
```python
# ❌ This defeats the purpose
def list_items(db: AsyncSession = Depends(get_async_db)):
    # Can't await inside sync def!
    items = db.execute(select(Item))  # won't work as expected
```

---

## Best Practices

1. **Default to `async def`** for new endpoints — easier to add async I/O later
2. **Use `def` only** when calling sync-only libraries (legacy DB drivers, etc.)
3. **Never call `time.sleep()`** in `async def` — use `await asyncio.sleep()`
4. **Use `asyncio.gather()`** for independent concurrent operations
5. **Limit connection pool sizes** — don't set unbounded pools
6. **Profile before optimizing** — async adds complexity only justified by performance needs
7. **Use `anyio.to_thread.run_sync()`** for unavoidable blocking calls inside async context

---

## Related Topics
- [`10-database-integration.md`](./10-database-integration.md) — Async DB sessions
- [`11-websockets-streaming.md`](./11-websockets-streaming.md) — Async streaming
- [`12-background-tasks-lifespan.md`](./12-background-tasks-lifespan.md) — Startup/shutdown lifecycle
