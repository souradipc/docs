# Dependency Injection

## What is Dependency Injection in FastAPI?

FastAPI's **Dependency Injection (DI) system** is one of its most powerful features. It allows path operation functions to declare what they need, and FastAPI provides it automatically.

A **dependency** is any callable (function, class, or callable instance) that FastAPI calls before your path operation, injects the result, and optionally handles cleanup.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

# A simple dependency
async def get_db_connection():
    conn = create_db_connection()
    return conn

# Injecting the dependency
@app.get("/items/")
async def list_items(db=Depends(get_db_connection)):
    return db.query("SELECT * FROM items")
```

---

## Why Use Dependency Injection?

Without DI, you repeat logic across endpoints:

```python
# ❌ Without DI — repeated authentication logic everywhere
@app.get("/items/")
async def list_items(token: str = Header()):
    user = verify_token(token)
    if not user:
        raise HTTPException(401)
    ...

@app.get("/orders/")
async def list_orders(token: str = Header()):
    user = verify_token(token)  # duplicated again
    if not user:
        raise HTTPException(401)
    ...
```

```python
# ✅ With DI — centralized, reusable logic
async def get_current_user(token: str = Header()):
    user = verify_token(token)
    if not user:
        raise HTTPException(401)
    return user

@app.get("/items/")
async def list_items(user=Depends(get_current_user)):
    ...

@app.get("/orders/")
async def list_orders(user=Depends(get_current_user)):
    ...
```

---

## How Dependencies Work Internally

1. FastAPI inspects the path operation's parameters
2. For each `Depends(callable)`, FastAPI:
   - Introspects the callable's parameters (recursively resolving sub-dependencies)
   - Calls the callable with its resolved parameters
   - Injects the return value into the path operation
3. If the dependency is a generator (`yield`), the cleanup code runs after the response

```
Request
  → Resolve dependency D1
      → Resolve sub-dependency D2 (D1 depends on D2)
          → Call D2() → inject result into D1
      → Call D1(d2_result) → inject result into handler
  → Call path_operation(d1_result, ...)
  → Send response
  → Run D1 cleanup (after yield)
  → Run D2 cleanup (after yield)
```

---

## Dependency Types

### 1. Function Dependency

```python
from fastapi import Depends, Query
from typing import Annotated

async def common_parameters(
    skip: int = 0,
    limit: int = 100,
    q: str | None = None
):
    return {"skip": skip, "limit": limit, "q": q}

CommonsDep = Annotated[dict, Depends(common_parameters)]

@app.get("/items/")
async def list_items(commons: CommonsDep):
    return commons

@app.get("/users/")
async def list_users(commons: CommonsDep):
    return commons
```

### 2. Class-Based Dependency

Classes are callable, so they work as dependencies. FastAPI calls `__init__`:

```python
class CommonQueryParams:
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        q: str | None = None
    ):
        self.skip = skip
        self.limit = limit
        self.q = q

@app.get("/items/")
async def list_items(commons: Annotated[CommonQueryParams, Depends()]):
    # Shortcut: Depends() without arg uses the annotated type
    return {"skip": commons.skip, "limit": commons.limit}
```

> **`Depends()` shortcut**: When the type annotation IS the dependency class, `Depends()` (empty) uses it automatically.

### 3. Yield Dependency (with Cleanup)

Use `yield` for dependencies that need cleanup (database sessions, file handles, connections):

```python
from sqlmodel import Session

def get_db():
    db = Session(engine)
    try:
        yield db          # ← value injected here
    finally:
        db.close()        # ← runs after response is sent

@app.get("/items/")
def list_items(db: Annotated[Session, Depends(get_db)]):
    return db.exec(select(Item)).all()
```

**Execution order**:
1. `db = Session(engine)` — setup
2. `yield db` — inject into handler
3. Handler executes, response is sent
4. `db.close()` — cleanup (always runs, even on exceptions)

---

## Sub-Dependencies (Dependency Chains)

Dependencies can depend on other dependencies:

```python
async def get_token(authorization: str = Header()):
    return authorization.split("Bearer ")[-1]

async def get_current_user(token: str = Depends(get_token)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return User(username=payload["sub"])

async def get_active_user(user: User = Depends(get_current_user)):
    if user.disabled:
        raise HTTPException(400, "Inactive user")
    return user

# All three deps resolved automatically
@app.get("/me")
async def read_me(user: User = Depends(get_active_user)):
    return user
```

---

## Dependency Caching

By default, FastAPI **caches** a dependency's result within a single request. If multiple path operation parameters use the same dependency, it's called **once**:

```python
async def get_db():
    print("Creating DB session")  # Called once per request, not twice
    with Session(engine) as session:
        yield session

@app.get("/items/")
async def list_items(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
    # get_current_user also uses Depends(get_db) — same session instance
):
    ...
```

### Disable Caching

Use `use_cache=False` when you need a **fresh** call each time:

```python
async def verify_nonce(nonce: str = Header()):
    """Must be called fresh each time — one-time tokens"""
    validate_and_consume_nonce(nonce)

@app.post("/action")
async def do_action(
    _nonce1: Annotated[None, Depends(verify_nonce, use_cache=False)],
    _nonce2: Annotated[None, Depends(verify_nonce, use_cache=False)],
):
    ...
```

---

## Router-Level and App-Level Dependencies

Apply dependencies to **all routes in a router** or **the entire app**:

```python
from fastapi import APIRouter, Depends

# All routes in this router require authentication
router = APIRouter(
    prefix="/admin",
    dependencies=[Depends(require_admin_user)],
)

@router.get("/users")
async def list_all_users():
    ...  # require_admin_user already enforced

@router.delete("/users/{id}")
async def delete_user(id: int):
    ...  # require_admin_user already enforced
```

```python
# App-level dependency — applies to ALL routes
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

```python
# Per-endpoint dependency (no return value needed)
@app.get("/items/", dependencies=[Depends(rate_limiter)])
async def list_items():
    ...
```

---

## Global Dependencies via `app` or `include_router`

```python
app.include_router(
    items.router,
    dependencies=[Depends(get_current_user)],  # All item routes require auth
)
```

---

## Yield Dependencies with Exception Handling

```python
async def get_db():
    db = Session(engine)
    try:
        yield db
        db.commit()      # Commit if no exception
    except Exception:
        db.rollback()    # Rollback on exception
        raise            # Re-raise so FastAPI handles it
    finally:
        db.close()       # Always close
```

---

## Background Tasks in Dependencies

```python
from fastapi import BackgroundTasks

async def log_query(background_tasks: BackgroundTasks, q: str):
    background_tasks.add_task(write_log, q)
    return q

@app.get("/items/")
async def list_items(query: str = Depends(log_query)):
    ...
```

---

## Using `Annotated` for Reusable Dependencies

```python
from typing import Annotated
from fastapi import Depends

# Create reusable dependency type aliases
CurrentUser = Annotated[User, Depends(get_current_user)]
ActiveUser = Annotated[User, Depends(get_active_user)]
AdminUser = Annotated[User, Depends(require_admin)]
DBSession = Annotated[Session, Depends(get_db)]

# Clean, readable endpoint signatures
@app.get("/items/")
async def list_items(
    db: DBSession,
    user: CurrentUser,
):
    ...

@app.delete("/admin/users/{id}")
async def delete_user(
    id: int,
    db: DBSession,
    admin: AdminUser,  # Automatically enforces admin role
):
    ...
```

---

## Testing — Override Dependencies

In tests, swap real dependencies with mocks:

```python
# tests/test_items.py
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_db, get_current_user

def get_test_db():
    yield test_db_session  # Use test DB

def get_mock_user():
    return User(id=1, username="testuser")

# Override for all tests in this file
app.dependency_overrides[get_db] = get_test_db
app.dependency_overrides[get_current_user] = get_mock_user

client = TestClient(app)

def test_list_items():
    response = client.get("/items/")
    assert response.status_code == 200
```

---

## Common Pitfalls

### 1. Async/Sync Mismatch in Dependencies
```python
# ❌ async dependency calling blocking code
async def get_db():
    return blocking_db.connect()  # Blocks event loop!

# ✅ Use sync dependency (FastAPI runs it in threadpool)
def get_db():
    return blocking_db.connect()  # Safe in sync def
```

### 2. Not Using `yield` for Cleanup
```python
# ❌ Session never closed
async def get_db():
    return Session(engine)

# ✅ Session properly closed
async def get_db():
    with Session(engine) as session:
        yield session
```

### 3. Heavy Computation in Dependencies
```python
# ❌ ML model loaded on every request
async def get_model():
    return load_heavy_ml_model()  # 10 seconds per request!

# ✅ Load once at startup, inject the cached model
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    ml_model = load_heavy_ml_model()  # once at startup
    yield

async def get_model():
    return ml_model
```

---

## Best Practices

1. **Use `Annotated` type aliases** to make endpoint signatures readable
2. **Always use `yield`** for dependencies that need cleanup
3. **Prefer class-based dependencies** for complex initialization
4. **Apply auth dependencies at router level** rather than per-endpoint
5. **Use `dependency_overrides`** in tests — never mock deep internals
6. **Keep dependencies focused** — one responsibility per dependency
7. **Avoid side effects in pure dependencies** — prefer returning values

```python
# Production pattern: layered auth dependencies
async def get_token_from_header(authorization: str = Header()) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authorization header")
    return authorization[7:]

async def get_user_from_token(token: str = Depends(get_token_from_header)) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return User(username=payload["sub"], role=payload.get("role"))
    except JWTError:
        raise HTTPException(401, "Invalid token")

async def require_admin(user: User = Depends(get_user_from_token)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin access required")
    return user

# Type aliases
TokenDep = Annotated[str, Depends(get_token_from_header)]
UserDep = Annotated[User, Depends(get_user_from_token)]
AdminDep = Annotated[User, Depends(require_admin)]
```

---

## Related Topics
- [`06-security-authentication.md`](./06-security-authentication.md) — Auth built on top of DI
- [`10-database-integration.md`](./10-database-integration.md) — DB sessions as dependencies
- [`15-testing.md`](./15-testing.md) — `dependency_overrides` in tests
