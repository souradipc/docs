# Testing FastAPI Applications

## Testing Philosophy

A well-tested FastAPI app has:

| Layer | Tool | What to test |
|---|---|---|
| Unit | pytest | Services, validators, utilities — no HTTP |
| Integration | TestClient / AsyncClient | Full request → response cycle |
| Contract | TestClient + response_model | API schema consistency |
| E2E | httpx / playwright | Full stack behavior |

FastAPI's **dependency override system** is the key tool for writing isolated, fast tests.

---

## Setup

```bash
pip install pytest pytest-asyncio httpx anyio
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"      # Automatically handle async tests
testpaths = ["tests"]
```

---

## Synchronous Testing with `TestClient`

`TestClient` wraps your FastAPI app in a synchronous HTTPX client. Tests don't need `async/await`:

```python
# tests/test_items.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_create_item():
    response = client.post(
        "/api/v1/items/",
        json={"name": "Widget", "price": 9.99},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert "id" in data
```

---

## Async Testing with `AsyncClient`

For true async test execution (required for async database operations):

```python
# tests/test_items_async.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_read_items():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/items/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
```

---

## `conftest.py` — Shared Fixtures

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, StaticPool
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.database import get_session
from app.core.config import get_settings, Settings
from app.models import *  # Import all models to register them

# ── Test Database Setup ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_engine():
    """In-memory SQLite engine shared across session"""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)

@pytest.fixture
def db_session(test_engine):
    """Fresh DB session per test — rolled back after each test"""
    with Session(test_engine) as session:
        yield session
        session.rollback()

# ── Override Dependencies ────────────────────────────────────────────────────

@pytest.fixture
def client(db_session):
    """TestClient with DB and settings overrides"""
    def override_get_db():
        yield db_session

    def override_get_settings():
        return Settings(
            ENV="test",
            DATABASE_URL="sqlite://",
            SECRET_KEY="test-secret-key-32-chars-minimum",
        )

    app.dependency_overrides[get_session] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

# ── Async Client ─────────────────────────────────────────────────────────────

@pytest.fixture
async def async_client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_session] = override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as c:
        yield c
    app.dependency_overrides.clear()

# ── Test Users ───────────────────────────────────────────────────────────────

@pytest.fixture
def test_user(db_session):
    from app.models.user import User
    user = User(username="testuser", email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def auth_headers(test_user):
    from app.core.security import create_access_token
    token = create_access_token({"sub": test_user.username})
    return {"Authorization": f"Bearer {token}"}
```

---

## Dependency Overrides

The most powerful testing tool in FastAPI:

```python
# Override a dependency for a specific test
def test_admin_only_endpoint(client):
    # Override current user with an admin
    def mock_admin():
        return User(username="admin", role="admin", is_admin=True)

    app.dependency_overrides[get_current_user] = mock_admin
    response = client.get("/admin/users")
    assert response.status_code == 200
    app.dependency_overrides.pop(get_current_user)
```

```python
# Override via fixture for a whole test module
@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user] = lambda: test_user
    yield
    app.dependency_overrides.clear()
```

---

## Testing Authentication

```python
def test_protected_endpoint_without_token(client):
    response = client.get("/api/v1/users/me")
    assert response.status_code == 401

def test_protected_endpoint_with_invalid_token(client):
    response = client.get(
        "/api/v1/users/me",
        headers={"Authorization": "Bearer invalid-token"}
    )
    assert response.status_code == 401

def test_protected_endpoint_with_valid_token(client, auth_headers):
    response = client.get("/api/v1/users/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"

def test_admin_endpoint_with_regular_user(client, auth_headers):
    response = client.get("/admin/users", headers=auth_headers)
    assert response.status_code == 403
```

---

## Testing CRUD Endpoints

```python
class TestItemsCRUD:
    def test_create_item(self, client, auth_headers):
        response = client.post(
            "/api/v1/items/",
            json={"name": "Widget", "price": 9.99, "description": "A fine widget"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Widget"
        assert data["price"] == 9.99
        assert "id" in data
        return data["id"]

    def test_create_item_invalid_price(self, client, auth_headers):
        response = client.post(
            "/api/v1/items/",
            json={"name": "Widget", "price": -1.0},  # Invalid price
            headers=auth_headers,
        )
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("price" in str(e["loc"]) for e in errors)

    def test_get_item(self, client, auth_headers, test_item):
        response = client.get(f"/api/v1/items/{test_item.id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == test_item.id

    def test_get_nonexistent_item(self, client, auth_headers):
        response = client.get("/api/v1/items/99999", headers=auth_headers)
        assert response.status_code == 404

    def test_update_item(self, client, auth_headers, test_item):
        response = client.patch(
            f"/api/v1/items/{test_item.id}",
            json={"name": "Updated Widget"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Widget"

    def test_delete_item(self, client, auth_headers, test_item):
        response = client.delete(f"/api/v1/items/{test_item.id}", headers=auth_headers)
        assert response.status_code == 204

        # Verify deleted
        response = client.get(f"/api/v1/items/{test_item.id}", headers=auth_headers)
        assert response.status_code == 404
```

---

## Testing Validation Errors

```python
def test_missing_required_field(client):
    response = client.post(
        "/api/v1/items/",
        json={"description": "No name or price"},  # Missing required fields
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    fields_with_errors = [e["loc"][-1] for e in detail]
    assert "name" in fields_with_errors
    assert "price" in fields_with_errors

def test_wrong_type(client):
    response = client.post(
        "/api/v1/items/",
        json={"name": "Widget", "price": "not-a-number"},
    )
    assert response.status_code == 422
```

---

## Testing WebSockets

```python
def test_websocket_echo(client):
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("Hello")
        data = websocket.receive_text()
        assert data == "Echo: Hello"

def test_websocket_json(client):
    with client.websocket_connect("/ws/json") as websocket:
        websocket.send_json({"type": "ping"})
        data = websocket.receive_json()
        assert data["type"] == "pong"

def test_websocket_auth_rejected(client):
    with pytest.raises(Exception):  # WebSocket connection rejected
        with client.websocket_connect("/ws/protected?token=invalid"):
            pass
```

---

## Testing Lifespan Events

```python
def test_startup_resources_initialized():
    items = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        items["data"] = "initialized"
        yield
        items.clear()

    test_app = FastAPI(lifespan=lifespan)

    assert items == {}  # Before startup

    with TestClient(test_app) as client:
        assert items == {"data": "initialized"}  # After startup

    assert items == {}  # After shutdown
```

---

## Testing Background Tasks

```python
from unittest.mock import patch, AsyncMock

def test_background_task_called(client, auth_headers):
    with patch("app.services.email.send_welcome_email") as mock_send:
        response = client.post(
            "/api/v1/users/",
            json={"username": "newuser", "email": "new@example.com", "password": "secret"},
        )
        assert response.status_code == 201

    # Background task was scheduled (not necessarily executed yet in sync test)
    mock_send.assert_called_once_with("new@example.com")
```

---

## Test Coverage

```bash
# Run with coverage
pytest --cov=app --cov-report=html tests/

# View report
open htmlcov/index.html
```

```toml
# pyproject.toml
[tool.coverage.run]
source = ["app"]
omit = ["app/tests/*", "*/migrations/*"]

[tool.coverage.report]
fail_under = 80
```

---

## Common Pitfalls

### 1. Not Clearing Dependency Overrides
```python
# ❌ Override leaks to other tests
app.dependency_overrides[get_db] = mock_db
def test_something():
    ...
# get_db is still overridden for next test!

# ✅ Always clear after test
@pytest.fixture
def client():
    app.dependency_overrides[get_db] = mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()  # Always cleanup
```

### 2. Using Real DB in Unit Tests
```python
# ❌ Slow, brittle, leaves data in DB
def test_create_item():
    client.post("/items/", json={"name": "Widget"})
    items_in_db = real_db.query(Item).all()

# ✅ Use in-memory SQLite + rollback fixture
```

### 3. Shared Mutable State Between Tests
```python
# ❌ In-memory dict shared across tests
fake_db = {}

def test_a():
    fake_db["item"] = "Widget"  # Leaks to test_b

def test_b():
    assert fake_db == {}  # Fails!

# ✅ Use fixtures with proper scope
```

---

## Best Practices

1. **Use `TestClient` for synchronous tests** — simpler and sufficient for most cases
2. **Use `AsyncClient` with `ASGITransport`** for async test scenarios
3. **Override dependencies, not internals** — test through the API
4. **Use an in-memory SQLite database** for isolation and speed
5. **Rollback transactions** after each test — not truncate tables
6. **Separate fixtures** by scope: `session`, `module`, `function`
7. **Test validation errors** explicitly — not just happy paths
8. **Mock external services** (email, payment, third-party APIs)

---

## Related Topics
- [`05-dependency-injection.md`](./05-dependency-injection.md) — `dependency_overrides`
- [`12-background-tasks-lifespan.md`](./12-background-tasks-lifespan.md) — Testing lifespan
- [`14-settings-configuration.md`](./14-settings-configuration.md) — Test settings override
