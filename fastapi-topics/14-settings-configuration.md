# Settings & Configuration

## Why Use `pydantic-settings`?

`pydantic-settings` is the standard way to manage configuration in FastAPI applications. It:

- Reads configuration from **environment variables** (12-factor app)
- Reads from **`.env` files** for local development
- Validates and **type-coerces** all settings (string `"true"` → `bool True`)
- Provides **editor autocompletion** for settings
- Supports **nested settings** and **secrets files**
- Raises clear errors if required settings are missing

---

## Installation

```bash
pip install pydantic-settings
```

---

## Basic Settings Class

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,      # ENV_VAR and env_var are the same
        extra="ignore",            # Ignore unknown env vars
    )

    # App settings
    APP_NAME: str = "My API"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ENV: Literal["development", "staging", "production"] = "development"

    # Database
    DATABASE_URL: str  # Required — must be in env
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Auth
    SECRET_KEY: str  # Required
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # External services
    REDIS_URL: str = "redis://localhost:6379"
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587

# Singleton instance
settings = Settings()
```

---

## `.env` File

```dotenv
# .env (never commit to version control)
APP_NAME="My Production API"
ENV=development
DEBUG=true

DATABASE_URL=postgresql://user:password@localhost:5432/mydb
SECRET_KEY=your-256-bit-secret-generated-with-openssl-rand-hex-32

REDIS_URL=redis://localhost:6379/0

ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8080"]

SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
```

> **Important**: Add `.env` to `.gitignore`. Commit `.env.example` with placeholder values instead.

---

## Environment Variable Naming

Pydantic Settings maps env vars to settings fields by name (case-insensitive by default):

```python
# Field name       → Env var (case-insensitive)
DATABASE_URL       → DATABASE_URL, database_url, Database_URL (all work)
DB_POOL_SIZE       → DB_POOL_SIZE, db_pool_size
ALLOWED_ORIGINS    → ALLOWED_ORIGINS
```

### Prefix for Namespacing

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_")

    name: str = "My App"
    debug: bool = False

# Reads: APP_NAME, APP_DEBUG from environment
```

---

## Nested Settings

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseModel):
    url: str = "postgresql://localhost/db"
    pool_size: int = 10
    max_overflow: int = 20

class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379"
    db: int = 0
    max_connections: int = 50

class Settings(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    debug: bool = False
```

Set nested values via environment:
```dotenv
DATABASE__URL=postgresql://prod-host/mydb
DATABASE__POOL_SIZE=20
REDIS__URL=redis://prod-redis:6379
```
(Double underscore `__` as separator for nested keys)

---

## Settings as a Dependency

For testability and dependency injection:

```python
# app/core/config.py
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Type alias for injection
SettingsDep = Annotated[Settings, Depends(get_settings)]
```

```python
# In a path operation
@app.get("/info")
async def app_info(settings: SettingsDep):
    return {
        "app_name": settings.APP_NAME,
        "env": settings.ENV,
        "version": "1.0.0",
    }
```

### Override in Tests

```python
# tests/conftest.py
from app.core.config import get_settings, Settings
from app.main import app

def get_test_settings():
    return Settings(
        ENV="testing",
        DATABASE_URL="sqlite:///./test.db",
        SECRET_KEY="test-secret-key",
    )

app.dependency_overrides[get_settings] = get_test_settings
```

> **Note**: `@lru_cache` caches the settings instance. Call `get_settings.cache_clear()` in tests that need fresh settings.

---

## Multi-Environment Configuration

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

ENV = os.getenv("ENV", "development")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f".env.{ENV}" if ENV != "production" else None,
        env_file_encoding="utf-8",
    )
    ...

settings = Settings()
```

File structure:
```
.env.development   # Local dev settings
.env.test          # Test environment settings
.env.example       # Template (committed to git)
# Production: env vars set directly in the container/server
```

---

## Secrets from Files (Docker Secrets, Kubernetes Secrets)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        secrets_dir="/run/secrets"  # Docker/K8s secrets mount path
    )

    database_password: str  # Read from /run/secrets/database_password
    secret_key: str         # Read from /run/secrets/secret_key
```

---

## Validation and Computed Properties

```python
from pydantic import field_validator, computed_field, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "mydb"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""

    # Computed field — assembled from other fields
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @field_validator("DB_PORT")
    @classmethod
    def valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"Invalid port: {v}")
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.ENV == "production":
            if not self.DB_PASSWORD:
                raise ValueError("DB_PASSWORD required in production")
            if self.SECRET_KEY == "dev-secret":
                raise ValueError("Must set a real SECRET_KEY in production")
        return self
```

---

## Complete Production Settings Example

```python
# app/core/config.py
from functools import lru_cache
from typing import Annotated, Literal
from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import Depends

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "FastAPI App"
    API_V1_STR: str = "/api/v1"
    ENV: Literal["development", "test", "staging", "production"] = "development"
    DEBUG: bool = False

    # ── Security ─────────────────────────────────────────────
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    # ── Redis ─────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Email ─────────────────────────────────────────────────
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM: str = "noreply@example.com"

    # ── Observability ─────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: str | None = None
    OTEL_ENDPOINT: str | None = None

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @model_validator(mode="after")
    def production_guard(self) -> "Settings":
        if self.is_production:
            assert self.SECRET_KEY != "dev-secret", "Real SECRET_KEY required in production"
            assert self.SENTRY_DSN, "SENTRY_DSN required in production"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()

SettingsDep = Annotated[Settings, Depends(get_settings)]
```

---

## Common Pitfalls

### 1. Committing `.env` to Git
```bash
# .gitignore — always include
.env
.env.local
.env.development
.env.staging
.env.production
```

### 2. Not Using `lru_cache`
```python
# ❌ Reads .env file on every request
def get_settings():
    return Settings()  # O(1000s of requests) * file I/O

# ✅ Cached after first call
@lru_cache
def get_settings():
    return Settings()
```

### 3. Hardcoded Defaults for Secrets
```python
# ❌ Default secret value — easily missed in prod
SECRET_KEY: str = "dev-secret-12345"

# ✅ No default — fails fast if not configured
SECRET_KEY: str  # Required — ValueError if missing

# Or validate in production only:
@model_validator(mode="after")
def check_secret(self):
    if self.ENV == "production" and len(self.SECRET_KEY) < 32:
        raise ValueError("SECRET_KEY must be at least 32 chars in production")
    return self
```

---

## Best Practices

1. **Always use `pydantic-settings`** — never `os.environ.get()` scattered throughout code
2. **No default for secrets** — `SECRET_KEY: str` (required) not `SECRET_KEY: str = "default"`
3. **Use `lru_cache`** on the settings factory function
4. **Validate production constraints** in `model_validator`
5. **Commit `.env.example`** with placeholder values — never commit `.env`
6. **Use Docker/K8s secrets** in production — not environment variables for sensitive values
7. **Inject settings as a dependency** for testability

---

## Related Topics
- [`16-deployment-production.md`](./16-deployment-production.md) — Environment-based config in Docker
- [`15-testing.md`](./15-testing.md) — Overriding settings in tests
- [`05-dependency-injection.md`](./05-dependency-injection.md) — Settings as a dependency
