# Database Integration

## Overview

FastAPI is **database-agnostic** — it doesn't bundle an ORM. You integrate your preferred database library via the **Dependency Injection** system. Common choices:

| Library | Type | Style |
|---|---|---|
| **SQLModel** | ORM | Pydantic + SQLAlchemy hybrid (recommended) |
| **SQLAlchemy 2.0** | ORM | Most mature, full-featured |
| **Tortoise ORM** | Async ORM | Django-like async ORM |
| **asyncpg** | Driver | Raw async PostgreSQL queries |
| **Motor** | Async ODM | MongoDB async |
| **Beanie** | Async ODM | MongoDB + Pydantic |

---

## SQLModel (Recommended for FastAPI)

SQLModel is built by the FastAPI author — it merges Pydantic models and SQLAlchemy table definitions into one class, eliminating duplication.

### Installation

```bash
pip install sqlmodel
```

### Model Definition

```python
from sqlmodel import Field, SQLModel
from typing import Optional

# Table model (maps to DB) + Pydantic model
class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    secret_name: str  # stored in DB but not in public API response
```

### Schema Separation Pattern

```python
# Base — shared fields
class HeroBase(SQLModel):
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)

# DB table — with all fields
class Hero(HeroBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    secret_name: str  # private field

# Create schema — what API accepts
class HeroCreate(HeroBase):
    secret_name: str

# Update schema — all optional for PATCH
class HeroUpdate(SQLModel):
    name: str | None = None
    age: int | None = None
    secret_name: str | None = None

# Response schema — what API returns (no secret_name)
class HeroPublic(HeroBase):
    id: int
```

---

## Database Connection & Session Management

### SQLite (Development)

```python
from sqlmodel import Session, SQLModel, create_engine

sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
```

### PostgreSQL (Production)

```python
from sqlmodel import Session, SQLModel, create_engine
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost/dbname"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

settings = Settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,      # Validate connections before use
    pool_recycle=1800,       # Recycle connections every 30 min
)
```

---

## Session as a Dependency

```python
from typing import Annotated
from fastapi import Depends

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]
```

Usage:

```python
@app.get("/heroes/{hero_id}", response_model=HeroPublic)
def get_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(404, "Hero not found")
    return hero
```

---

## Full CRUD Implementation

```python
from fastapi import FastAPI, HTTPException, Query
from sqlmodel import select

app = FastAPI()

# CREATE
@app.post("/heroes/", response_model=HeroPublic, status_code=201)
def create_hero(hero: HeroCreate, session: SessionDep):
    db_hero = Hero.model_validate(hero)
    session.add(db_hero)
    session.commit()
    session.refresh(db_hero)
    return db_hero

# READ (list with pagination)
@app.get("/heroes/", response_model=list[HeroPublic])
def list_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes

# READ (single)
@app.get("/heroes/{hero_id}", response_model=HeroPublic)
def get_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(404, "Hero not found")
    return hero

# UPDATE (partial — PATCH)
@app.patch("/heroes/{hero_id}", response_model=HeroPublic)
def update_hero(hero_id: int, hero_update: HeroUpdate, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(404, "Hero not found")

    # Only update provided fields
    hero_data = hero_update.model_dump(exclude_unset=True)
    hero.sqlmodel_update(hero_data)
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero

# DELETE
@app.delete("/heroes/{hero_id}", status_code=204)
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(404, "Hero not found")
    session.delete(hero)
    session.commit()
```

---

## Async Database with SQLAlchemy

For high-throughput async APIs:

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, select

# Async engine
async_engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/dbname",
    pool_size=10,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Dependency
async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]

# Endpoint
@app.get("/heroes/")
async def list_heroes(session: AsyncSessionDep):
    result = await session.execute(select(Hero))
    return result.scalars().all()
```

---

## Relationships

```python
from sqlmodel import Relationship

class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    heroes: list["Hero"] = Relationship(back_populates="team")

class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")
```

### Eager Loading (avoid N+1 queries)

```python
from sqlmodel import select
from sqlalchemy.orm import selectinload

# Load heroes with their teams in one query
stmt = select(Hero).options(selectinload(Hero.team))
heroes = session.exec(stmt).all()
```

---

## Transactions

```python
def transfer_funds(from_id: int, to_id: int, amount: float, session: Session):
    from_account = session.get(Account, from_id)
    to_account = session.get(Account, to_id)

    if from_account.balance < amount:
        raise ValueError("Insufficient funds")

    from_account.balance -= amount
    to_account.balance += amount

    session.add(from_account)
    session.add(to_account)
    # session.commit() happens at the end of the dependency
```

The `yield` dependency pattern handles commit/rollback automatically:

```python
def get_session():
    with Session(engine) as session:
        try:
            yield session
            session.commit()   # Commits if no exception
        except Exception:
            session.rollback() # Rolls back on any error
            raise
```

---

## Alembic — Database Migrations

Alembic handles schema migrations for SQLAlchemy (and SQLModel):

```bash
pip install alembic
alembic init alembic
```

`alembic/env.py`:
```python
from sqlmodel import SQLModel
from app.models import *  # Import all models

target_metadata = SQLModel.metadata
```

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "add age column to hero"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Connection Pooling Best Practices

```python
# For web apps with multiple workers:
engine = create_engine(
    DATABASE_URL,
    # Each worker should have its own pool
    # pool_size = workers * connections_per_worker
    pool_size=5,        # Per-process pool size
    max_overflow=10,    # Burst capacity
    pool_timeout=30,    # Raise error if no connection available in 30s
    pool_recycle=1800,  # Replace connections every 30 minutes
    pool_pre_ping=True, # Test connection health before use (handles DB restarts)
)
```

---

## Common Pitfalls

### 1. Session Scope Too Wide
```python
# ❌ Session opened globally — shared across requests (NOT thread-safe)
session = Session(engine)

# ✅ Session opened per-request via dependency
def get_session():
    with Session(engine) as session:
        yield session
```

### 2. N+1 Query Problem
```python
# ❌ N+1: one query for heroes, N queries for each team
heroes = session.exec(select(Hero)).all()
for hero in heroes:
    print(hero.team.name)  # Each access hits DB!

# ✅ Eager load with joinedload or selectinload
from sqlalchemy.orm import selectinload
stmt = select(Hero).options(selectinload(Hero.team))
heroes = session.exec(stmt).all()
```

### 3. Missing `pool_pre_ping` in Production
```python
# Without pool_pre_ping, stale connections cause 500 errors after DB restart
engine = create_engine(url, pool_pre_ping=True)
```

### 4. Exposing Internal Models Directly
```python
# ❌ Returns all fields including sensitive ones
@app.get("/users/{id}")
def get_user(id: int, session: SessionDep):
    return session.get(User, id)  # includes hashed_password!

# ✅ Use response_model to filter
@app.get("/users/{id}", response_model=UserPublic)
def get_user(id: int, session: SessionDep):
    return session.get(User, id)
```

---

## Best Practices

1. **Use connection pooling** — always, in every production deployment
2. **Use `pool_pre_ping=True`** to handle DB restarts gracefully
3. **Separate schemas** — `Create`, `Update`, `Public`/`Response` models
4. **Use Alembic** for schema migrations — never `create_all()` in production
5. **Set `pool_size` based on worker count** — 5-10 connections per worker is typical
6. **Use `expire_on_commit=False`** for async sessions to avoid lazy load errors
7. **Always paginate** list endpoints — never return unbounded queries

---

## Related Topics
- [`05-dependency-injection.md`](./05-dependency-injection.md) — Session as yield dependency
- [`03-request-body-pydantic.md`](./03-request-body-pydantic.md) — Schema separation
- [`12-background-tasks-lifespan.md`](./12-background-tasks-lifespan.md) — DB setup at startup
- [`16-deployment-production.md`](./16-deployment-production.md) — Connection pool tuning
