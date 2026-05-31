# Request Body & Pydantic Models

## What is the Request Body?

The **request body** is data sent by the client in the HTTP body (typically JSON). FastAPI uses **Pydantic models** to declare the expected shape, perform validation, and serialize the data.

Unlike path and query parameters (which are strings from the URL), body parameters are structured JSON objects with full type hierarchies.

---

## Defining a Request Body

Declare a Pydantic model class and use it as a type hint in the path operation:

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

**What FastAPI does automatically:**
1. Reads the request body as JSON
2. Validates against the `Item` schema
3. Returns `422 Unprocessable Entity` if validation fails
4. Converts the data to an `Item` instance — available in the function
5. Generates the JSON Schema for Swagger UI docs

---

## Pydantic Fields — Validation and Metadata

Use `Field()` to add validation constraints and documentation metadata to model fields:

```python
from pydantic import BaseModel, Field
from typing import Annotated

class Item(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
        description="Item name, must be 1-100 characters"
    )
    price: float = Field(
        gt=0,
        description="Price in USD, must be positive"
    )
    quantity: int = Field(default=1, ge=1, le=10000)
    tags: list[str] = Field(default_factory=list)
    sku: str | None = Field(
        default=None,
        pattern=r"^[A-Z]{3}-\d{4}$",
        examples=["ABC-1234"]
    )
```

### Field Constraints Reference

| Constraint | Types | Description |
|---|---|---|
| `min_length` | str, list | Minimum length |
| `max_length` | str, list | Maximum length |
| `pattern` | str | Regex validation |
| `gt` / `ge` | int, float | Greater than / greater or equal |
| `lt` / `le` | int, float | Less than / less or equal |
| `multiple_of` | int, float | Must be a multiple of this value |
| `default` | any | Default value when field is omitted |
| `default_factory` | callable | Callable to generate default |

---

## Optional vs Required Fields

```python
class ItemCreate(BaseModel):
    name: str                          # Required — must be provided
    price: float                       # Required
    description: str | None = None    # Optional — defaults to None
    tax: float = 0.0                   # Optional — defaults to 0.0
    tags: list[str] = []              # Optional — defaults to empty list
```

> **Pydantic v2 note**: Use `default_factory=list` instead of `= []` for mutable defaults at the class level in complex scenarios:
```python
tags: list[str] = Field(default_factory=list)
```

---

## Nested Models

Pydantic models can be nested — FastAPI validates the entire hierarchy:

```python
class Image(BaseModel):
    url: str
    name: str

class Item(BaseModel):
    name: str
    price: float
    image: Image | None = None
    images: list[Image] = []
```

```json
// Valid request body:
{
    "name": "Widget",
    "price": 9.99,
    "image": {
        "url": "https://cdn.example.com/widget.jpg",
        "name": "Widget Photo"
    },
    "images": [
        {"url": "...", "name": "..."}
    ]
}
```

---

## Multiple Body Parameters

When you have multiple Pydantic models as parameters, FastAPI expects a JSON object with each model name as a key:

```python
class Item(BaseModel):
    name: str
    price: float

class User(BaseModel):
    username: str
    full_name: str | None = None

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, user: User):
    ...
```

Expected request body:
```json
{
    "item": {"name": "Widget", "price": 9.99},
    "user": {"username": "johndoe"}
}
```

### Mixing Path + Query + Body

```python
from fastapi import Path, Query
from typing import Annotated

@app.put("/items/{item_id}")
async def update_item(
    item_id: Annotated[int, Path(ge=1)],  # path param
    q: str | None = None,                  # query param
    item: Item | None = None,              # body param (optional)
):
    result = {"item_id": item_id}
    if q:
        result.update({"q": q})
    if item:
        result.update({"item": item})
    return result
```

---

## `Body()` — Forcing a Parameter to be a Body Field

For singular types (non-model) that you want in the body:

```python
from fastapi import Body

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    importance: Annotated[int, Body(gt=0)],  # singular body field
):
    ...
```

Expected request body:
```json
{
    "item": {"name": "Widget", "price": 9.99},
    "importance": 5
}
```

### Embed a Single Model

Force a single model to be wrapped in a key:

```python
@app.put("/items/{item_id}")
async def update_item(
    item: Annotated[Item, Body(embed=True)]
):
    ...
```

Expected:
```json
{"item": {"name": "Widget", "price": 9.99}}
```

Instead of bare:
```json
{"name": "Widget", "price": 9.99}
```

---

## Schema Inheritance and Model Separation

A common production pattern is to separate read/write schemas:

```python
from pydantic import BaseModel
from datetime import datetime

# Base — shared fields
class ItemBase(BaseModel):
    name: str
    description: str | None = None
    price: float

# Create — what the client sends (no ID, no timestamps)
class ItemCreate(ItemBase):
    pass

# Update — all fields optional for PATCH
class ItemUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    price: float | None = None

# Response — what the API returns (includes server-generated fields)
class ItemResponse(ItemBase):
    id: int
    created_at: datetime
    owner_id: int

    model_config = {"from_attributes": True}  # Pydantic v2: ORM mode
```

### The 3-Model Pattern (Best Practice)

```
ItemCreate   → what the API accepts (POST body)
ItemUpdate   → what the API accepts for updates (PATCH body)
ItemResponse → what the API returns (response_model)
```

This prevents accidentally exposing internal fields like `hashed_password`.

---

## Pydantic v2 Model Config

```python
from pydantic import BaseModel, ConfigDict

class ItemResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,    # Enable ORM mode (read from ORM objects)
        populate_by_name=True,   # Allow field name as alias
        str_strip_whitespace=True,  # Auto-strip whitespace from strings
        json_schema_extra={      # Extra OpenAPI schema info
            "example": {
                "name": "Widget",
                "price": 9.99
            }
        }
    )
    name: str
    price: float
```

---

## Custom Validators

### Field Validators (Pydantic v2)

```python
from pydantic import BaseModel, field_validator, model_validator

class Item(BaseModel):
    name: str
    price: float
    discount: float = 0.0

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty or whitespace")
        return v.strip()

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return round(v, 2)  # round to 2 decimal places

    @model_validator(mode="after")
    def discount_less_than_price(self) -> "Item":
        if self.discount >= self.price:
            raise ValueError("Discount must be less than price")
        return self
```

---

## Extra Fields Handling

```python
from pydantic import BaseModel, ConfigDict

class StrictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields
    name: str
    price: float

# POST /items with {"name": "x", "price": 1.0, "hacker_field": "inject"}
# → 422: extra inputs are not permitted
```

Options for `extra`:
- `"ignore"` (default) — silently ignore extra fields
- `"forbid"` — raise validation error for extra fields
- `"allow"` — allow and store extra fields

> **Production recommendation**: Use `extra="forbid"` for security-sensitive endpoints to prevent unknown field injection.

---

## `model_dump()` and Serialization

```python
item = Item(name="Widget", price=9.99, description=None)

# All fields
item.model_dump()
# → {"name": "Widget", "price": 9.99, "description": None}

# Exclude None values
item.model_dump(exclude_none=True)
# → {"name": "Widget", "price": 9.99}

# Exclude unset fields (fields not provided in creation)
item.model_dump(exclude_unset=True)
# → {"name": "Widget", "price": 9.99}  (description not included)

# Include only specific fields
item.model_dump(include={"name", "price"})
# → {"name": "Widget", "price": 9.99}
```

---

## Common Pitfalls

### 1. Forgetting `from_attributes=True` for ORM objects
```python
# ❌ Will fail when trying to create from SQLAlchemy model
ItemResponse.model_validate(db_item)

# ✅ With ORM mode enabled
class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
```

### 2. Using `dict()` instead of `model_dump()`
```python
# ❌ Deprecated in Pydantic v2
item.dict()

# ✅ Pydantic v2
item.model_dump()
```

### 3. Mutable Default Fields
```python
# ❌ Potential issue
class Item(BaseModel):
    tags: list[str] = []  # Shared across instances (Pydantic handles this, but be explicit)

# ✅ Explicit factory
class Item(BaseModel):
    tags: list[str] = Field(default_factory=list)
```

---

## Best Practices

1. **Separate create/update/response schemas** — never expose `hashed_password` or internal IDs in creation schemas
2. **Use `extra="forbid"`** on input schemas to prevent injection attacks
3. **Add `description` and `examples`** to `Field()` — they show up in Swagger UI
4. **Use `exclude_unset=True`** in PATCH handlers to update only provided fields
5. **Enable `from_attributes=True`** on response models that map from ORM objects

```python
# Production PATCH pattern
@router.patch("/{item_id}", response_model=ItemResponse)
async def update_item(
    item_id: int,
    item_update: ItemUpdate,
    db: SessionDep,
):
    db_item = db.get(Item, item_id)
    if not db_item:
        raise HTTPException(404, "Item not found")

    # Only update fields that were explicitly provided
    update_data = item_update.model_dump(exclude_unset=True)
    db_item.sqlmodel_update(update_data)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

---

## Related Topics
- [`02-parameters.md`](./02-parameters.md) — Query, path, header parameters
- [`04-response-models.md`](./04-response-models.md) — Controlling response shape
- [`10-database-integration.md`](./10-database-integration.md) — Pydantic + ORM integration
