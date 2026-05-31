# Security & Authentication

## Security in FastAPI

FastAPI provides a complete, standard-based security system built on top of its **Dependency Injection** mechanism. Rather than a monolithic auth framework, FastAPI provides composable building blocks:

- `OAuth2PasswordBearer` — for token extraction
- `OAuth2PasswordRequestForm` — for login form handling
- `HTTPBearer`, `HTTPBasic`, `APIKeyHeader` — alternative schemes
- `Security()` — like `Depends()` but with OAuth scopes support

---

## Security Schemes Overview

| Scheme | Use Case | FastAPI Class |
|---|---|---|
| OAuth2 Password Flow | First-party apps (login form) | `OAuth2PasswordBearer` |
| OAuth2 Client Credentials | Service-to-service | Custom |
| Bearer Token (JWT) | Stateless APIs | `HTTPBearer` or custom |
| API Key (Header) | Simple service auth | `APIKeyHeader` |
| API Key (Query) | Webhook, simple integrations | `APIKeyQuery` |
| HTTP Basic | Dev/internal tools | `HTTPBasic` |

---

## Complete OAuth2 + JWT Implementation

### 1. Dependencies and Setup

```python
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel

# --- Constants ---
SECRET_KEY = "your-256-bit-secret"   # Use: openssl rand -hex 32
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 Scheme ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
```

### 2. Pydantic Models

```python
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None
    scopes: list[str] = []

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool = False
    role: str = "user"

class UserInDB(User):
    hashed_password: str
```

### 3. Helper Functions

```python
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(db: dict, username: str) -> UserInDB | None:
    if username in db:
        return UserInDB(**db[username])
    return None

def authenticate_user(db: dict, username: str, password: str) -> UserInDB | False:
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user
```

### 4. Dependencies

```python
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception

    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Type aliases
CurrentUser = Annotated[User, Depends(get_current_active_user)]
```

### 5. Endpoints

```python
app = FastAPI()

@app.post("/auth/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: CurrentUser) -> User:
    return current_user
```

---

## OAuth2 Scopes

Scopes allow fine-grained permission control:

```python
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from fastapi import Security

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "items:read": "Read items",
        "items:write": "Create and update items",
        "admin": "Admin access",
    }
)

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> User:
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, username=username)
    except InvalidTokenError:
        raise credentials_exception

    user = get_user(fake_db, username=token_data.username)
    if user is None:
        raise credentials_exception

    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user

# Usage — require specific scopes
@app.get("/items/", dependencies=[Security(get_current_user, scopes=["items:read"])])
async def list_items():
    ...

@app.post("/items/")
async def create_item(
    item: ItemCreate,
    user: User = Security(get_current_user, scopes=["items:write"])
):
    ...
```

---

## API Key Authentication

```python
from fastapi.security import APIKeyHeader, APIKeyQuery, APIKeyCookie

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.get("/items/", dependencies=[Depends(verify_api_key)])
async def list_items():
    ...
```

---

## HTTP Basic Auth

```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "secret")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
```

> **Security note**: Always use `secrets.compare_digest()` to prevent timing attacks. Never use `==` for password comparison.

---

## Role-Based Access Control (RBAC)

```python
from enum import Enum

class Role(str, Enum):
    user = "user"
    moderator = "moderator"
    admin = "admin"

def require_role(*roles: Role):
    """Dependency factory — returns a dependency that checks roles"""
    async def check_role(current_user: CurrentUser) -> User:
        if current_user.role not in [r.value for r in roles]:
            raise HTTPException(
                status_code=403,
                detail=f"Required roles: {[r.value for r in roles]}"
            )
        return current_user
    return check_role

# Usage
@app.get("/admin/users", dependencies=[Depends(require_role(Role.admin))])
async def list_all_users():
    ...

@app.post("/moderate/items", dependencies=[Depends(require_role(Role.admin, Role.moderator))])
async def moderate_item():
    ...
```

---

## Refresh Tokens Pattern

```python
class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

@app.post("/auth/token", response_model=TokenPair)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(401, "Incorrect credentials")

    access_token = create_access_token(
        {"sub": user.username, "type": "access"},
        expires_delta=timedelta(minutes=15)
    )
    refresh_token = create_access_token(
        {"sub": user.username, "type": "refresh"},
        expires_delta=timedelta(days=7)
    )
    return TokenPair(access_token=access_token, refresh_token=refresh_token)

@app.post("/auth/refresh", response_model=Token)
async def refresh_access_token(refresh_token: str = Body(...)):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
        username = payload["sub"]
    except InvalidTokenError:
        raise HTTPException(401, "Invalid refresh token")

    new_access_token = create_access_token(
        {"sub": username, "type": "access"},
        expires_delta=timedelta(minutes=15)
    )
    return Token(access_token=new_access_token, token_type="bearer")
```

---

## Timing Attack Prevention

```python
import secrets

# ❌ Vulnerable to timing attacks
if api_key == expected_key:
    ...

# ✅ Constant-time comparison
if not secrets.compare_digest(api_key, expected_key):
    raise HTTPException(403)
```

---

## Common Pitfalls

### 1. Hardcoded Secrets
```python
# ❌ Never hardcode
SECRET_KEY = "mysecret"

# ✅ Load from environment
import os
SECRET_KEY = os.environ["JWT_SECRET_KEY"]
```

### 2. No Token Expiry
```python
# ❌ No expiry — tokens valid forever
payload = {"sub": username}
jwt.encode(payload, SECRET_KEY)

# ✅ Always set expiry
payload = {"sub": username, "exp": datetime.now(UTC) + timedelta(minutes=30)}
```

### 3. Using `==` for Secret Comparison
```python
# ❌ Timing attack vulnerable
if token == expected_token:

# ✅ Constant-time
if not secrets.compare_digest(token, expected_token):
```

### 4. Weak Password Hashing
```python
# ❌ Never use MD5 or SHA1 for passwords
hashlib.md5(password.encode()).hexdigest()

# ✅ Use bcrypt, argon2, or scrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
```

---

## Production Security Checklist

- [ ] Rotate `SECRET_KEY` regularly and store in secrets manager (AWS Secrets Manager, Vault)
- [ ] Use short-lived access tokens (15-30 minutes)
- [ ] Implement refresh token rotation (invalidate old refresh tokens on use)
- [ ] Log all authentication events (success, failure, suspicious patterns)
- [ ] Implement rate limiting on login endpoint
- [ ] Use HTTPS — never serve tokens over HTTP
- [ ] Set `HTTPOnly` and `Secure` flags on cookie-based tokens
- [ ] Validate token `aud` (audience) claim in multi-service architectures
- [ ] Implement token revocation (blocklist) for logout

---

## Related Topics
- [`05-dependency-injection.md`](./05-dependency-injection.md) — DI system powering auth
- [`07-middleware-cors.md`](./07-middleware-cors.md) — Middleware-based auth
- [`15-testing.md`](./15-testing.md) — Testing authenticated endpoints
