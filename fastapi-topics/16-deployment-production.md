# Deployment & Production

## The Production Stack

A production FastAPI deployment consists of:

```
Internet
    ↓
[ Load Balancer / CDN ]        (AWS ALB, Cloudflare, etc.)
    ↓
[ Reverse Proxy ]              (nginx, Traefik, Caddy)
    ↓
[ Gunicorn (process manager) ] (manages worker processes)
    ↓
[ Uvicorn (ASGI server) ]      (multiple workers, each handles concurrent requests)
    ↓
[ FastAPI Application ]
    ↓
[ Database / Redis / External Services ]
```

---

## ASGI Servers

### Uvicorn

The standard ASGI server for FastAPI:

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production — single process
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Production — with options
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --loop uvloop \
    --http httptools \
    --proxy-headers \            # Trust X-Forwarded-For from reverse proxy
    --forwarded-allow-ips "*" \  # Trust all reverse proxy IPs (or specific IP)
    --access-log \
    --log-level info
```

### FastAPI CLI

```bash
# Development
fastapi dev app/main.py

# Production
fastapi run app/main.py --port 8000 --workers 4
```

### Gunicorn + Uvicorn Workers (Recommended for Production)

Gunicorn manages multiple Uvicorn processes, providing:
- Automatic worker restart on crash
- Graceful reload without downtime
- Process management and monitoring

```bash
pip install gunicorn uvicorn[standard]

gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --max-requests 1000 \          # Restart worker after N requests (memory leak prevention)
    --max-requests-jitter 100 \   # Randomize restart to avoid thundering herd
    --access-logfile - \
    --error-logfile -
```

### Worker Count Formula

```
Workers = (2 × CPU cores) + 1

# For 4 CPU cores: 9 workers
# For 8 CPU cores: 17 workers
# For I/O-heavy apps: can go higher (up to 4× cores)
```

```bash
# Get optimal worker count at runtime
gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers $(python -c "import multiprocessing; print(2 * multiprocessing.cpu_count() + 1)") \
    --bind 0.0.0.0:8000
```

---

## Docker

### Dockerfile (Production)

```dockerfile
# ── Stage 1: Build dependencies ──────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt --target /app/deps

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/deps /app/deps
ENV PYTHONPATH=/app/deps

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser ./app /app/app

# Environment defaults (override at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "app.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

### `docker-compose.yml` (Full Stack)

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

---

## Nginx Configuration

```nginx
# nginx.conf
upstream fastapi_backend {
    server api:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    client_max_body_size 10M;

    # FastAPI API
    location /api/ {
        proxy_pass http://fastapi_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;     # WebSocket support
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
    }

    # SSE / Streaming — disable buffering
    location /api/events {
        proxy_pass http://fastapi_backend;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding on;
    }

    # OpenAPI docs
    location /docs {
        proxy_pass http://fastapi_backend;
    }
}
```

---

## Health Check Endpoint

Every production API must have a health check:

```python
from fastapi import FastAPI
from sqlmodel import Session, text

@app.get("/health")
async def health_check(db: SessionDep):
    checks = {"status": "healthy", "checks": {}}

    # DB connectivity
    try:
        db.exec(text("SELECT 1"))
        checks["checks"]["database"] = "healthy"
    except Exception as e:
        checks["checks"]["database"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    # Redis connectivity
    try:
        await redis.ping()
        checks["checks"]["redis"] = "healthy"
    except Exception as e:
        checks["checks"]["redis"] = f"unhealthy: {e}"
        checks["status"] = "degraded"

    status_code = 200 if checks["status"] == "healthy" else 503
    return JSONResponse(content=checks, status_code=status_code)
```

---

## Structured Logging

```python
# app/core/logging.py
import logging
import json
import sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)

def setup_logging(log_level: str = "INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers = [handler]

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

---

## Environment Variables in Production

```bash
# Never use .env files in production — use:
# 1. Container environment variables
docker run -e SECRET_KEY=xxx -e DATABASE_URL=yyy myapp

# 2. Kubernetes secrets
kubectl create secret generic app-secrets \
    --from-literal=SECRET_KEY=xxx \
    --from-literal=DATABASE_URL=yyy

# 3. AWS Parameter Store / Secrets Manager
# 4. HashiCorp Vault
```

---

## Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
        - name: api
          image: myrepo/myapp:v1.2.3
          ports:
            - containerPort: 8000
          env:
            - name: ENV
              value: "production"
            - name: SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: SECRET_KEY
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: DATABASE_URL
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

---

## Migrations in CI/CD

```bash
# In your deployment pipeline, run migrations before starting the app
alembic upgrade head && gunicorn app.main:app ...

# Or as an init container in Kubernetes
initContainers:
  - name: migrations
    image: myrepo/myapp:v1.2.3
    command: ["alembic", "upgrade", "head"]
```

---

## Observability — OpenTelemetry

```python
# app/core/observability.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_observability(app: FastAPI, settings: Settings):
    if not settings.OTEL_ENDPOINT:
        return

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=settings.OTEL_ENDPOINT)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI, SQLAlchemy, HTTPX
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument(engine=engine)
    HTTPXClientInstrumentor().instrument()
```

---

## Production Checklist

### Security
- [ ] HTTPS enforced everywhere
- [ ] CORS configured with explicit origins
- [ ] `TrustedHostMiddleware` configured
- [ ] Secrets in secrets manager (not env vars or .env)
- [ ] Rate limiting on public endpoints
- [ ] Input size limits (body size, pagination limits)
- [ ] SQL injection protection (ORM, parameterized queries)

### Reliability
- [ ] Health check endpoint (`/health`)
- [ ] Liveness and readiness probes in Kubernetes
- [ ] Database connection pool with `pool_pre_ping=True`
- [ ] Graceful shutdown (Gunicorn `--graceful-timeout`)
- [ ] Retry logic for external service calls
- [ ] Circuit breaker for downstream services

### Performance
- [ ] Multiple Uvicorn workers (Gunicorn)
- [ ] Connection pool sized per worker count
- [ ] Async for I/O-heavy operations
- [ ] Response compression (GZip middleware)
- [ ] CDN for static assets

### Observability
- [ ] Structured JSON logging
- [ ] Request ID in all logs and responses
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Metrics (Prometheus / Datadog)
- [ ] Error tracking (Sentry)
- [ ] Uptime monitoring

### Database
- [ ] Database migrations with Alembic
- [ ] Connection pooling configured
- [ ] Automated database backups
- [ ] Read replicas for read-heavy queries

---

## Common Pitfalls

### 1. Running Without Multiple Workers
```bash
# ❌ Single worker — can't use multiple CPU cores
uvicorn app.main:app

# ✅ Multiple workers
gunicorn app.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4
```

### 2. Forgetting `--proxy-headers`
```bash
# ❌ request.client.host shows load balancer IP instead of real client
uvicorn app.main:app

# ✅ Trust X-Forwarded-For from reverse proxy
uvicorn app.main:app --proxy-headers --forwarded-allow-ips "*"
```

### 3. `create_all()` in Production
```python
# ❌ Overwrites schema changes, not safe for production
SQLModel.metadata.create_all(engine)

# ✅ Use Alembic migrations
alembic upgrade head
```

---

## Related Topics
- [`12-background-tasks-lifespan.md`](./12-background-tasks-lifespan.md) — Startup resource management
- [`14-settings-configuration.md`](./14-settings-configuration.md) — Environment configuration
- [`09-async-concurrency.md`](./09-async-concurrency.md) — Async performance patterns
- [`10-database-integration.md`](./10-database-integration.md) — Connection pool configuration
