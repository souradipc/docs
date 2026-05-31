# WebSockets & Streaming

## WebSockets

### What Are WebSockets?

WebSockets provide a **persistent, bidirectional** communication channel between client and server over a single TCP connection. Unlike HTTP (request → response), WebSockets allow:
- Server → Client push (no polling needed)
- Client → Server messages at any time
- Low latency, real-time communication

**Use cases**: chat applications, live dashboards, collaborative editing, gaming, real-time notifications, AI streaming responses.

---

### Basic WebSocket Endpoint

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()    # Accept the connection
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
```

---

### WebSocket Lifecycle

```
1. Client sends HTTP upgrade request
2. Server accepts: await websocket.accept()
3. Bidirectional messages exchanged
4. Either side closes: await websocket.close()
   OR client disconnects (raises WebSocketDisconnect)
```

### WebSocket Message Types

```python
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive methods
            text = await websocket.receive_text()
            raw = await websocket.receive_bytes()
            data = await websocket.receive_json()

            # Send methods
            await websocket.send_text("Hello")
            await websocket.send_bytes(b"\x00\x01\x02")
            await websocket.send_json({"type": "message", "content": "Hello"})
    except WebSocketDisconnect as e:
        print(f"Disconnected with code: {e.code}")
    finally:
        # Cleanup
        pass
```

---

### WebSocket with Query Parameters & Headers

```python
from fastapi import WebSocket, Query, Depends

@app.websocket("/ws")
async def websocket_auth(
    websocket: WebSocket,
    token: str = Query(...),        # ?token=xxx in WS URL
    client_id: int = Query(...),
):
    # Validate token before accepting
    user = await validate_token(token)
    if not user:
        await websocket.close(code=1008)  # Policy violation
        return

    await websocket.accept()
    ...
```

---

### WebSocket Manager — Multiple Connections

```python
from fastapi import WebSocket, WebSocketDisconnect
from typing import Any

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_personal_message(self, client_id: str, message: Any):
        ws = self.active_connections.get(client_id)
        if ws:
            await ws.send_json(message)

    async def broadcast(self, message: Any, exclude: str | None = None):
        for client_id, ws in list(self.active_connections.items()):
            if client_id != exclude:
                try:
                    await ws.send_json(message)
                except Exception:
                    self.disconnect(client_id)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(client_id: str, websocket: WebSocket):
    await manager.connect(client_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Broadcast to all other clients
            await manager.broadcast(
                {"from": client_id, "message": data["text"]},
                exclude=client_id
            )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast({"system": f"{client_id} left the room"})
```

---

### WebSocket with Dependencies

```python
from fastapi import WebSocket, Depends, Cookie
from fastapi.websockets import WebSocketException
import starlette.status as status

async def get_ws_user(
    websocket: WebSocket,
    session: str | None = Cookie(default=None),
    token: str | None = Query(default=None),
):
    if not session and not token:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    return await validate_auth(session or token)

@app.websocket("/ws")
async def protected_ws(
    websocket: WebSocket,
    user: User = Depends(get_ws_user),
):
    await websocket.accept()
    await websocket.send_json({"welcome": user.username})
    ...
```

---

## Streaming Responses

### `StreamingResponse` — For Large or Dynamic Content

`StreamingResponse` sends content incrementally without buffering the entire response:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

# Async generator streaming
async def generate_large_csv():
    yield "id,name,value\n"
    for i in range(1_000_000):
        yield f"{i},item_{i},{i * 1.5}\n"
        if i % 1000 == 0:
            await asyncio.sleep(0)  # Yield control to event loop

@app.get("/export/csv")
async def export_csv():
    return StreamingResponse(
        content=generate_large_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"}
    )
```

### Streaming from an External Source

```python
import httpx

@app.get("/proxy/stream")
async def proxy_stream(url: str):
    async def stream_content():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    yield chunk

    return StreamingResponse(
        content=stream_content(),
        media_type="application/octet-stream"
    )
```

---

## Server-Sent Events (SSE)

SSE is a one-way streaming protocol — server pushes events to the browser over HTTP. Simpler than WebSockets when you only need server → client communication.

```python
import asyncio
import json
from fastapi.responses import StreamingResponse

async def event_stream(user_id: int):
    """Generate SSE-formatted events"""
    while True:
        # Fetch latest data
        events = await fetch_new_events(user_id)
        for event in events:
            # SSE format: "data: {json}\n\n"
            yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(1)  # Poll interval

@app.get("/events/stream")
async def stream_events(user_id: int):
    return StreamingResponse(
        content=event_stream(user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

### Named SSE Events

```python
async def event_stream():
    # Event with type
    yield f"event: user_joined\ndata: {json.dumps({'user': 'alice'})}\n\n"

    # Event with ID (for reconnection)
    yield f"id: 42\ndata: {json.dumps({'msg': 'hello'})}\n\n"

    # Heartbeat (keep connection alive through proxies)
    yield ": keepalive\n\n"
```

---

## AI Streaming Responses (LLM Pattern)

Common pattern for streaming LLM token output:

```python
from fastapi.responses import StreamingResponse
import httpx
import json

@app.post("/chat/stream")
async def stream_chat(prompt: str):
    async def generate():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                yield f"data: {json.dumps({'text': content})}\n\n"
                        except json.JSONDecodeError:
                            pass

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## `FileResponse` — Serving Files

```python
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"/data/files/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    return FileResponse(
        path=file_path,
        filename=filename,                    # Download filename
        media_type="application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"}
    )
```

---

## WebSocket Close Codes

```python
# Standard close codes
1000 — Normal closure
1001 — Going away (server restart, page navigation)
1002 — Protocol error
1003 — Unsupported data type
1008 — Policy violation (authentication failure)
1011 — Server error
1012 — Service restart
1013 — Try again later

await websocket.close(code=1000, reason="Goodbye")
```

---

## Common Pitfalls

### 1. Not Handling Disconnect
```python
# ❌ No disconnect handling — coroutine runs forever if client leaves
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()  # Raises WebSocketDisconnect on close

# ✅ Handle disconnection
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        pass  # Clean exit
```

### 2. Nginx Buffering SSE Streams
```nginx
# ❌ Nginx buffers SSE by default
location /events {
    proxy_pass http://backend;
}

# ✅ Disable buffering for SSE
location /events {
    proxy_pass http://backend;
    proxy_buffering off;
    proxy_cache off;
}
```

### 3. Memory Leak in Connection Manager
```python
# ❌ Connections never removed on error
self.active_connections.append(websocket)

# ✅ Always remove on disconnect
try:
    ...
except WebSocketDisconnect:
    manager.disconnect(client_id)
```

---

## Best Practices

1. **Always handle `WebSocketDisconnect`** in a try/except
2. **Authenticate before `accept()`** — close with `1008` if auth fails
3. **Use a Connection Manager** class for multi-client scenarios
4. **Implement heartbeats** for long-lived connections (proxy timeouts)
5. **Set `X-Accel-Buffering: no`** header for SSE when behind nginx
6. **Use Redis Pub/Sub** for WebSocket broadcasting across multiple workers
7. **Limit connections per user** to prevent resource exhaustion

---

## Related Topics
- [`09-async-concurrency.md`](./09-async-concurrency.md) — Async patterns for streaming
- [`12-background-tasks-lifespan.md`](./12-background-tasks-lifespan.md) — Resource management
- [`16-deployment-production.md`](./16-deployment-production.md) — Reverse proxy configuration
