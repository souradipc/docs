# Memory & Conversation History

## Types of Memory

LLM applications need memory to maintain context across turns:

| Memory Type | Scope | Implementation |
|---|---|---|
| **In-context (short-term)** | Current session | Message list in prompt |
| **Persistent (long-term)** | Across sessions | Database + checkpointing |
| **Semantic (episodic)** | Key facts | Vector store + retrieval |
| **Summary memory** | Compressed history | LLM-generated summaries |

---

## The Fundamental Problem

LLMs are stateless. Every call starts fresh. To give an LLM "memory", you must pass prior messages in the `messages` list on every call.

```python
# This loses context on turn 2:
model.invoke("My name is Alice.")    # Turn 1
model.invoke("What's my name?")      # Turn 2 → "I don't know your name"

# This preserves context:
messages = [
    HumanMessage("My name is Alice."),
    AIMessage("Nice to meet you, Alice!"),
    HumanMessage("What's my name?"),
]
model.invoke(messages)  # "Your name is Alice."
```

---

## `InMemoryChatMessageHistory` — Simple Session Memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create history store (per session)
history = InMemoryChatMessageHistory()

# Add messages
history.add_user_message("My name is Alice.")
history.add_ai_message("Nice to meet you, Alice!")

# Access messages
print(history.messages)  # [HumanMessage(...), AIMessage(...)]

# Use with a chain
messages = history.messages + [HumanMessage("What's my name?")]
response = model.invoke(messages)
history.add_ai_message(response.content)
```

---

## `RunnableWithMessageHistory` — Auto-Managed History

Wrap any chain to automatically inject and update message history:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Storage dict (session_id → history)
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Chain definition
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()

# Wrap with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Use with session_id to maintain separate histories
config = {"configurable": {"session_id": "alice-session-1"}}

response1 = chain_with_history.invoke(
    {"input": "My name is Alice."},
    config=config,
)

response2 = chain_with_history.invoke(
    {"input": "What's my name?"},
    config=config,
)
print(response2)  # "Your name is Alice."
```

---

## Trimming Message History — Prevent Context Overflow

As conversation grows, the token count can exceed the model's context window. Trim messages to a manageable size:

```python
from langchain_core.messages import trim_messages, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Keep most recent messages that fit within token limit
trimmer = trim_messages(
    max_tokens=4096,              # Max tokens to keep
    strategy="last",              # Keep most recent messages
    token_counter=ChatOpenAI(model="gpt-4o"),  # Uses model's tokenizer
    include_system=True,          # Always keep system message
    allow_partial=False,          # Don't split messages
    start_on="human",             # Always start with a human message
)

# Apply trimmer to message list
trimmed = trimmer.invoke(all_messages)

# Use in a chain
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
```

---

## Message Summarization — Compress History

When history gets long, summarize it instead of discarding:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState

async def summarize_conversation(state: MessagesState):
    """Node that summarizes old messages when history gets long."""
    summary = state.get("summary", "")

    if summary:
        # Extend existing summary
        summary_message = (
            f"Previous summary: {summary}\n\n"
            f"Extend the summary with new messages above."
        )
    else:
        summary_message = "Summarize the conversation above in a few sentences."

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await ChatOpenAI(model="gpt-4o").ainvoke(messages)

    # Remove old messages, keep last 2
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": response.content, "messages": delete_messages}
```

---

## LangGraph Checkpointing — Persistent Memory

The most production-ready approach to memory — stores full conversation state in a database:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpointer (dev)
checkpointer = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=checkpointer)

# Thread ID = unique conversation session
config = {"configurable": {"thread_id": "user-123-thread-1"}}

# Turn 1
agent.invoke({"messages": [("human", "I work in security.")]}, config=config)

# Turn 2 — agent remembers
result = agent.invoke({"messages": [("human", "What field do I work in?")]}, config=config)
print(result["messages"][-1].content)  # "You work in security."
```

### PostgreSQL Persistent Checkpointing

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

async def setup_agent():
    async with AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost:5432/mydb"
    ) as checkpointer:
        await checkpointer.setup()  # Creates tables

        agent = create_react_agent(model, tools, checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "user-123-thread-1"}}

        # Conversations persist across application restarts
        result = await agent.ainvoke(
            {"messages": [("human", "Continue our previous discussion")]},
            config=config,
        )
        return result
```

### SQLite Persistent Checkpointing

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def setup_agent():
    async with AsyncSqliteSaver.from_conn_string("conversations.db") as checkpointer:
        await checkpointer.setup()
        agent = create_react_agent(model, tools, checkpointer=checkpointer)
        return agent
```

---

## Semantic / Long-Term Memory

For remembering facts across sessions without storing all messages:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class SemanticMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(embedding_function=self.embeddings)

    def save(self, user_id: str, key: str, value: str):
        """Save a fact to semantic memory."""
        doc = Document(
            page_content=f"{key}: {value}",
            metadata={"user_id": user_id, "key": key},
        )
        self.store.add_documents([doc])

    def recall(self, user_id: str, query: str, k: int = 3) -> list[str]:
        """Recall relevant memories for a query."""
        docs = self.store.similarity_search(
            query,
            k=k,
            filter={"user_id": user_id},
        )
        return [doc.page_content for doc in docs]

# Usage
memory = SemanticMemory()
memory.save("alice", "preference", "Prefers Python over JavaScript")
memory.save("alice", "role", "Works as a security analyst")

# Recall relevant facts
facts = memory.recall("alice", "programming language preference")
```

---

## Multi-User Memory Management

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from collections import defaultdict

class MemoryManager:
    """Manages conversation history for multiple users and sessions."""

    def __init__(self):
        # user_id:session_id → history
        self._store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(self, user_id: str, session_id: str):
        key = f"{user_id}:{session_id}"
        if key not in self._store:
            self._store[key] = InMemoryChatMessageHistory()
        return self._store[key]

    def clear_session(self, user_id: str, session_id: str):
        key = f"{user_id}:{session_id}"
        self._store.pop(key, None)

    def list_sessions(self, user_id: str) -> list[str]:
        return [
            key.split(":")[1]
            for key in self._store
            if key.startswith(f"{user_id}:")
        ]
```

---

## Common Pitfalls

### 1. Growing History Without Bounds
```python
# ❌ History grows forever → eventually exceeds context window
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
# After 100 turns, 100K tokens → API error

# ✅ Use trim_messages or summarization
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt | model
)
```

### 2. Using a Single Thread ID for All Users
```python
# ❌ All users share the same memory!
config = {"configurable": {"thread_id": "main"}}

# ✅ Unique thread per user session
config = {"configurable": {"thread_id": f"user-{user_id}-{session_id}"}}
```

### 3. Storing Sensitive Data in Plain Memory
```python
# ❌ PII stored in memory logs / LangSmith traces
history.add_user_message("My SSN is 123-45-6789")

# ✅ Sanitize sensitive data before storing
import re
def sanitize(text: str) -> str:
    # Mask SSN patterns
    return re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN REDACTED]', text)
```

---

## Related Topics
- [`02-lcel-runnables.md`](./02-lcel-runnables.md) — `RunnableWithMessageHistory`
- [`11-langgraph.md`](./11-langgraph.md) — Graph-based memory management
- [`12-langgraph-persistence-checkpointing.md`](./12-langgraph-persistence-checkpointing.md) — Persistent state
