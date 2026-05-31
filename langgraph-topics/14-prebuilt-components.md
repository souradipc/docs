# Prebuilt Components

## Overview

LangGraph ships prebuilt components that eliminate boilerplate for common patterns. These are production-ready but fully customizable.

| Component | Module | Purpose |
|---|---|---|
| `create_react_agent` | `langgraph.prebuilt` | One-line ReAct agent |
| `ToolNode` | `langgraph.prebuilt` | Execute tools from AI messages |
| `tools_condition` | `langgraph.prebuilt` | Route to tools or END |
| `MessagesState` | `langgraph.graph` | State with `messages` + `add_messages` |
| `InjectedToolArg` | `langgraph.prebuilt` | Inject runtime args into tools |

---

## MessagesState

The simplest state for conversational agents:

```python
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import BaseMessage

# MessagesState is equivalent to this TypedDict:
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Use it directly
from langgraph.graph import MessagesState, StateGraph

builder = StateGraph(MessagesState)
```

`add_messages` reducer:
- Appends new messages by default
- **Deduplicates by `id`** — if a message with the same `id` exists, it's replaced (useful for streaming partial updates)

---

## create_react_agent — Full Reference

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Minimal
agent = create_react_agent(model=llm, tools=[search, calculator])

# Full options
agent = create_react_agent(
    model=llm,
    tools=[search, calculator, code_interpreter],

    # Modify the system prompt
    prompt="You are a helpful assistant specializing in cybersecurity.",

    # OR: dynamic state modifier (called before every LLM call)
    # state_modifier=my_state_modifier_fn,

    # State schema (extend MessagesState with custom fields)
    state_schema=AgentState,        # Must extend MessagesState

    # Persistence
    checkpointer=checkpointer,

    # HITL breakpoints
    interrupt_before=["tools"],     # Pause before tool execution
    interrupt_after=["agent"],      # Pause after LLM response

    # Long-term memory
    store=store,

    # Tool call behavior
    # tools=ToolNode(tools, handle_tool_errors=True),  # Can pass a ToolNode directly
)

result = agent.invoke({"messages": [HumanMessage("What is 2+2?")]})
```

### Dynamic State Modifier

```python
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState

def state_modifier(state: MessagesState) -> list:
    """Called before each LLM invocation — returns the messages to send."""
    system = SystemMessage(
        content=f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
                f"User ID: {state.get('user_id', 'unknown')}"
    )
    return [system] + state["messages"]

agent = create_react_agent(model=llm, tools=tools, state_modifier=state_modifier)
```

### Custom State Schema

```python
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    user_id: str
    permissions: list[str]
    context: dict

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=AgentState,
)

result = agent.invoke({
    "messages": [HumanMessage("Run security scan")],
    "user_id": "user-123",
    "permissions": ["read", "scan"],
    "context": {"tenant": "acme"},
})
```

---

## ToolNode — Executing Tools

`ToolNode` is a node that:
1. Reads the last AI message for tool calls
2. Executes all tool calls (in parallel if multiple)
3. Returns `ToolMessage` results

```python
from langgraph.prebuilt import ToolNode

tools = [search, calculator, code_interpreter]
tool_node = ToolNode(tools)

# Direct use in graph
builder.add_node("tools", tool_node)
```

### Error Handling in ToolNode

```python
# Default: errors become ToolMessage content (won't crash the graph)
tool_node = ToolNode(tools, handle_tool_errors=True)

# Custom error handler
def my_error_handler(error: Exception, tool_call: dict) -> str:
    return f"Tool failed: {type(error).__name__}: {str(error)}"

tool_node = ToolNode(tools, handle_tool_errors=my_error_handler)
```

### Parallel Tool Execution

`ToolNode` automatically executes **multiple tool calls in parallel** when an LLM returns multiple tool calls in one message:

```python
# LLM can call multiple tools in one turn
# ToolNode runs them concurrently and returns all results
```

---

## tools_condition — Standard Routing

Route to `tools` if there are pending tool calls, otherwise to `END`:

```python
from langgraph.prebuilt import tools_condition

builder.add_conditional_edges(
    "agent",
    tools_condition,
    # Default mapping: {"tools": "tools", "__end__": END}
    # Or override with custom routing
    {
        "tools": "tools",
        "__end__": "post_process",   # Custom terminal node
    }
)
```

`tools_condition` returns `"tools"` if the last AI message has tool calls, `"__end__"` otherwise.

---

## InjectedToolArg — Runtime Dependency Injection

Inject runtime values (store, config, state) into tool functions without exposing them as LLM parameters:

```python
from langgraph.prebuilt import InjectedToolArg
from langgraph.store.base import BaseStore
from typing import Annotated

# Store injection
async def save_memory(
    content: str,                                           # LLM provides this
    store: Annotated[BaseStore, InjectedToolArg],          # Injected at runtime
) -> str:
    """Save a memory. Use this when user shares important information."""
    await store.aput(("memories",), uuid4().hex, {"text": content})
    return f"Memory saved: {content}"

# Config injection
from langchain_core.runnables import RunnableConfig

async def get_user_data(
    field: str,                                              # LLM provides this
    config: Annotated[RunnableConfig, InjectedToolArg],    # Injected
) -> str:
    """Get user data field."""
    user_id = config["configurable"]["user_id"]
    return await user_db.get(user_id, field)

# State injection
async def search_with_context(
    query: str,
    state: Annotated[AgentState, InjectedToolArg],         # Full state injected
) -> list[str]:
    """Search using the user's current context."""
    context = state["context"]
    return await search_api(query, context=context)

# Wire up with create_react_agent (store is auto-injected)
agent = create_react_agent(
    model=llm,
    tools=[save_memory, get_user_data, search_with_context],
    store=store,
)
```

---

## Building a Production Agent with Prebuilt Components

```python
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition, InjectedToolArg
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import Annotated
from langchain_openai import ChatOpenAI

# Tools
async def web_search(query: str) -> str:
    """Search the web for information."""
    return await search_api(query)

async def remember(
    fact: str,
    store: Annotated[BaseStore, InjectedToolArg],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Remember an important fact about the user."""
    user_id = config["configurable"]["user_id"]
    await store.aput((user_id, "facts"), uuid4().hex, {"text": fact})
    return "Remembered!"

# Custom state
class MyState(MessagesState):
    user_id: str

# System prompt using state
def system_prompt(state: MyState) -> list:
    return [
        SystemMessage(f"You are a helpful assistant for user {state['user_id']}."),
        *state["messages"],
    ]

# Build agent
async with AsyncSqliteSaver.from_conn_string("agent.db") as checkpointer:
    store = InMemoryStore()
    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[web_search, remember],
        state_schema=MyState,
        state_modifier=system_prompt,
        checkpointer=checkpointer,
        store=store,
    )

    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("My name is Alice")],
            "user_id": "user-123",
        },
        config={"configurable": {"thread_id": "t1", "user_id": "user-123"}},
    )
```

---

## When to Use Prebuilt vs Custom Graph

| Use Case | Recommendation |
|---|---|
| Standard ReAct agent | `create_react_agent` |
| ReAct + custom state fields | `create_react_agent` with `state_schema` |
| Complex multi-agent routing | Custom `StateGraph` |
| Non-tool loops (validation, retry) | Custom `StateGraph` |
| Parallel tool branches | Custom `StateGraph` with `ToolNode` |
| Tool execution only | `ToolNode` in custom graph |

---

## Related Topics
- [`01-state-graph-fundamentals.md`](./01-state-graph-fundamentals.md) — Building custom graphs
- [`03-nodes-edges-routing.md`](./03-nodes-edges-routing.md) — Custom routing
- [`05-human-in-the-loop.md`](./05-human-in-the-loop.md) — HITL with `interrupt_before`
- [`07-multi-agent-systems.md`](./07-multi-agent-systems.md) — Multi-agent with prebuilt agents
- [`10-memory-store.md`](./10-memory-store.md) — Store with `InjectedToolArg`
