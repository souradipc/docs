# Agents & Tools

## What Is an Agent?

An agent uses an LLM as a reasoning engine to decide which tools to call, in what order, with what arguments — and when to return a final answer. Unlike a chain (fixed execution path), an agent dynamically selects actions based on model reasoning.

```
User Input
    ↓
LLM (reason: "I need to look up X")
    ↓
Tool Call: search("X")
    ↓
Tool Result: "X is..."
    ↓
LLM (reason: "I have enough info to answer")
    ↓
Final Answer
```

---

## Defining Tools

### `@tool` Decorator — Preferred

```python
from langchain_core.tools import tool
from typing import Optional

@tool
def get_threat_intel(ioc: str, ioc_type: str = "ip") -> str:
    """Look up threat intelligence for an indicator of compromise.

    Args:
        ioc: The indicator of compromise (IP, domain, hash, URL)
        ioc_type: Type of IOC - 'ip', 'domain', 'hash', or 'url'
    """
    # Real implementation would call a threat intel API
    return f"Threat intel for {ioc_type} {ioc}: High risk, associated with APT28"

# Tool metadata
print(get_threat_intel.name)         # "get_threat_intel"
print(get_threat_intel.description)  # The docstring
print(get_threat_intel.args)         # {"ioc": {"type": "string"}, ...}
```

### `StructuredTool` — Explicit Schema

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class ScanInput(BaseModel):
    target: str = Field(description="Target IP or hostname to scan")
    ports: list[int] = Field(default=[80, 443], description="Ports to scan")
    timeout: int = Field(default=30, description="Timeout in seconds")

def run_scan(target: str, ports: list[int], timeout: int) -> str:
    """Run a port scan on target."""
    return f"Scan of {target}:{ports} completed. Open: {ports[:1]}"

scan_tool = StructuredTool.from_function(
    func=run_scan,
    name="port_scan",
    description="Scan target for open ports",
    args_schema=ScanInput,
)
```

### Tool with Async Support

```python
from langchain_core.tools import tool

@tool
async def async_lookup(query: str) -> str:
    """Look up information asynchronously."""
    import asyncio
    await asyncio.sleep(0.1)  # Simulated async API call
    return f"Result for: {query}"
```

---

## Creating an Agent (LangGraph Approach — Recommended)

The modern way to create agents is with `create_react_agent` from LangGraph:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # Integrate with Tavily, DuckDuckGo, etc.
    return f"Web results for '{query}': ..."

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

tools = [search_web, calculator]
model = ChatOpenAI(model="gpt-4o")

# Create agent
agent = create_react_agent(model, tools)

# Run agent
result = agent.invoke({
    "messages": [("human", "What is 25 * 4 + the current year?")]
})

# Get the final response
print(result["messages"][-1].content)
```

---

## ReAct Loop — How It Works

The ReAct (Reasoning + Acting) pattern:

1. **Reason**: LLM sees the input and prior tool results, decides what to do next
2. **Act**: LLM outputs a tool call (name + args)
3. **Observe**: Tool executes, result added to context
4. **Repeat** until LLM decides to give a final answer

```
[HumanMessage] "What's the weather in Paris and London?"
    ↓
[AIMessage] tool_calls=[get_weather(city="Paris"), get_weather(city="London")]
    ↓
[ToolMessage] "Paris: 22°C"
[ToolMessage] "London: 15°C"
    ↓
[AIMessage] "Paris is 22°C and London is 15°C."
```

---

## Built-in Tools

```python
# Tavily web search
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(max_results=3)

# Wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Python REPL (execute code)
from langchain_experimental.tools import PythonREPLTool
repl = PythonREPLTool()

# Shell (dangerous! sandboxed use only)
from langchain_community.tools import ShellTool
shell = ShellTool()

# SQL Database
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///data.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = toolkit.get_tools()

# DuckDuckGo search (no API key)
from langchain_community.tools import DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()
```

---

## Passing Context to Tools at Runtime

Sometimes tools need runtime context (user ID, session info):

```python
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from typing import Annotated

# Method 1: Via RunnableConfig (injected at runtime)
@tool
def get_user_data(query: str, config: RunnableConfig) -> str:
    """Get data for the current user."""
    user_id = config["configurable"].get("user_id")
    return f"Data for user {user_id}: {query}"

# Inject at invoke time
agent.invoke(
    {"messages": [("human", "Show my profile")]},
    config={"configurable": {"user_id": "alice123"}},
)

# Method 2: Annotated injection
@tool
def look_up_account(
    query: str,
    user_id: Annotated[str, InjectedToolArg],  # Not in schema, injected
) -> str:
    """Look up account information."""
    return f"Account for {user_id}: {query}"
```

---

## Tool Error Handling

```python
from langchain_core.tools import ToolException

@tool
def risky_lookup(ioc: str) -> str:
    """Look up threat intel — may fail for unknown IOCs."""
    if not ioc.startswith("192."):
        raise ToolException(f"Unknown IOC format: {ioc}. Expected IP address.")
    return f"Safe IP: {ioc}"

# Agents automatically handle ToolException and retry or route differently

# Explicitly configure error handling
risky_lookup_with_fallback = risky_lookup.with_config(
    handle_tool_error=True,  # Return error message as tool result instead of crash
)

# Custom error handler
def handle_error(error: ToolException) -> str:
    return f"Tool failed: {error}. Please try with a different input."

risky_lookup_graceful = risky_lookup.with_config(
    handle_tool_error=handle_error,
)
```

---

## Agents with Memory (Checkpointing)

Add memory to agents using LangGraph's checkpointing:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "user-alice-session-1"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [("human", "My name is Alice.")]},
    config=config,
)

# Turn 2 — agent remembers Alice
result2 = agent.invoke(
    {"messages": [("human", "What's my name?")]},
    config=config,
)
print(result2["messages"][-1].content)  # "Your name is Alice."
```

---

## System Prompts for Agents

```python
from langgraph.prebuilt import create_react_agent

system_prompt = """You are a cybersecurity analyst assistant with access to threat intelligence tools.

When analyzing security incidents:
1. First, gather threat intelligence on the IOC
2. Check if it's associated with known threat actors
3. Provide actionable remediation steps
4. Always cite your sources

Available capabilities: threat intel lookup, malware analysis, vulnerability search.
"""

agent = create_react_agent(
    model,
    tools,
    state_modifier=system_prompt,  # System prompt for the agent
)
```

---

## Multi-Tool Agents (Parallel Tool Calling)

Modern LLMs can call multiple tools simultaneously:

```python
from langchain_openai import ChatOpenAI

# GPT-4o can call multiple tools in one turn
model = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(model, tools)

result = agent.invoke({
    "messages": [("human", "Check the weather in Paris, London, and Tokyo simultaneously")]
})

# The AIMessage will have multiple tool_calls
# Agent will dispatch all 3 weather checks in parallel → faster execution
```

---

## Agent Streaming

```python
# Stream agent steps as they happen
for event in agent.stream(
    {"messages": [("human", "Analyze this IP: 1.2.3.4")]},
    stream_mode="values",  # Stream full state at each step
):
    # Show the latest message
    latest_message = event["messages"][-1]
    print(f"[{type(latest_message).__name__}]: {latest_message.content[:100]}")

# Stream individual tokens
for chunk, metadata in agent.stream(
    {"messages": [("human", "Analyze this IP")]},
    stream_mode="messages",  # Stream individual message chunks
):
    if hasattr(chunk, "content"):
        print(chunk.content, end="", flush=True)
```

---

## Security Considerations

### Sandboxing Code Execution Tools
```python
# ❌ Dangerous — arbitrary code execution
from langchain_experimental.tools import PythonREPLTool
repl = PythonREPLTool()  # User can run any Python code

# ✅ Limit what agent can do
@tool
def safe_calculate(expression: str) -> str:
    """Calculate a math expression safely."""
    # Only allow math operations
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        raise ToolException("Only math expressions are allowed.")
    return str(eval(expression))
```

### Input Validation on Tools
```python
@tool
def query_database(sql: str) -> str:
    """Query the database. Only SELECT statements allowed."""
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        raise ToolException("Only SELECT queries are permitted.")
    if any(keyword in sql_upper for keyword in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        raise ToolException("Destructive operations are not allowed.")
    # Run the query
    return run_query(sql)
```

---

## Common Pitfalls

### 1. Vague Tool Descriptions
```python
# ❌ Model can't decide when to use this
@tool
def lookup(x: str) -> str:
    """Look up something."""
    ...

# ✅ Specific, actionable description
@tool
def lookup_ip_reputation(ip_address: str) -> str:
    """Look up the threat reputation of an IP address.
    Returns risk score (0-100), associated malware families,
    and known threat actor attribution if available.
    Use this when investigating suspicious IP addresses from security alerts."""
    ...
```

### 2. Infinite Agent Loops
```python
# ✅ Set recursion limit to prevent infinite loops
result = agent.invoke(
    {"messages": [("human", "Research everything about security.")]},
    config={"recursion_limit": 25},  # Max 25 LLM calls
)
```

---

## Related Topics
- [`11-langgraph.md`](./11-langgraph.md) — Building custom agents with LangGraph
- [`10-memory-conversation-history.md`](./10-memory-conversation-history.md) — Agent memory
- [`12-langgraph-persistence.md`](./12-langgraph-persistence-checkpointing.md) — Persistent agent state
