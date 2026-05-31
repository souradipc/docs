# Multi-Agent Systems

## Why Multi-Agent?

Single-agent systems have limits:
- **Context window**: One agent can't hold arbitrarily large working context
- **Specialization**: A generalist agent performs worse than specialists for complex domains
- **Parallelism**: One agent does things sequentially; multiple agents can parallelize
- **Reliability**: One agent failure kills the entire task; multi-agent can isolate failures
- **Scalability**: Different agents can run on different hardware/models based on task needs

LangGraph supports composing multiple agents as subgraphs, with explicit handoffs and shared state.

---

## The Two Core Patterns

### Pattern 1: Supervisor (Hierarchical)

A supervisor agent orchestrates specialized subagents:

```
         Supervisor Agent
        /       |        \
   Research   Writing   Review
    Agent      Agent     Agent
```

The supervisor decides which subagent to call based on the task, receives results, and decides what to do next.

### Pattern 2: Handoff (Sequential/Peer-to-Peer)

Agents pass control directly to each other:

```
Agent A → (decides to hand off) → Agent B → (decides to hand off) → Agent C → END
```

Each agent can see the full conversation history and explicitly route to another agent.

---

## Pattern 1: Supervisor with Subagents as Tools

The cleanest approach: subagents are wrapped as tools that the supervisor can call:

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Define specialized agents
research_agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[web_search, arxiv_search],
    state_modifier="You are a research specialist. Provide thorough, cited research.",
)

analysis_agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[code_interpreter, data_analyzer],
    state_modifier="You are a data analyst. Provide rigorous quantitative analysis.",
)

# Wrap subagents as tools for the supervisor
@tool
def research_topic(topic: str) -> str:
    """Research a topic thoroughly using web and academic sources."""
    result = research_agent.invoke({"messages": [("human", topic)]})
    return result["messages"][-1].content

@tool
def analyze_data(data_description: str) -> str:
    """Perform quantitative analysis on data."""
    result = analysis_agent.invoke({"messages": [("human", data_description)]})
    return result["messages"][-1].content

# Supervisor orchestrates the specialized agents
supervisor = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[research_topic, analyze_data],
    state_modifier="""You are a project supervisor. Break down complex tasks,
delegate to specialists, and synthesize their outputs into a final report.""",
)

result = supervisor.invoke({
    "messages": [("human", "Analyze the current state of quantum computing in cybersecurity")]
})
```

---

## Pattern 2: Supervisor as a Router Graph

For more complex orchestration where the supervisor has explicit control flow:

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langchain_openai import ChatOpenAI

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str

MEMBERS = ["researcher", "analyst", "writer"]
SYSTEM_PROMPT = f"""You are a supervisor managing these agents: {MEMBERS}.
Given the conversation, decide which agent should act next.
When the task is complete, respond with "FINISH".
Respond with ONLY the agent name or FINISH."""

supervisor_llm = ChatOpenAI(model="gpt-4o")

def supervisor_node(state: SupervisorState) -> Command[Literal["researcher", "analyst", "writer", "__end__"]]:
    response = supervisor_llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        *[msg for msg in state["messages"]],
    ])
    next_step = response.content.strip()

    if next_step == "FINISH":
        return Command(goto=END)
    return Command(goto=next_step)

def make_agent_node(agent_name: str, tools: list, system_prompt: str):
    """Factory for creating agent nodes."""
    agent = create_react_agent(
        ChatOpenAI(model="gpt-4o"),
        tools=tools,
        state_modifier=system_prompt,
    )

    def agent_node(state: SupervisorState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        # Add agent's response tagged with its name
        last_msg = result["messages"][-1]
        last_msg.name = agent_name
        return {"messages": [last_msg]}

    return agent_node

# Build supervisor graph
builder = StateGraph(SupervisorState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", make_agent_node("researcher", [web_search], "You research topics."))
builder.add_node("analyst", make_agent_node("analyst", [code_exec], "You analyze data."))
builder.add_node("writer", make_agent_node("writer", [], "You write clear reports."))

builder.add_edge(START, "supervisor")
# After each agent, return to supervisor
for member in MEMBERS:
    builder.add_edge(member, "supervisor")

graph = builder.compile()
```

---

## Pattern 3: Agent Handoffs (Swarm)

Agents hand off to each other directly using `Command(goto=agent_name)`:

```python
from langgraph.types import Command
from langgraph.graph import StateGraph, START, MessagesState

def create_handoff_tool(*, agent_name: str, description: str):
    """Create a handoff tool that routes to a specific agent."""
    @tool(description=description)
    def handoff() -> Command:
        return Command(goto=agent_name)
    handoff.__name__ = f"handoff_to_{agent_name}"
    return handoff

# Agent-specific tools + handoff tools
billing_agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[
        check_balance,
        process_refund,
        handoff_to_tech := create_handoff_tool(
            agent_name="tech_agent",
            description="Hand off to tech support if this is a technical issue",
        ),
    ],
    state_modifier="You handle billing and payment questions.",
)

tech_agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=[
        check_system_status,
        create_ticket,
        handoff_to_billing := create_handoff_tool(
            agent_name="billing_agent",
            description="Hand off to billing if this is a billing question",
        ),
    ],
    state_modifier="You handle technical support questions.",
)

# Build swarm graph
def billing_node(state: MessagesState) -> Command:
    result = billing_agent.invoke(state)
    return Command(update={"messages": result["messages"]}, goto="supervisor")

def tech_node(state: MessagesState) -> Command:
    result = tech_agent.invoke(state)
    return Command(update={"messages": result["messages"]}, goto="supervisor")

builder = StateGraph(MessagesState)
builder.add_node("billing_agent", billing_node)
builder.add_node("tech_agent", tech_node)
# Initial router
builder.add_edge(START, "billing_agent")  # Default entry
```

---

## Subgraphs as Agent Nodes

Use compiled subgraphs directly as nodes when they share state keys:

```python
from langgraph.graph import StateGraph, MessagesState

# Sub-agent has MessagesState (shares "messages" key with parent)
sub_agent = (
    StateGraph(MessagesState)
    .add_node("llm", call_llm)
    .add_node("tools", ToolNode(tools))
    .add_edge(START, "llm")
    .add_conditional_edges("llm", tools_condition)
    .add_edge("tools", "llm")
    .compile()
)

# Parent graph uses sub-agent as a node directly
parent = (
    StateGraph(MessagesState)
    .add_node("router", route_request)
    .add_node("specialist", sub_agent)  # Direct subgraph reference
    .add_node("summarize", summarize_node)
    .add_edge(START, "router")
    .add_edge("router", "specialist")
    .add_edge("specialist", "summarize")
    .add_edge("summarize", END)
    .compile()
)
```

---

## Shared vs Isolated Memory in Multi-Agent

```python
# SHARED memory: all agents read/write the same checkpointer
config = {"configurable": {"thread_id": "shared-conversation-1"}}
# All agents in the graph contribute to the same conversation history

# ISOLATED memory per subagent: each subagent has its own checkpointer
# Use when agents should have private long-term memory

from langgraph.checkpoint.memory import InMemorySaver

agent_a_checkpointer = InMemorySaver()
agent_b_checkpointer = InMemorySaver()

agent_a = create_react_agent(model, tools_a, checkpointer=agent_a_checkpointer)
agent_b = create_react_agent(model, tools_b, checkpointer=agent_b_checkpointer)

# agent_a and agent_b have separate conversation histories
# but share the parent graph's state via the shared "messages" key
```

---

## Namespace Isolation for Subgraphs

When running multiple subgraph instances (e.g., multiple agent copies), use unique node names to prevent checkpoint collisions:

```python
def create_specialized_agent(model_name: str, *, name: str, tools: list):
    """Wrap agent with a unique node name for namespace isolation."""
    agent = create_react_agent(model_name, tools=tools)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)   # Unique node name
        .add_edge(START, name)
        .compile()
    )

# Each subagent has its own checkpoint namespace
security_agent = create_specialized_agent("gpt-4o", name="security_agent", tools=[sec_tools])
network_agent = create_specialized_agent("gpt-4o", name="network_agent", tools=[net_tools])
```

---

## Multi-Agent Design Guidelines

| Concern | Recommendation |
|---|---|
| How many agents? | Start with 2-3 focused agents; add more only when needed |
| State sharing | Share `messages` via `add_messages`; use separate state for agent-private data |
| Orchestration | Supervisor for structured tasks; handoffs for conversational routing |
| Model selection | Use cheaper models for routing/planning; premium models for complex reasoning |
| Error isolation | Wrap subagent calls in try/except; don't let one agent failure crash everything |
| Loops | Set `recursion_limit` to prevent infinite handoff loops |
| Observability | Tag each agent's messages with `name` attribute for trace visibility |

---

## Recursion Limit — Prevent Infinite Loops

```python
# Set a limit on how many steps the multi-agent graph can take
result = graph.invoke(
    state,
    config={
        "recursion_limit": 50,   # Default is 25
        "configurable": {"thread_id": "task-1"},
    },
)
```

---

## Related Topics
- [`08-subgraphs.md`](./08-subgraphs.md) — Nested graph details
- [`09-map-reduce-parallelism.md`](./09-map-reduce-parallelism.md) — Parallel agent execution
- [`14-prebuilt-components.md`](./14-prebuilt-components.md) — `create_react_agent`
