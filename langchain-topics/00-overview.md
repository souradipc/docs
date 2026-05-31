# LangChain Overview — Architecture, Philosophy & Ecosystem

## What is LangChain?

LangChain is an open-source framework for building applications powered by **large language models (LLMs)**. It provides standardized interfaces, composable abstractions, and an extensive integration ecosystem — enabling developers to build production-grade AI applications without rewriting common patterns.

LangChain is **not** a model provider or a cloud service. It is a **composition layer** that connects models, data, tools, and memory into coherent workflows.

---

## Why LangChain Exists

Before frameworks like LangChain, building with LLMs meant:

- Manually formatting prompts for each provider
- Hand-rolling retry/fallback logic for API calls
- No standard way to add retrieval, tools, or memory
- No observability into what prompts were sent or what responses came back
- Every team solving the same plumbing problems from scratch

LangChain abstracts this plumbing and provides:

1. **Uniform interfaces** across model providers (OpenAI, Anthropic, Google, Cohere, local models)
2. **Composable chains** via LCEL (LangChain Expression Language)
3. **Built-in integrations** for 200+ vector stores, document loaders, and tools
4. **Agent frameworks** for multi-step reasoning with tool use
5. **Observability** via LangSmith

---

## The LangChain Ecosystem

```
┌────────────────────────────────────────────────────────────┐
│                     LangChain Ecosystem                     │
│                                                            │
│  langchain-core      → Interfaces, base classes, LCEL      │
│  langchain           → High-level chains, agents, helpers  │
│  langchain-community → 3rd party integrations              │
│  langchain-openai    → OpenAI-specific integration         │
│  langchain-anthropic → Anthropic integration               │
│  langchain-google-*  → Google AI integrations              │
│  langchain-text-splitters → Text chunking utilities        │
│                                                            │
│  LangGraph           → Stateful agent graphs               │
│  LangSmith           → Observability + evaluation          │
│  LangServe           → Serve chains as REST APIs           │
└────────────────────────────────────────────────────────────┘
```

### Package Structure (Python)

```bash
pip install langchain-core          # Core abstractions — always needed
pip install langchain               # High-level chains and agents
pip install langchain-openai        # OpenAI models + embeddings
pip install langchain-anthropic     # Anthropic Claude models
pip install langchain-community     # Community integrations
pip install langchain-text-splitters # Text chunking
pip install langgraph               # Stateful agent orchestration
pip install langsmith               # Observability SDK
pip install langchain-chroma        # Chroma vector store
pip install langchain-pinecone      # Pinecone vector store
```

---

## Core Abstractions

LangChain is built on a hierarchy of composable primitives:

### 1. Models
Wrappers around LLM APIs with uniform interface:
- **ChatModels** → Accept messages, return messages (modern)
- **LLMs** → Accept strings, return strings (legacy)
- **Embeddings** → Convert text to vectors

### 2. Messages
Structured conversation turns:
- `HumanMessage` — from the user
- `AIMessage` — from the model
- `SystemMessage` — instructions/context
- `ToolMessage` — tool call results
- `FunctionMessage` — legacy function call results

### 3. Prompts
Templates for constructing model inputs:
- `ChatPromptTemplate` — multi-message prompt templates
- `PromptTemplate` — single string templates
- `FewShotPromptTemplate` — example-based prompts
- `MessagesPlaceholder` — inject dynamic message history

### 4. Output Parsers
Transform model output strings/messages into structured data:
- `StrOutputParser` — extract string from AIMessage
- `JsonOutputParser` — parse JSON output
- `PydanticOutputParser` — parse into Pydantic models
- `StructuredOutputParser` — parse into dicts

### 5. Retrievers & Vector Stores
Retrieve relevant documents at query time:
- `VectorStore` — store and search embeddings
- `Retriever` — interface for document retrieval
- `DocumentLoader` — ingest raw documents
- `TextSplitter` — chunk documents into pieces

### 6. Tools
Functions the LLM can call:
- `@tool` decorator — convert Python function to tool
- `StructuredTool` — tool with structured input
- Built-in tools: web search, calculator, Python REPL, etc.

### 7. Memory & State
Manage conversation history:
- LangGraph's `MemorySaver` (checkpointing) — recommended
- Legacy `ConversationBufferMemory`, `ConversationSummaryMemory`

---

## The Request Lifecycle

```
User Input
    ↓
Prompt Template → formats into messages
    ↓
Chat Model → calls LLM API, returns AIMessage
    ↓
Output Parser → extracts/validates output
    ↓
(Optional) Tool Call → executes tool, adds ToolMessage
    ↓
(Optional) Retriever → fetches context docs
    ↓
Structured Output → returned to caller
```

---

## LCEL — The Composition Model

LangChain Expression Language (LCEL) is the modern way to compose these primitives using the `|` operator (pipe):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Each component is a Runnable
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain via | operator — returns a Runnable
chain = prompt | model | parser

# Invoke synchronously
result = chain.invoke({"question": "What is LangChain?"})

# Stream tokens
for chunk in chain.stream({"question": "What is LangChain?"}):
    print(chunk, end="", flush=True)

# Batch multiple inputs concurrently
results = chain.batch([
    {"question": "What is LangChain?"},
    {"question": "What is LangGraph?"},
])
```

Every component in the chain is a **`Runnable`** — implementing `.invoke()`, `.stream()`, `.batch()`, and `.astream()`.

---

## LangGraph — The Agent Layer

LangGraph is LangChain's stateful orchestration layer for **complex, multi-step agents**. It models agents as **directed graphs** with:

- **Nodes** — processing steps (LLM calls, tool calls, transformations)
- **Edges** — transitions between steps (fixed or conditional)
- **State** — shared typed dictionary persisted across steps
- **Checkpointers** — persist state for resumability and human-in-the-loop

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list
    context: str

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
```

---

## Version History & Stability

| Version | Key Change |
|---|---|
| v0.1.x | Initial public stable release |
| v0.2.x | Deprecated legacy chains, pushed LCEL |
| v0.3.x | Pydantic v2, improved streaming, modular packages |
| v0.3+ | `langchain-core` is the stable foundation |

> **Modern guidance**: Use `langchain-core` for building new abstractions. Use `langchain` for high-level utilities. Use `LangGraph` for agent workflows — not the legacy `AgentExecutor`.

---

## When to Use LangChain vs Building from Scratch

### Use LangChain when:
- You need rapid prototyping with multiple LLM providers
- You want uniform streaming/batching across providers
- You need RAG patterns (document loading, chunking, retrieval)
- You're building agents with tool use
- You need observability without building it yourself

### Build from scratch when:
- You have a very simple, single-LLM application (just use `httpx`)
- You need absolute control over prompt format and request structure
- LangChain abstractions add confusion for your use case
- You're in an environment with strict dependency budgets

---

## Architectural Tradeoffs

| Aspect | LangChain Approach | Tradeoff |
|---|---|---|
| Abstraction | High-level interfaces | Easier to start; harder to debug at depth |
| Integration | 200+ built-in integrations | Large dependency tree |
| Flexibility | Composable Runnables | Learning curve for LCEL |
| Agents | LangGraph (recommended) | LangGraph has its own learning curve |
| Observability | LangSmith (SaaS) | Vendor lock-in for full observability |

---

## Cost Considerations

LangChain itself is free and open-source. Costs come from:

1. **LLM API calls** — OpenAI, Anthropic, etc. (per token)
2. **Embedding API calls** — OpenAI, Cohere, etc. (per token)
3. **Vector databases** — Pinecone, Weaviate, etc. (hosted plans)
4. **LangSmith** — free tier available; paid plans for production scale
5. **Compute** — running local models, vector DBs

---

## Related Topics
- [`02-lcel-runnables.md`](./02-lcel-runnables.md) — LCEL composition system
- [`03-chat-models-llms.md`](./03-chat-models-llms.md) — Model interfaces
- [`07-rag-pipeline.md`](./07-rag-pipeline.md) — Full RAG architecture
- [`09-agents-tools.md`](./09-agents-tools.md) — Agents and tool calling
- [`11-langgraph.md`](./11-langgraph.md) — LangGraph stateful agents
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — Observability
