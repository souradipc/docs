# LangChain Knowledge Base

A comprehensive topic-wise reference for the LangChain ecosystem, covering everything from basic chain composition to production-grade agents with persistence, observability, and cost control.

---

## Topics

| # | File | What It Covers |
|---|---|---|
| 00 | [Overview & Ecosystem](./00-overview.md) | LangChain architecture, packages, when to use, request lifecycle |
| 02 | [LCEL & Runnables](./02-lcel-runnables.md) | `\|` pipe operator, invoke/stream/batch, parallel, passthrough, fallbacks |
| 03 | [Chat Models & Embeddings](./03-chat-models-llms.md) | `init_chat_model`, streaming, structured output, caching, fallbacks |
| 04 | [Prompts & Templates](./04-prompts-templates.md) | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot, injection defense |
| 05 | [Output Parsers](./05-output-parsers-structured-output.md) | `with_structured_output`, Pydantic, JSON, auto-retry parsers |
| 06 | [Document Loaders & Splitters](./06-document-loaders-splitters.md) | PDF/web/CSV loaders, `RecursiveCharacterTextSplitter`, chunk strategies |
| 07 | [Embeddings & Vector Stores](./07-embeddings-vector-stores.md) | OpenAI/HuggingFace embeddings, Chroma/FAISS/Pinecone, MMR, hybrid search |
| 08 | [RAG Pipeline](./08-rag-pipeline.md) | Full RAG, conversational RAG, MultiQuery, HyDE, reranking |
| 09 | [Agents & Tools](./09-agents-tools.md) | `@tool`, `create_react_agent`, tool error handling, security |
| 10 | [Memory & Conversation History](./10-memory-conversation-history.md) | Short/long-term memory, trimming, summarization, multi-user |
| 11 | [LangGraph](./11-langgraph.md) | `StateGraph`, nodes, conditional edges, subgraphs, streaming |
| 12 | [LangGraph Persistence](./12-langgraph-persistence-checkpointing.md) | Checkpointers, human-in-the-loop, time travel, thread IDs |
| 13 | [Callbacks & Streaming](./13-callbacks-streaming.md) | `BaseCallbackHandler`, `astream_events`, FastAPI SSE streaming |
| 14 | [Evaluation & Testing](./14-evaluation-testing.md) | LangSmith datasets, evaluators, LLM-as-judge, regression testing |
| 15 | [LangSmith Observability](./15-langsmith-observability.md) | `@traceable`, tracing, experiments, production monitoring |
| 16 | [Production Deployment](./16-production-deployment.md) | LangServe, FastAPI integration, async, rate limiting, multi-tenancy |
| 17 | [Cost Optimization](./17-cost-optimization.md) | Token counting, caching, model tiering, embedding cost reduction |

---

## Quick Reference

### Minimal Working Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

result = chain.invoke({"question": "What is LangChain?"})
```

### Minimal RAG Chain

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough

docs = WebBaseLoader("https://docs.example.com").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000).split_documents(docs)
retriever = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings()).as_retriever()

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([
        ("system", "Answer using only this context:\n{context}"),
        ("human", "{question}"),
    ])
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)
```

### Minimal Agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = create_react_agent(ChatOpenAI(model="gpt-4o"), [search])
result = agent.invoke({"messages": [("human", "Search for LangChain tutorials")]})
```

### Agent with Memory

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(model, tools, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user-123"}}

agent.invoke({"messages": [("human", "My name is Alice")]}, config=config)
agent.invoke({"messages": [("human", "What's my name?")]}, config=config)
# "Your name is Alice."
```

---

## Ecosystem Decision Guide

| Need | Solution |
|---|---|
| Simple single-turn LLM call | `model.invoke()` directly |
| Multi-step chain | LCEL with `\|` operator |
| Chat with memory | `RunnableWithMessageHistory` |
| RAG over documents | LCEL chain + retriever |
| Multi-tool agent | `create_react_agent` (LangGraph) |
| Complex branching logic | Custom `StateGraph` |
| Persistent conversations | LangGraph + `AsyncPostgresSaver` |
| Human approval steps | LangGraph `interrupt_before` |
| Token streaming in API | `astream_events` + SSE |
| Production observability | LangSmith tracing |
| Evaluation & testing | LangSmith datasets + evaluators |

---

## Key Environment Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=my-project

# Optional
LANGCHAIN_VERBOSE=false         # Verbose logging
OPENAI_TIMEOUT=60               # Request timeout
```

---

## Package Installation

```bash
# Core
pip install langchain langchain-core

# Model providers
pip install langchain-openai langchain-anthropic langchain-google-genai

# Vector stores (choose one or more)
pip install langchain-community chromadb faiss-cpu langchain-pinecone

# Agents and graphs
pip install langgraph

# Observability
pip install langsmith

# Document loaders
pip install pypdf unstructured bs4

# Text splitters
pip install langchain-text-splitters tiktoken

# Local embeddings
pip install sentence-transformers
```
