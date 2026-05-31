# Installation & Setup

## Package Structure

LangChain is split into focused packages — install only what you need:

```bash
# Core (required for everything)
pip install langchain-core          # Primitives: Runnable, Messages, PromptTemplate
pip install langchain               # Higher-level chains, memory utilities

# Model providers (install the ones you use)
pip install langchain-openai        # OpenAI + Azure OpenAI
pip install langchain-anthropic     # Anthropic Claude
pip install langchain-google-genai  # Google Gemini
pip install langchain-ollama        # Ollama (local models)
pip install langchain-aws           # Amazon Bedrock
pip install langchain-groq          # Groq (fast inference)

# Community integrations (large, pick pieces)
pip install langchain-community     # Loaders, vector stores, tools, etc.

# Agent framework
pip install langgraph               # Stateful agent graphs

# Observability
pip install langsmith               # Tracing, evaluation, monitoring

# Text processing
pip install langchain-text-splitters
pip install tiktoken                # OpenAI tokenizer (for token counting)

# Embeddings and vector stores
pip install chromadb                # Chroma (local vector store)
pip install faiss-cpu               # FAISS (fast similarity search)
pip install langchain-pinecone      # Pinecone (cloud vector store)
pip install langchain-postgres      # pgvector (PostgreSQL)
pip install sentence-transformers   # HuggingFace local embeddings

# Document loaders (as needed)
pip install pypdf                   # PDF loading
pip install unstructured            # Word, PPT, HTML, etc.
pip install beautifulsoup4          # Web scraping
```

---

## Minimal Install for Common Use Cases

```bash
# Basic QA chatbot
pip install langchain-openai langchain-core

# RAG application
pip install langchain langchain-openai langchain-community chromadb langchain-text-splitters

# Agent with LangGraph
pip install langchain langchain-openai langgraph

# Full production stack
pip install langchain langchain-core langchain-openai langchain-community \
            langchain-text-splitters langgraph langsmith chromadb tiktoken
```

---

## Environment Variables

Create a `.env` file (never commit to version control):

```bash
# .env
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# LangSmith (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_...
LANGCHAIN_PROJECT=my-project-dev  # Organize by project/env

# Optional tuning
LANGCHAIN_VERBOSE=false    # Set to true for debug logging
```

Load in Python:

```python
from dotenv import load_dotenv
load_dotenv()  # pip install python-dotenv

# Or explicitly
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## Verifying Installation

```python
# Quick smoke test
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke("Say hello in one word.")
print(response.content)  # "Hello!"
print(response.usage_metadata)  # Token usage

# Check version
import langchain, langchain_core, langgraph
print(f"langchain:      {langchain.__version__}")
print(f"langchain-core: {langchain_core.__version__}")
print(f"langgraph:      {langgraph.__version__}")
```

---

## LangSmith Setup

```python
import os

# Enable tracing (all LangChain calls auto-traced)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Verify tracing
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
model.invoke("Hello")  # Will appear in LangSmith dashboard
```

---

## Related Topics
- [`00-overview.md`](./00-overview.md) — Full ecosystem overview
- [`03-chat-models-llms.md`](./03-chat-models-llms.md) — Model setup and configuration
