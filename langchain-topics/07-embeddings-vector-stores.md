# Embeddings & Vector Stores

## What Are Embeddings?

Embeddings are dense numeric vectors that encode the semantic meaning of text. Semantically similar texts produce vectors that are geometrically close (high cosine similarity).

```
"The cat sat on the mat"     → [0.12, -0.45, 0.78, ...]  ← 1536 dimensions
"A feline rested on the rug" → [0.11, -0.44, 0.79, ...]  ← Very similar vector
"Stock market fell today"    → [-0.33, 0.21, -0.61, ...]  ← Very different
```

This enables **semantic search** — finding documents by meaning, not just keywords.

---

## Embedding Models

### OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# Recommended model (cheap + good)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Higher quality, more expensive
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Reduce dimensions (saves storage + cost)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=512,  # Reduce from 3072 to 512 with minimal quality loss
)

# Embed a single query
vector = embeddings.embed_query("What is malware?")
print(len(vector))  # 1536

# Embed multiple documents
vectors = embeddings.embed_documents(["doc1 text", "doc2 text", "doc3 text"])
print(len(vectors))     # 3
print(len(vectors[0]))  # 1536
```

### HuggingFace Embeddings (Free, Local)

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Best open-source model (runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},        # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True},
)

# Lighter model (faster, less accurate)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Embedding Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | Fast | ★★★★ | $0.02/M tokens |
| `text-embedding-3-large` | 3072 | Fast | ★★★★★ | $0.13/M tokens |
| `text-embedding-ada-002` | 1536 | Fast | ★★★ | $0.10/M tokens (legacy) |
| `BAAI/bge-large-en-v1.5` | 1024 | Medium | ★★★★★ | Free (local) |
| `all-mpnet-base-v2` | 768 | Fast | ★★★★ | Free (local) |
| `all-MiniLM-L6-v2` | 384 | Very Fast | ★★★ | Free (local) |

---

## Vector Stores

Vector stores persist embeddings and enable fast similarity search.

### In-Memory (Dev / Testing)

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create from documents
vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# Or create empty and add later
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(chunks)
```

### Chroma (Local Persistent)

```python
from langchain_community.vectorstores import Chroma

# Create from documents (persists to disk)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_docs",
)

# Load existing collection
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_docs",
)
```

### FAISS (Fast, Local, Production-Ready)

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# Merge two FAISS indexes
vectorstore.merge_from(other_vectorstore)
```

### Pinecone (Managed, Production Cloud)

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pc = Pinecone(api_key="...")
if "my-index" not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name="my-index",
        dimension=1536,  # Must match embedding model dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index",
)
```

### pgvector (PostgreSQL Extension)

```python
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql://user:pass@localhost:5432/mydb"

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="documents",
    connection=CONNECTION_STRING,
    use_jsonb=True,
)
```

### Redis as Vector Store

```python
from langchain_community.vectorstores import Redis

vectorstore = Redis.from_documents(
    documents=chunks,
    embedding=embeddings,
    redis_url="redis://localhost:6379",
    index_name="docs_index",
)
```

---

## Vector Store Comparison

| Store | Persistence | Scale | Use Case |
|---|---|---|---|
| `InMemoryVectorStore` | None | Small | Testing, prototypes |
| `Chroma` | Local disk | Medium | Dev, small production |
| `FAISS` | Local disk | Large | Offline, on-prem, fast search |
| `Pinecone` | Cloud | Very Large | Production SaaS |
| `pgvector` | PostgreSQL | Large | Already using Postgres |
| `Qdrant` | Local or Cloud | Very Large | Production, advanced filtering |
| `Weaviate` | Cloud | Very Large | Hybrid search (BM25 + vector) |
| `Redis` | In-memory + persist | Large | Low-latency production |

---

## Similarity Search

```python
# Basic similarity search
results = vectorstore.similarity_search(
    query="SQL injection attack",
    k=4,  # Number of results
)

for doc in results:
    print(doc.page_content[:200])
    print(doc.metadata)

# With scores
results = vectorstore.similarity_search_with_score(
    query="SQL injection attack",
    k=4,
)

for doc, score in results:
    print(f"Score: {score:.4f}")  # Lower = more similar for L2; Higher for cosine
    print(doc.page_content[:100])

# Filter by metadata
results = vectorstore.similarity_search(
    query="security vulnerability",
    k=4,
    filter={"department": "security", "year": 2024},
)
```

---

## Retrievers

Convert a vector store into a retriever (the `Runnable` interface for retrieval):

```python
# Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # Default
    search_kwargs={"k": 4},
)

# Invoke
docs = retriever.invoke("What is phishing?")

# MMR retriever — Maximal Marginal Relevance (diverse results, less redundancy)
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,          # Number of results to return
        "fetch_k": 20,   # Number to fetch before MMR selection
        "lambda_mult": 0.5,  # 0 = max diversity, 1 = max relevance
    },
)

# Similarity with score threshold — filter out weak matches
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7},
)
```

---

## MultiQueryRetriever

Generate multiple query phrasings to retrieve more relevant results:

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# Automatically generates 3 query variations and deduplicates results
docs = multi_query_retriever.invoke("How do I prevent SQL injection?")
```

---

## Contextual Compression Retriever

Filter retrieved docs to only the relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

base_retriever = vectorstore.as_retriever()
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Returns only the relevant portions of retrieved documents
docs = compression_retriever.invoke("What is phishing?")
```

---

## Adding Documents to Existing Stores

```python
from langchain_core.documents import Document

new_docs = [
    Document(
        page_content="New security policy added in 2024.",
        metadata={"source": "policy-2024.pdf", "year": 2024},
    )
]

# Add to existing store
ids = vectorstore.add_documents(new_docs)
print(ids)  # ["uuid1", "uuid2"]

# Delete by id
vectorstore.delete(ids=["uuid1"])
```

---

## Hybrid Search (BM25 + Vector)

Combines keyword search with semantic search for best results:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 (keyword)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Dense vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Ensemble: combines both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # Equal weight to keyword and semantic
)

docs = ensemble_retriever.invoke("SQL injection prevention")
```

---

## Common Pitfalls

### 1. Dimension Mismatch
```python
# ❌ Embedding dims must match vector store index dims
embeddings_3small = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dims
# Pinecone index created with 3072 dims → ERROR

# ✅ Create index with correct dimensions
pc.create_index("my-index", dimension=1536)  # Match embedding model
```

### 2. Re-embedding on Restart (No Persistence)
```python
# ❌ Re-embedding thousands of docs every restart
vectorstore = InMemoryVectorStore.from_documents(all_chunks, embeddings)

# ✅ Persist to disk
vectorstore = Chroma.from_documents(
    all_chunks, embeddings, persist_directory="./chroma_db"
)
# On restart:
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

### 3. Not Using Metadata Filters
```python
# ❌ Searching all documents when you only need recent ones
docs = retriever.invoke("security policy")

# ✅ Filter first for precision + cost savings
docs = vectorstore.similarity_search(
    "security policy",
    filter={"year": 2024, "department": "security"},
)
```

---

## Related Topics
- [`06-document-loaders-splitters.md`](./06-document-loaders-splitters.md) — Creating chunks to embed
- [`08-rag-pipeline.md`](./08-rag-pipeline.md) — Full RAG pipeline
