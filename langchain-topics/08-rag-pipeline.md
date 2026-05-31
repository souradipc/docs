# RAG Pipeline — Retrieval-Augmented Generation

## What Is RAG?

RAG addresses a fundamental limitation of LLMs: their knowledge is frozen at training time and they can't access your private data.

**Without RAG:** Model answers from training data only → stale, generic, no private knowledge
**With RAG:** Model retrieves relevant documents at query time → accurate, current, domain-specific

The pipeline has two phases:

```
INDEXING (offline, one-time):
  Source → Load → Split → Embed → Store in VectorDB

RETRIEVAL + GENERATION (online, per query):
  Query → Retrieve relevant chunks → Augment prompt with chunks → Generate answer
```

---

## Minimal RAG Pipeline

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load
loader = WebBaseLoader("https://docs.example.com")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{context}"""),
    ("human", "{question}"),
])

# 5. Chain
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Query
answer = rag_chain.invoke("What are the key features?")
print(answer)
```

---

## RAG with Source Attribution

```python
from langchain_core.runnables import RunnableParallel

# Return both answer and the source documents
rag_chain_with_sources = RunnableParallel({
    "answer": rag_chain,
    "sources": (lambda x: x["question"]) | retriever,
})

result = rag_chain_with_sources.invoke({"question": "What is phishing?"})
print(result["answer"])
for doc in result["sources"]:
    print(f"- {doc.metadata.get('source', 'unknown')}")
```

---

## Conversational RAG (with Chat History)

RAG that supports multi-turn conversations by rephrasing questions with history:

```python
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o")

# 1. Contextualize question — rephrase standalone question given history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Given a chat history and the latest user question,
formulate a standalone question that can be understood without the chat history.
Do NOT answer the question. Only reformulate if needed, otherwise return as-is."""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# 2. Answer with context
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the provided context.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 3. Use with history
chat_history = []

def chat(question: str):
    result = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history,
    })
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=result["answer"]),
    ])
    return result["answer"]

print(chat("What is phishing?"))
print(chat("What are its variants?"))  # Rephrases to "What are the variants of phishing?"
```

---

## Advanced Retrieval Strategies

### 1. MultiQuery Retrieval

```python
from langchain.retrievers import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)
# Generates 3-5 phrasings of the query, deduplicates results
docs = multi_retriever.invoke("How to prevent attacks?")
```

### 2. Contextual Compression

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter

# LLM-based extraction — keeps only relevant sentences
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Cheaper alternative — embedding-based filtering
from langchain.retrievers.document_compressors import EmbeddingsFilter
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever,
)
```

### 3. Parent Document Retriever

Store small chunks for precise retrieval, but return larger parent chunks to the LLM:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Parent splitter — larger chunks returned to LLM
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Child splitter — smaller chunks used for search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryByteStore()
vectorstore = Chroma(embedding_function=embeddings)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

# Retrieves small chunks for matching, returns their parent docs
results = retriever.invoke("What is SQL injection?")
```

### 4. Self-Query Retriever

Let the LLM extract metadata filters from natural language:

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="year", description="Year published", type="integer"),
    AttributeInfo(name="department", description="Department", type="string"),
    AttributeInfo(name="severity", description="Severity level", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Security alerts and vulnerability reports",
    metadata_field_info=metadata_field_info,
)

# Model extracts filter: {year: 2024, severity: "critical"}
docs = retriever.invoke("Show me critical vulnerabilities from 2024")
```

### 5. HyDE — Hypothetical Document Embeddings

Generate a hypothetical answer, embed that, search by it — often finds better matches:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

hyde_template = ChatPromptTemplate.from_messages([
    ("system", "Write a short, factual paragraph that would answer the following question:"),
    ("human", "{question}"),
])

# Generate hypothetical document → embed → search
hyde_chain = (
    hyde_template
    | llm
    | StrOutputParser()
    | (lambda hyp_doc: vectorstore.similarity_search(hyp_doc, k=4))
)

docs = hyde_chain.invoke({"question": "What are the best practices for password security?"})
```

---

## Document Format in Context

How you format retrieved documents affects answer quality:

```python
# Simple join
def format_docs_simple(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# With source attribution
def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i+1}] Source: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# With metadata for grounding
def format_docs_full(docs):
    formatted = []
    for doc in docs:
        meta = doc.metadata
        header = f"Source: {meta.get('source')} | Page: {meta.get('page', 'N/A')} | Date: {meta.get('date', 'N/A')}"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)
```

---

## RAG Prompt Engineering

```python
# High-quality RAG prompt template
RAG_SYSTEM_PROMPT = """You are an expert assistant. Answer the user's question based ONLY on the provided context.

Rules:
1. Only use information from the provided context
2. If the answer is not in the context, say: "I don't have enough information to answer that."
3. Cite the source document when possible
4. Be concise and accurate
5. Never make up facts or statistics

Context:
{context}
"""
```

---

## Reranking Retrieved Documents

After initial retrieval, rerank by relevance using a cross-encoder:

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever

# Reranker model (local)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=3)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
# Retrieves 10, reranks to top 3
docs = reranking_retriever.invoke("phishing prevention")
```

---

## RAG Evaluation

```python
from langsmith import evaluate
from langsmith.evaluation import LangChainStringEvaluator

# Define evaluator
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": llm})

# Run evaluation
results = evaluate(
    rag_chain.invoke,
    data="my-rag-dataset",   # LangSmith dataset name
    evaluators=[qa_evaluator],
    experiment_prefix="rag-v1",
)
```

---

## Common Pitfalls

### 1. Not Filtering Retrieved Chunks by Relevance Score
```python
# ❌ All retrieved chunks regardless of relevance
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ✅ Use score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5},
)
```

### 2. Context Window Overflow
```python
# ❌ Retrieving too many large chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # 20 × 1000 chars = 20K tokens

# ✅ Budget your context window
# gpt-4o = 128K tokens, but cost grows with tokens
# Leave ~4K for output; use ~8K for context = ~8 × 1000-char chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
```

### 3. Not Handling Empty Retrieval
```python
# ❌ Chain fails if no docs retrieved
chain = retriever | format_docs | prompt | llm | StrOutputParser()

# ✅ Handle empty results gracefully
def format_docs_safe(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)
```

---

## Related Topics
- [`06-document-loaders-splitters.md`](./06-document-loaders-splitters.md) — Building document chunks
- [`07-embeddings-vector-stores.md`](./07-embeddings-vector-stores.md) — Vector store setup
- [`09-agents-tools.md`](./09-agents-tools.md) — Agents that can do RAG + more
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — Tracing RAG chains
