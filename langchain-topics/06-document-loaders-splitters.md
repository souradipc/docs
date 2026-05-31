# Document Loaders & Text Splitters

## The Ingestion Pipeline

Before you can do RAG or search over documents, you need to:

1. **Load** raw documents from sources (PDFs, URLs, databases, files)
2. **Split** large documents into chunks that fit within a model's context window
3. **Embed** and store chunks in a vector store

This file covers steps 1 and 2.

```
Source → DocumentLoader → [Document, Document, ...] → TextSplitter → [Chunk, Chunk, ...]
```

---

## The `Document` Object

All loaders return a list of `Document` objects:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the text content of the document.",
    metadata={
        "source": "path/to/file.pdf",
        "page": 1,
        "author": "John Doe",
    }
)

print(doc.page_content)  # "This is the text content..."
print(doc.metadata)      # {"source": "...", "page": 1, ...}
```

Metadata is crucial — it gets stored with embeddings in the vector store, enabling filtered search.

---

## Document Loaders

### Text and Markdown Files

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("path/to/file.txt", encoding="utf-8")
docs = loader.load()  # Returns list[Document]

# Directory of files
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "path/to/docs/",
    glob="**/*.md",          # Only markdown files
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True,
)
docs = loader.load()
```

### PDF Files

```python
# pypdf (no extra dependencies)
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
docs = loader.load()  # One Document per page

# PDFMiner (better text extraction)
from langchain_community.document_loaders import PDFMinerLoader

loader = PDFMinerLoader("report.pdf")
docs = loader.load()
```

### Web Pages

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Single page
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# With BeautifulSoup selector (extract only main content)
loader = WebBaseLoader(
    web_paths=["https://example.com/article"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("article-body", "main-content")
        )
    }
)
docs = loader.load()

# Multiple URLs
loader = WebBaseLoader(["https://url1.com", "https://url2.com"])
docs = loader.load()
```

### CSV Files

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    "data.csv",
    source_column="url",          # Column to use as source in metadata
    content_columns=["title", "body"],  # Columns to use as content
    metadata_columns=["date", "author"],
)
docs = loader.load()
```

### JSON / JSONL Files

```python
from langchain_community.document_loaders import JSONLoader
import jq  # pip install jq

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",              # jq expression to extract records
    content_key="body",           # Which key is the text content
    metadata_func=lambda record, _: {"source": record.get("url")},
)
docs = loader.load()
```

### Databases

```python
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("postgresql://user:pass@localhost/mydb")

loader = SQLDatabaseLoader(
    query="SELECT id, title, content FROM articles WHERE published = true",
    db=db,
    page_content_mapper=lambda row: row["content"],
    metadata_mapper=lambda row: {"id": row["id"], "title": row["title"]},
)
docs = loader.load()
```

### Unstructured Files (Docx, PPT, HTML, etc.)

```python
# pip install unstructured
from langchain_community.document_loaders import UnstructuredFileLoader

loader = UnstructuredFileLoader(
    "document.docx",
    mode="elements",  # "single" | "paged" | "elements"
)
docs = loader.load()
```

---

## Lazy Loading (Memory-Efficient)

For large document sets, use `lazy_load()` to avoid loading everything into memory:

```python
loader = DirectoryLoader("massive-corpus/", glob="**/*.txt")

# Process one document at a time
for doc in loader.lazy_load():
    # process each doc individually
    process(doc)

# Or use async loading for I/O-bound sources
async for doc in loader.alazy_load():
    await async_process(doc)
```

---

## Text Splitters

Large documents must be split into chunks because:
- LLMs have context window limits
- Smaller, focused chunks improve retrieval precision
- Cost is proportional to token count

### `RecursiveCharacterTextSplitter` — Default Choice

Recursively splits on a hierarchy of separators: `["\n\n", "\n", " ", ""]`. Tries to keep semantically related content together.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Max characters per chunk
    chunk_overlap=200,     # Overlap to preserve context across boundaries
    length_function=len,   # Character count (default)
    add_start_index=True,  # Adds 'start_index' to metadata
)

# Split documents (preserves metadata from original docs)
chunks = splitter.split_documents(docs)

# Split plain text
texts = splitter.split_text("Long text here...")
```

### `TokenTextSplitter` — Token-Aware Splitting

Split by actual tokens, not characters. More accurate for LLM context limits:

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=512,    # Tokens per chunk
    chunk_overlap=50,
    encoding_name="cl100k_base",  # GPT-4 tokenizer
)
chunks = splitter.split_documents(docs)
```

### `MarkdownHeaderTextSplitter` — Structure-Aware

Splits Markdown at headers and preserves header hierarchy in metadata:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # Keep headers in chunk text
)

md_text = """# Overview
## Installation
Install with pip.
## Usage
Use like this.
"""

chunks = splitter.split_text(md_text)
# chunk[0].metadata = {"h1": "Overview", "h2": "Installation"}
# chunk[1].metadata = {"h1": "Overview", "h2": "Usage"}
```

### `HTMLHeaderTextSplitter`

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_string)
```

### `PythonCodeTextSplitter` and `Language`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Split code files by function/class boundaries
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=100,
)
chunks = splitter.split_text(python_code)
```

---

## Chunking Strategy — Decision Guide

| Situation | Recommended Splitter | Chunk Size | Overlap |
|---|---|---|---|
| General text (articles, docs) | `RecursiveCharacterTextSplitter` | 1000 chars | 200 chars |
| Precise token budget | `TokenTextSplitter` | 512 tokens | 50 tokens |
| Structured Markdown docs | `MarkdownHeaderTextSplitter` | N/A | N/A |
| Source code | `RecursiveCharacterTextSplitter.from_language()` | 2000 chars | 200 chars |
| Tables/structured data | Custom or direct embedding | Per row | None |
| Long legal/technical docs | `RecursiveCharacterTextSplitter` | 2000 chars | 400 chars |

---

## Chunk Size Tradeoffs

| Smaller Chunks | Larger Chunks |
|---|---|
| More precise retrieval | More context per retrieved chunk |
| Lower token cost per embed | Fewer chunks to manage |
| May miss cross-sentence context | May dilute relevance signal |
| More chunks to store | Fewer chunks to store |

**Rule of thumb**: Start with `chunk_size=1000, chunk_overlap=200`. Tune based on retrieval evaluation.

---

## Preserving and Enriching Metadata

```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load with custom metadata
docs = [
    Document(
        page_content="Long document text...",
        metadata={
            "source": "policy-v3.pdf",
            "department": "Legal",
            "date": "2024-01-15",
            "version": "3.0",
        }
    )
]

# Split — metadata is inherited by all chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.metadata)
    # {'source': 'policy-v3.pdf', 'department': 'Legal', 'date': '2024-01-15', 'version': '3.0', 'start_index': 0}
```

---

## Full Ingestion Pipeline Example

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load
loader = WebBaseLoader("https://docs.example.com/guide")
raw_docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
chunks = splitter.split_documents(raw_docs)

print(f"Loaded {len(raw_docs)} documents → {len(chunks)} chunks")

# 3. Embed and store (covered in 07-embeddings-vector-stores.md)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
```

---

## Common Pitfalls

### 1. No Overlap → Broken Context at Boundaries
```python
# ❌ No overlap — a concept split across two chunks is lost
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# ✅ Overlap of 15-20% ensures continuity
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

### 2. Chunk Size Too Small
```python
# ❌ 100 characters rarely contains enough context for meaningful search
splitter = RecursiveCharacterTextSplitter(chunk_size=100)

# ✅ 500-2000 chars is typical for good retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
```

### 3. Stripping Metadata
```python
# ❌ Raw text loading discards file source information
raw_text = open("file.txt").read()
chunks = splitter.split_text(raw_text)  # No metadata!

# ✅ Use Document with metadata, then split_documents
doc = Document(page_content=raw_text, metadata={"source": "file.txt"})
chunks = splitter.split_documents([doc])  # Metadata preserved
```

---

## Related Topics
- [`07-embeddings-vector-stores.md`](./07-embeddings-vector-stores.md) — Storing chunks
- [`08-rag-pipeline.md`](./08-rag-pipeline.md) — Full retrieval pipeline
