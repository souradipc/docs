# Output Parsers & Structured Output

## Why Output Parsers?

LLMs return plain text. Output parsers transform that text into structured Python objects:

- Extract JSON from model responses
- Cast to Pydantic models for type safety
- Retry failed parses automatically
- Enable downstream code to work with typed data instead of strings

---

## Parser Types — Quick Reference

| Parser | Output Type | Method |
|---|---|---|
| `StrOutputParser` | `str` | Text extraction |
| `JsonOutputParser` | `dict` | JSON mode parsing |
| `PydanticOutputParser` | Pydantic model | Instructions + parsing |
| `with_structured_output()` | Pydantic / dict | Native tool calling |
| `XMLOutputParser` | `dict` | XML parsing |
| `CommaSeparatedListOutputParser` | `list[str]` | Simple list splitting |

---

## `StrOutputParser` — Extract Plain Text

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | model
    | StrOutputParser()  # Extracts .content from AIMessage
)

result = chain.invoke({"question": "What is 2 + 2?"})
print(result)        # "4"
print(type(result))  # <class 'str'>
```

Without `StrOutputParser`, the chain returns an `AIMessage` object. With it, you get a plain string.

---

## `JsonOutputParser` — Parse JSON Responses

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()

template = ChatPromptTemplate.from_messages([
    ("system", "Always respond with valid JSON."),
    ("human", "{question}\n\n{format_instructions}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = template | model | parser

result = chain.invoke({"question": "List 3 programming languages with their year created."})
# result is now a Python dict or list

# Streaming JSON (incrementally builds object)
for partial in chain.stream({"question": "List 3 languages"}):
    print(partial)  # Progressively-built dict
```

---

## `PydanticOutputParser` — Type-Safe Parsing

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class CodeAnalysis(BaseModel):
    language: str = Field(description="Programming language detected")
    bugs: List[str] = Field(description="List of bugs found")
    severity: str = Field(description="Overall severity: low/medium/high/critical")
    refactoring_suggestions: List[str] = Field(description="Suggestions for improvement")

parser = PydanticOutputParser(pydantic_object=CodeAnalysis)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a code analyzer. {format_instructions}"),
    ("human", "Analyze this code:\n\n{code}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = template | model | parser

result = chain.invoke({"code": "def divide(a, b): return a / b"})
print(result.language)    # "Python"
print(result.bugs)        # ["Division by zero not handled"]
print(result.severity)    # "medium"
print(type(result))       # <class 'CodeAnalysis'>
```

---

## `with_structured_output()` — Recommended Approach

This is the **modern, preferred** method. Instead of instructing the model about format in the prompt, it uses the model's native tool-calling capability:

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class SecurityAlert(BaseModel):
    alert_id: str = Field(description="Unique alert identifier")
    severity: str = Field(description="Severity level: low/medium/high/critical")
    attack_type: str = Field(description="Type of attack")
    affected_systems: List[str] = Field(description="List of affected system names")
    recommended_actions: List[str] = Field(description="Remediation steps")

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(SecurityAlert)

alert = structured_model.invoke(
    "Analyze this alert: SSH brute force attack detected on web-server-1 and db-server-2."
)

print(alert.severity)          # "high"
print(alert.affected_systems)  # ["web-server-1", "db-server-2"]
print(type(alert))             # <class 'SecurityAlert'>
```

### Why `with_structured_output()` is Better

| Feature | `PydanticOutputParser` | `with_structured_output()` |
|---|---|---|
| Format instructions in prompt | ✅ Yes (wastes tokens) | ❌ No |
| Uses tool calling API | ❌ No | ✅ Yes |
| Parse reliability | Lower | Higher |
| Works with streaming | ❌ No | ✅ Partial objects |
| Requires prompt modification | ✅ Yes | ❌ No |

### Options for `with_structured_output()`

```python
# Method 1: Pydantic (recommended)
structured = model.with_structured_output(MyModel)

# Method 2: JSON Schema dict
structured = model.with_structured_output({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number"},
    },
    "required": ["name", "score"],
})

# Method 3: Include raw response for debugging
structured = model.with_structured_output(MyModel, include_raw=True)
result = structured.invoke("...")
print(result["parsed"])         # MyModel instance
print(result["raw"])            # AIMessage with tool_calls
print(result["parsing_error"])  # None if success
```

---

## Streaming Structured Output

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class Analysis(BaseModel):
    topic: str
    summary: str
    score: int

model = ChatOpenAI(model="gpt-4o")
structured_stream = model.with_structured_output(Analysis)

# Stream partial objects as they build up
for partial in structured_stream.stream("Analyze cybersecurity trends"):
    print(partial)  # Partial Analysis objects
    # {'topic': 'cyber'} → {'topic': 'cybersecurity'} → {'topic': 'cybersecurity', 'summary': ''} ...
```

---

## `OutputFixingParser` — Auto-Retry on Parse Failure

```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

base_parser = PydanticOutputParser(pydantic_object=SecurityAlert)

# Wraps the base parser; if parsing fails, asks model to fix its output
fixing_parser = OutputFixingParser.from_llm(
    llm=ChatOpenAI(model="gpt-4o"),
    parser=base_parser,
)

# Even if the model returns malformed JSON, the fixing parser will retry
result = fixing_parser.parse(malformed_model_output)
```

---

## `CommaSeparatedListOutputParser`

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse("apple, banana, cherry")
print(result)  # ["apple", "banana", "cherry"]

template = ChatPromptTemplate.from_messages([
    ("human", "List 5 {category}.\n{format_instructions}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = template | model | parser
items = chain.invoke({"category": "programming languages"})
# ["Python", "JavaScript", "Go", "Rust", "TypeScript"]
```

---

## Custom Output Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class BulletListParser(BaseOutputParser[List[str]]):
    """Parse bullet point lists from model output."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        items = []
        for line in lines:
            line = line.strip()
            if line.startswith(("- ", "• ", "* ", "· ")):
                items.append(line[2:].strip())
            elif line and line[0].isdigit() and ". " in line:
                items.append(line.split(". ", 1)[1].strip())
        return items

    @property
    def _type(self) -> str:
        return "bullet_list_parser"

parser = BulletListParser()
result = parser.parse("- First item\n- Second item\n- Third item")
# ["First item", "Second item", "Third item"]
```

---

## Parsing Nested and Complex Structures

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CVE(BaseModel):
    cve_id: str
    score: float = Field(ge=0, le=10)

class ThreatReport(BaseModel):
    title: str
    summary: str
    severity: Severity
    affected_platforms: List[str]
    cves: List[CVE]
    mitigations: List[str]
    false_positive: bool = False
    additional_context: Optional[str] = None

model = ChatOpenAI(model="gpt-4o")
structured = model.with_structured_output(ThreatReport)

report = structured.invoke(
    "Generate a threat report for a critical SQL injection vulnerability affecting Linux web servers."
)
print(report.severity)     # Severity.CRITICAL
print(report.cves[0].cve_id)  # "CVE-2024-..."
```

---

## Common Pitfalls

### 1. Using `PydanticOutputParser` When `with_structured_output` is Available
```python
# ❌ Old way — wastes tokens, less reliable
parser = PydanticOutputParser(pydantic_object=MyModel)
# + format_instructions added to prompt

# ✅ Modern way
structured_model = model.with_structured_output(MyModel)
```

### 2. Accessing `.content` Instead of Using Parser
```python
# ❌ Manual string parsing — fragile
response = model.invoke(messages)
data = json.loads(response.content)  # Fails if model adds markdown

# ✅ Use JsonOutputParser or with_structured_output
chain = template | model | JsonOutputParser()
```

### 3. Forgetting `include_raw=True` When Debugging
```python
# When debugging parse failures, get the raw response too
result = model.with_structured_output(
    MyModel,
    include_raw=True
).invoke(input)

if result["parsing_error"]:
    print("Raw:", result["raw"].content)  # See what the model actually returned
```

---

## Related Topics
- [`03-chat-models-llms.md`](./03-chat-models-llms.md) — Model invocation basics
- [`04-prompts-templates.md`](./04-prompts-templates.md) — Formatting model inputs
- [`08-rag-pipeline.md`](./08-rag-pipeline.md) — Structured output in RAG pipelines
