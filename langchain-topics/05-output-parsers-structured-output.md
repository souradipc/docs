# Output Parsers & Structured Output

## Why Output Parsers?

LLMs return plain text. Output parsers transform that text into structured Python objects:

- Extract JSON from model responses
- Cast to Pydantic models for type safety
- Retry failed parses automatically
- Enable downstream code to work with typed data instead of strings

---

## Parser Types — Quick Reference

| Parser | Output Type | Method | When to Use |
|---|---|---|---|
| `StrOutputParser` | `str` | Text extraction | Simple text responses, no structure needed |
| `JsonOutputParser` | `dict` | JSON mode parsing | Model returns raw JSON, no strict schema |
| `PydanticOutputParser` | Pydantic model | Instructions + parsing | Older models without tool-calling support |
| `PydanticToolsParser` | `list[Pydantic]` | Tool call parsing | Extract **multiple** structured objects from tool calls |
| `JsonOutputToolsParser` | `list[dict]` | Tool call parsing | Extract tool call args as raw dicts (no schema validation) |
| `JsonOutputKeyToolsParser` | `list[dict]` | Tool call parsing | Filter tool calls by a specific tool name |
| `with_structured_output()` | Pydantic / dict | Native tool calling | **Recommended default** — single structured response |
| `XMLOutputParser` | `dict` | XML parsing | Models that output XML (e.g., Anthropic Claude legacy) |
| `CommaSeparatedListOutputParser` | `list[str]` | Simple list splitting | Quick comma-separated lists, no structure needed |
| `OutputFixingParser` | Any | Auto-retry wrapper | Wrap any parser for auto-repair on malformed output |

### Decision Guide — Which Parser to Use?

```
Do you need structured output?
├── No → StrOutputParser
└── Yes
    ├── Model supports tool calling (GPT-4, Claude 3+, Gemini)?
    │   ├── Single object → with_structured_output()  ✅ recommended
    │   ├── Multiple tool calls / objects → PydanticToolsParser
    │   ├── Raw dicts (no Pydantic) → JsonOutputToolsParser
    │   └── Filter by tool name → JsonOutputKeyToolsParser
    ├── Model does NOT support tool calling (older models)?
    │   └── PydanticOutputParser or JsonOutputParser
    ├── Model outputs XML? → XMLOutputParser
    └── Simple list? → CommaSeparatedListOutputParser
```

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

## `PydanticToolsParser` — Multiple Structured Objects from Tool Calls

Use when you want to extract **multiple typed objects** from a single model response using tool calling. The model may call multiple tools in one response; this parser maps each call to its corresponding Pydantic model.

```python
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

class MalwareIndicator(BaseModel):
    """Represents a single malware indicator of compromise."""
    ioc_type: str = Field(description="Type: hash, ip, domain, url")
    value: str = Field(description="The actual IOC value")
    confidence: str = Field(description="Confidence level: low/medium/high")

class ThreatActor(BaseModel):
    """Represents a threat actor mentioned in the report."""
    name: str = Field(description="Threat actor name or alias")
    motivation: str = Field(description="Primary motivation: espionage/financial/hacktivist")

model = ChatOpenAI(model="gpt-4o")

# Bind both tools so the model can call either (or both) in one response
model_with_tools = model.bind_tools([MalwareIndicator, ThreatActor])
parser = PydanticToolsParser(tools=[MalwareIndicator, ThreatActor])

chain = model_with_tools | parser

results = chain.invoke(
    "Extract all IOCs and threat actors from: APT29 used domain evil.ru and hash abc123 to deploy malware."
)
# results is a list — each item is either a MalwareIndicator or ThreatActor instance
for item in results:
    print(type(item).__name__, item)
```

**When to use:** You need to extract several different structured objects in one pass — e.g., IOCs + threat actors + CVEs from a single threat report.

---

## `JsonOutputToolsParser` — Raw Tool Call Dicts (No Schema)

Same as `PydanticToolsParser` but returns plain `list[dict]` instead of Pydantic instances. Useful when you don't have a Pydantic model or want raw flexibility.

```python
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_alert",
            "description": "Extract alert details",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["severity", "description"],
            },
        },
    }
]

model_with_tools = model.bind_tools(tools)
parser = JsonOutputToolsParser()  # Returns list of {"type": "tool_name", "args": {...}}

chain = model_with_tools | parser

results = chain.invoke("High severity SQL injection detected on login endpoint.")
# [{"type": "extract_alert", "args": {"severity": "high", "description": "SQL injection..."}}]
print(results[0]["args"]["severity"])  # "high"
```

**When to use:** You have dynamic or schema-less tools, or you want raw dicts without Pydantic validation overhead.

---

## `JsonOutputKeyToolsParser` — Filter by Specific Tool Name

A focused variant of `JsonOutputToolsParser` that only returns calls to **one specific tool**, ignoring others. Handy when the model can call multiple tools but you only care about one.

```python
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Model is bound to multiple tools but we only want results from "extract_cve"
model_with_tools = model.bind_tools([extract_cve_tool, extract_ioc_tool])

# Only parse calls to "extract_cve", ignore everything else
parser = JsonOutputKeyToolsParser(key_name="extract_cve")

chain = model_with_tools | parser

results = chain.invoke("CVE-2024-1234 and CVE-2024-5678 were exploited via a phishing email.")
# Only extract_cve tool call args are returned
# [{"cve_id": "CVE-2024-1234", ...}, {"cve_id": "CVE-2024-5678", ...}]
```

**When to use:** The model has access to multiple tools but downstream you only need results from one specific tool — avoids manually filtering the list yourself.

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
