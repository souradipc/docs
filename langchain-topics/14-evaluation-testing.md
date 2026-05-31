# Evaluation & Testing

## Why Evaluate LLM Applications?

Traditional software has deterministic outputs — easy to unit test. LLM applications:
- Return different text every run (non-deterministic)
- "Correct" is subjective (open-ended QA)
- Fail in subtle ways (hallucinations, partial answers, tone)
- Can regress silently when you change a prompt or model

Evaluation bridges this gap: it lets you measure quality, catch regressions, and compare approaches systematically.

---

## Evaluation Approaches

| Method | When to Use | Scalability |
|---|---|---|
| **Human evaluation** | Highest accuracy, setting ground truth | Low (expensive) |
| **LLM-as-judge** | Automated quality scoring | High |
| **Reference-based** | When exact/close answer is known | Medium |
| **Unit tests** | Deterministic behavior (tool calls, structured output) | High |
| **Online evaluation** | Production monitoring with user feedback | High |

---

## Unit Testing Chains and Tools

```python
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

# Test a tool
from myapp.tools import get_threat_intel

def test_get_threat_intel_returns_string():
    result = get_threat_intel.invoke({"ioc": "1.2.3.4", "ioc_type": "ip"})
    assert isinstance(result, str)
    assert len(result) > 0

def test_get_threat_intel_invalid_ioc():
    with pytest.raises(Exception):
        get_threat_intel.invoke({"ioc": "", "ioc_type": "ip"})

# Test a chain with mocked LLM
def test_rag_chain_with_mock():
    from myapp.chains import rag_chain

    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="The answer is 42.")

    with patch("myapp.chains.model", mock_model):
        result = rag_chain.invoke({"question": "What is the answer?"})
        assert "42" in result

# Test structured output
def test_structured_output():
    from myapp.extractors import extract_entities

    result = extract_entities.invoke("Alice is 30 years old from New York.")
    assert result.name == "Alice"
    assert result.age == 30
    assert result.location == "New York"
```

---

## LangSmith Datasets

Create evaluation datasets in LangSmith for systematic testing:

```python
from langsmith import Client

client = Client()

# Create a dataset
dataset = client.create_dataset(
    "rag-qa-v1",
    description="RAG question answering evaluation dataset",
)

# Add examples (input/output pairs)
examples = [
    {
        "inputs": {"question": "What is phishing?"},
        "outputs": {"answer": "Phishing is a type of social engineering attack..."},
    },
    {
        "inputs": {"question": "How does SQL injection work?"},
        "outputs": {"answer": "SQL injection exploits unvalidated input..."},
    },
]

client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

# OR create from existing runs (capture production data)
client.create_examples_from_runs(
    run_ids=["run-id-1", "run-id-2"],
    dataset_id=dataset.id,
)
```

---

## Running Evaluations

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith import Client

client = Client()

# Define the system under test
def run_rag_chain(inputs: dict) -> dict:
    result = rag_chain.invoke(inputs["question"])
    return {"answer": result}

# Built-in evaluators
qa_evaluator = LangChainStringEvaluator("qa")         # Is answer correct?
cot_qa = LangChainStringEvaluator("cot_qa")           # Chain-of-thought QA
criteria_evaluator = LangChainStringEvaluator(         # Custom criteria
    "criteria",
    config={"criteria": {"conciseness": "Is the answer concise?"}},
)

# Run evaluation
results = evaluate(
    run_rag_chain,
    data="rag-qa-v1",             # Dataset name or ID
    evaluators=[qa_evaluator],
    experiment_prefix="rag-v2",
    max_concurrency=4,
    metadata={"model": "gpt-4o", "chunk_size": 1000},
)

print(f"Mean score: {results.to_pandas()['feedback.correctness'].mean()}")
```

---

## Custom Evaluators

```python
from langsmith.evaluation import EvaluationResult
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# LLM-as-judge for relevance scoring
def evaluate_relevance(run, example) -> EvaluationResult:
    """Judge whether the answer is relevant to the question."""
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict evaluator. Score relevance of an AI answer.
Score 1 if the answer directly addresses the question, 0 if it does not.
Respond with ONLY a JSON: {"score": 0 or 1, "reasoning": "..."}"""),
        ("human", "Question: {question}\nAnswer: {answer}"),
    ])

    judge_model = ChatOpenAI(model="gpt-4o", temperature=0)
    judge_chain = judge_prompt | judge_model | JsonOutputParser()

    result = judge_chain.invoke({
        "question": example.inputs["question"],
        "answer": run.outputs.get("answer", ""),
    })

    return EvaluationResult(
        key="relevance",
        score=result["score"],
        comment=result["reasoning"],
    )

# Run with custom evaluator
results = evaluate(
    run_rag_chain,
    data="rag-qa-v1",
    evaluators=[evaluate_relevance],
    experiment_prefix="rag-v3-custom",
)
```

---

## Pairwise Evaluation — A/B Comparison

Compare two systems side by side:

```python
from langsmith.evaluation import evaluate_comparative

def rag_v1(inputs): return {"answer": rag_chain_v1.invoke(inputs["question"])}
def rag_v2(inputs): return {"answer": rag_chain_v2.invoke(inputs["question"])}

# Compare which is better
results = evaluate_comparative(
    [rag_v1, rag_v2],
    evaluators=["pairwise_qa"],  # Which answer is better?
    data="rag-qa-v1",
)
# Results show preference rates for each system
```

---

## Regression Testing

Run evaluation on every pull request to catch quality regressions:

```python
# tests/test_eval.py
import pytest
from langsmith.evaluation import evaluate

@pytest.mark.eval
def test_rag_chain_quality():
    """Ensure RAG chain maintains >80% correctness."""

    def run_chain(inputs):
        return {"answer": rag_chain.invoke(inputs["question"])}

    results = evaluate(
        run_chain,
        data="rag-qa-v1",
        evaluators=[LangChainStringEvaluator("qa")],
        experiment_prefix="ci-test",
    )

    df = results.to_pandas()
    mean_score = df["feedback.correctness"].mean()

    assert mean_score >= 0.8, f"Quality regression: {mean_score:.2f} < 0.80"
```

---

## Online Evaluation — Production Feedback

Capture real user feedback and attach to LangSmith traces:

```python
from langsmith import Client

client = Client()

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Capture user thumbs up/down on a response."""
    client.create_feedback(
        run_id=request.run_id,    # LangSmith run ID (returned in trace)
        key="user_feedback",
        score=1 if request.thumbs_up else 0,
        comment=request.comment,
    )

# In your chain, capture the run_id from the trace
from langsmith.run_helpers import traceable

@traceable
def run_pipeline(question: str) -> dict:
    answer = rag_chain.invoke(question)
    return {"answer": answer}
```

---

## Testing Tips — Best Practices

```python
# 1. Use deterministic temperature=0 for tests
test_model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Mock the LLM to test chain logic (not model quality)
from unittest.mock import AsyncMock

async def test_chain_logic():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content='{"status": "ok"}')

    chain = prompt | mock_llm | JsonOutputParser()
    result = await chain.ainvoke({"input": "test"})
    assert result["status"] == "ok"

# 3. Test tool schemas explicitly
def test_tool_schema():
    assert "ioc" in get_threat_intel.args
    assert get_threat_intel.args["ioc"]["type"] == "string"

# 4. Test error handling paths
def test_tool_handles_invalid_input():
    result = risky_tool.invoke({"input": "bad_value"})
    # Tool should return error message, not raise
    assert "error" in result.lower() or "invalid" in result.lower()
```

---

## Common Pitfalls

### 1. Testing Only the Happy Path
```python
# ❌ Only tests successful cases
def test_rag():
    result = rag_chain.invoke("What is phishing?")
    assert len(result) > 0  # Any non-empty response "passes"

# ✅ Also test: empty retrieval, adversarial inputs, edge cases
def test_rag_no_context():
    result = rag_chain.invoke("What's the latest stock price?")  # Not in KB
    assert "don't have" in result.lower() or "not available" in result.lower()
```

### 2. Using Live LLM in All Unit Tests
```python
# ❌ Slow and expensive — calls OpenAI API on every test run
def test_my_chain():
    result = chain.invoke({"input": "test"})  # Real API call!

# ✅ Mock the LLM for unit tests; use real LLM only for integration/eval tests
@pytest.fixture
def mock_chain(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="mocked response")
    monkeypatch.setattr("myapp.chains.model", mock_llm)
```

---

## Related Topics
- [`15-langsmith-observability.md`](./15-langsmith-observability.md) — LangSmith tracing and datasets
- [`08-rag-pipeline.md`](./08-rag-pipeline.md) — RAG pipeline to evaluate
- [`09-agents-tools.md`](./09-agents-tools.md) — Testing agent behavior
