# 19. Agentic RAG

## Overview

Agentic RAG replaces the static retrieve→generate pipeline with an LLM agent that dynamically plans its retrieval strategy. The agent decides *when* to retrieve, *what* to retrieve, *how many* times to retrieve, and *how* to combine results — using tools, iteration, and self-reflection.

---

## Why This Exists

Standard RAG is a one-shot pipeline. Given a question, it retrieves once, then answers. This breaks for:

```
User: "Compare the security posture of Service A and Service B given recent CVEs"

Standard RAG:
  retrieve("compare security Service A Service B") → top-5 chunks
  → often misses B, retrieves wrong CVEs, produces hallucination

Agentic RAG:
  Agent thinks: "I need to search Service A vulnerabilities, then Service B separately"
  Step 1: search("Service A CVE vulnerabilities") → retrieves A's info
  Step 2: search("Service B CVE vulnerabilities") → retrieves B's info
  Step 3: compare_and_summarize(A_results, B_results) → accurate comparison
```

---

## Core Concepts

### Agent Loop (ReAct Pattern)

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

The agent:
1. **Reasons** about what information it needs
2. **Acts** by calling retrieval tools
3. **Observes** retrieved results
4. Decides whether to retrieve more or answer

### Tool Types in Agentic RAG

| Tool | Purpose |
|------|---------|
| `vector_search` | Semantic similarity search |
| `keyword_search` | BM25 / exact match |
| `metadata_filter_search` | Filtered retrieval by date, source, type |
| `graph_traversal` | Entity relationship lookups |
| `web_search` | Real-time knowledge via Tavily/Serper |
| `calculate` | Arithmetic and aggregations |
| `summarize_chunks` | Compress a large set of retrieved chunks |

---

## Basic Agentic RAG (Tool-Calling Agent)

```python
from openai import AsyncOpenAI
from dataclasses import dataclass, field
import json
import asyncio

@dataclass
class AgentState:
    query: str
    history: list[dict] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    final_answer: str | None = None
    steps_taken: int = 0
    max_steps: int = 5

class AgenticRAG:
    """
    Agentic RAG using OpenAI function calling.
    Agent decides which retrieval tools to use and in what order.
    """
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "Search the knowledge base using semantic similarity. Use for conceptual questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "k": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "filtered_search",
                "description": "Search with metadata filters. Use when you need content from specific sources, dates, or categories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {"type": "string", "description": "Filter by document source"},
                        "date_after": {"type": "string", "description": "ISO date string, filter documents after this date"},
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_answer",
                "description": "Provide the final answer to the user's question. Call this when you have enough information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final answer"},
                        "sources": {"type": "array", "items": {"type": "string"}, "description": "Sources used"},
                    },
                    "required": ["answer"]
                }
            }
        }
    ]
    
    SYSTEM_PROMPT = """You are a helpful research assistant with access to a knowledge base.

For each question:
1. Analyze what information you need
2. Use the search tools to retrieve relevant information
3. If the first search is insufficient, search again with different queries
4. Provide a final answer only when you have enough information

Important:
- Search multiple times if needed (up to {max_steps} searches)
- Use different search queries to get diverse information
- Always cite where information came from
- If you cannot find enough information, say so clearly"""
    
    def __init__(self, client: AsyncOpenAI, retriever):
        self.client = client
        self.retriever = retriever
    
    async def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        state.history = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(max_steps=state.max_steps)},
            {"role": "user", "content": query}
        ]
        
        while state.steps_taken < state.max_steps and state.final_answer is None:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=state.history,
                tools=self.TOOLS,
                tool_choice="auto",
            )
            
            msg = response.choices[0].message
            state.history.append(msg.model_dump())
            
            if not msg.tool_calls:
                # Agent decided to answer without tool call
                state.final_answer = msg.content
                break
            
            # Execute tool calls
            for tool_call in msg.tool_calls:
                result = await self._execute_tool(tool_call, state)
                state.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                
                # Check if finalize_answer was called
                if tool_call.function.name == "finalize_answer":
                    args = json.loads(tool_call.function.arguments)
                    state.final_answer = args["answer"]
            
            state.steps_taken += 1
        
        if state.final_answer is None:
            state.final_answer = "Could not determine a confident answer within the step limit."
        
        return state
    
    async def _execute_tool(self, tool_call, state: AgentState) -> str:
        fn_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        if fn_name == "vector_search":
            results = await self.retriever.search(
                query=args["query"],
                k=args.get("k", 5),
            )
            context = self._format_results(results)
            state.retrieved_contexts.append(context)
            return context
        
        elif fn_name == "filtered_search":
            results = await self.retriever.search(
                query=args["query"],
                filters={
                    "source": args.get("source"),
                    "date_after": args.get("date_after"),
                }
            )
            context = self._format_results(results)
            state.retrieved_contexts.append(context)
            return context
        
        elif fn_name == "finalize_answer":
            return "Answer recorded."
        
        return f"Unknown tool: {fn_name}"
    
    def _format_results(self, results: list) -> str:
        if not results:
            return "No results found."
        parts = []
        for i, r in enumerate(results, 1):
            text = getattr(r, "text", str(r))
            parts.append(f"[{i}] {text[:500]}")
        return "\n\n".join(parts)
```

---

## LangGraph-Style Agentic RAG

```python
from typing import Annotated, TypedDict
import operator

# LangGraph state machine for Agentic RAG
class RAGState(TypedDict):
    query: str
    sub_queries: list[str]          # Decomposed sub-questions
    retrieved_docs: Annotated[list[str], operator.add]  # Accumulates across steps
    current_step: str
    answer: str | None
    iterations: int

# Node implementations (each is an async function)
async def decompose_query(state: RAGState) -> dict:
    """Break complex query into sub-queries."""
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Break this question into 2-3 specific sub-questions that can be answered independently.
            
Question: {state['query']}

Return only a JSON array of strings: ["sub-question 1", "sub-question 2", ...]"""
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = json.loads(response.choices[0].message.content)
    # Handle different possible JSON structures
    sub_queries = content if isinstance(content, list) else content.get("questions", [state["query"]])
    return {"sub_queries": sub_queries, "current_step": "retrieval"}

async def retrieve_for_sub_queries(state: RAGState, retriever) -> dict:
    """Retrieve documents for each sub-query in parallel."""
    tasks = [retriever.search(q, k=3) for q in state["sub_queries"]]
    all_results = await asyncio.gather(*tasks)
    
    docs = []
    for sub_q, results in zip(state["sub_queries"], all_results):
        docs.append(f"Sub-query: {sub_q}")
        for r in results:
            docs.append(r.text)
    
    return {"retrieved_docs": docs, "current_step": "synthesis"}

async def synthesize_answer(state: RAGState) -> dict:
    """Generate final answer from all retrieved documents."""
    client = AsyncOpenAI()
    context = "\n\n".join(state["retrieved_docs"][:20])
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer questions using only the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['query']}"}
        ],
        temperature=0,
    )
    return {"answer": response.choices[0].message.content, "current_step": "done"}

def should_continue(state: RAGState) -> str:
    """Routing function: decide what to do next."""
    if state["current_step"] == "decompose":
        return "decompose"
    elif state["current_step"] == "retrieval":
        return "retrieve"
    elif state["current_step"] == "synthesis":
        return "synthesize"
    return "end"
```

---

## Iterative Retrieval Agent

```python
class IterativeRetrievalAgent:
    """
    Agent that iteratively refines retrieval based on what it has found so far.
    Useful for research-style queries where you don't know what you don't know.
    """
    
    REFLECTION_PROMPT = """You are helping answer this question: {query}

Retrieved so far:
{context}

Determine if you have enough information to answer the question.
If yes, set "enough": true and provide "answer".
If no, set "enough": false and specify "missing_info" (what's still needed) 
and "follow_up_query" (the next search query to run).

Return JSON: {{"enough": bool, "answer": "...", "missing_info": "...", "follow_up_query": "..."}}"""
    
    def __init__(self, client: AsyncOpenAI, retriever, max_iterations: int = 3):
        self.client = client
        self.retriever = retriever
        self.max_iterations = max_iterations
    
    async def run(self, query: str) -> dict:
        accumulated_context = []
        current_query = query
        
        for iteration in range(self.max_iterations):
            # Retrieve
            results = await self.retriever.search(current_query, k=5)
            new_context = [r.text for r in results]
            accumulated_context.extend(new_context)
            
            # Reflect: do I have enough?
            reflection = await self._reflect(
                query=query,
                context=accumulated_context
            )
            
            if reflection.get("enough"):
                return {
                    "answer": reflection["answer"],
                    "iterations": iteration + 1,
                    "queries_used": [query] + [current_query] if iteration > 0 else [query],
                }
            
            # Continue with follow-up query
            current_query = reflection.get("follow_up_query", query)
        
        # Max iterations reached - answer with what we have
        return await self._force_answer(query, accumulated_context)
    
    async def _reflect(self, query: str, context: list[str]) -> dict:
        context_str = "\n\n".join(context[:10])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": self.REFLECTION_PROMPT.format(
                    query=query, context=context_str
                )
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"enough": False, "follow_up_query": query}
    
    async def _force_answer(self, query: str, context: list[str]) -> dict:
        context_str = "\n\n".join(context[:15])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer using only provided context. Acknowledge if information is incomplete."},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
            ],
        )
        return {"answer": response.choices[0].message.content, "iterations": self.max_iterations}
```

---

## Production FastAPI Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio, json

app = FastAPI()

@app.post("/api/v1/agentic-rag")
async def agentic_rag_endpoint(request: dict):
    """
    Streaming agentic RAG endpoint.
    Streams intermediate steps as SSE events.
    """
    query = request["query"]
    agent = AgenticRAG(client=AsyncOpenAI(), retriever=get_retriever())
    
    async def event_stream():
        state = AgentState(query=query)
        state.history = [
            {"role": "system", "content": AgenticRAG.SYSTEM_PROMPT.format(max_steps=5)},
            {"role": "user", "content": query}
        ]
        
        while state.steps_taken < state.max_steps and state.final_answer is None:
            # Stream thinking step
            yield f"data: {json.dumps({'type': 'thinking', 'step': state.steps_taken})}\n\n"
            
            response = await agent.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=state.history,
                tools=AgenticRAG.TOOLS,
            )
            
            msg = response.choices[0].message
            state.history.append(msg.model_dump())
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tc.function.name})}\n\n"
                    result = await agent._execute_tool(tc, state)
                    state.history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    
                    if tc.function.name == "finalize_answer":
                        args = json.loads(tc.function.arguments)
                        state.final_answer = args["answer"]
                        break
            else:
                state.final_answer = msg.content
            
            state.steps_taken += 1
            await asyncio.sleep(0)  # Yield to event loop
        
        yield f"data: {json.dumps({'type': 'answer', 'content': state.final_answer, 'steps': state.steps_taken})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## When To Use Agentic RAG

| Scenario | Reason |
|----------|--------|
| Multi-faceted questions | Need to search multiple aspects separately |
| Comparison queries | Retrieve A's info, then B's info independently |
| Unknown unknowns | Agent discovers what to retrieve iteratively |
| Research queries | Long-horizon information gathering |
| Tool augmentation needed | Combine retrieval with calculation, APIs |

---

## When Not To Use

- **Simple factual Q&A** → Single-shot RAG is 10x faster and cheaper
- **Latency-critical applications** → Agent loops add 2-5x latency
- **Deterministic workflows** → If retrieval strategy is known, hardcode it (Advanced RAG)
- **Very large doc sets** → Agent may burn tokens on wrong searches

---

## Common Mistakes

1. **Unbounded loops** — Always set max_iterations
2. **No tool call validation** — Agents can call tools with bad arguments; validate inputs
3. **Accumulating too much context** — Trim accumulated context to stay in token budget
4. **Using a weak model** — Agentic RAG requires a model capable of tool use + planning (GPT-4o-mini minimum)
5. **Missing fallback** — If max iterations reached without answer, return best effort, don't error

---

## Cost and Latency Estimates

```
Standard RAG:
  - 1 embedding call + 1 LLM call
  - Latency: ~1-2 seconds
  - Cost: ~0.001¢

Agentic RAG (3 iterations):
  - 3 embedding calls + 4 LLM calls (3 reasoning + 1 synthesis)
  - Latency: ~5-10 seconds
  - Cost: ~0.01¢ (10x)

Rule of thumb: Use Agentic RAG only when quality justifies the 10x cost/latency premium
```

---

## Related Concepts

- [12. Advanced RAG](12-advanced-rag.md)
- [14. Multi-Query Retrieval](14-multi-query-retrieval.md)
- [16. Corrective RAG](16-corrective-rag.md)
- [20. Multi-Hop Retrieval](20-multi-hop-retrieval.md)

---

## Interview Questions

**Q: What's the difference between Agentic RAG and standard RAG?**  
A: Standard RAG is a fixed pipeline (retrieve once → generate). Agentic RAG lets the LLM dynamically decide how many times to retrieve, what to search for, and which tools to use. It handles complex multi-step queries but costs ~10x more in latency and tokens.

**Q: How do you prevent an agent loop from running indefinitely?**  
A: Enforce a hard max_iterations limit, set a wall-clock timeout, and include a "finalize_answer" tool that the agent must call to stop. Monitor tool call counts in production and alert on outliers.

---

## References

- Yao, S. et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

## Summary

Agentic RAG replaces the fixed retrieve→generate pipeline with an LLM agent that dynamically plans retrieval. Using the ReAct pattern (Reason → Act → Observe), the agent calls retrieval tools iteratively, accumulates context, reflects on sufficiency, and generates an answer only when ready. Best for complex multi-faceted queries, comparisons, and research tasks. Key trade-off: significantly higher quality for hard queries at 5-10x the latency and cost of standard RAG.
