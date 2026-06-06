# 20. Multi-Hop Retrieval

## Overview

Multi-hop retrieval handles questions that require chaining multiple retrieval steps to reach an answer. Each hop retrieves new information that informs the next query — like following a chain of evidence. This is distinct from multi-query retrieval (parallel searches) — multi-hop is sequential, with each hop depending on the previous.

---

## Why This Exists

Some questions fundamentally require chained reasoning:

```
Question: "What is the CEO of the company that makes the database used by Service X?"

Hop 1: "What database does Service X use?" → PostgreSQL
Hop 2: "What company makes PostgreSQL?" → PostgreSQL Global Development Group (a non-profit)
Hop 3: "Who leads the PostgreSQL Global Development Group?" → Has no single CEO (open-source)

Answer: "PostgreSQL is developed by a non-profit community, not a company with a CEO"

This cannot be answered with a single vector search.
```

---

## Core Concepts

### Single-Hop vs. Multi-Hop

| Aspect | Single-Hop | Multi-Hop |
|--------|-----------|-----------|
| Query type | Direct factual Q&A | Bridge / Comparison / Composition |
| Retrieval | Once | 2-5 sequential steps |
| Each step | Independent | Depends on previous |
| Complexity | Low | High |
| Example | "What is HNSW?" | "Which CVE affects the indexing library used by our search cluster?" |

### Multi-Hop Question Types

1. **Bridge questions** — "A's property, where A is defined by B" → Requires finding A first
2. **Comparison questions** — "Compare X and Y on dimension Z" → Retrieve X, then Y
3. **Aggregation questions** — "Count all instances of X across Y" → Retrieve all Y, then filter
4. **Causal chains** — "How did A lead to B lead to C?" → Hop through causal links

---

## Basic Multi-Hop Retriever

```python
from openai import AsyncOpenAI
from dataclasses import dataclass, field
import json

@dataclass
class HopResult:
    hop_index: int
    query: str
    retrieved_texts: list[str]
    extracted_fact: str | None = None

@dataclass
class MultiHopResult:
    original_query: str
    hops: list[HopResult]
    final_answer: str
    
    def reasoning_chain(self) -> str:
        parts = [f"Original question: {self.original_query}"]
        for hop in self.hops:
            parts.append(f"Hop {hop.hop_index}: {hop.query}")
            if hop.extracted_fact:
                parts.append(f"  Found: {hop.extracted_fact}")
        return "\n".join(parts)


class MultiHopRetriever:
    """
    Iterative multi-hop retrieval:
    1. Decompose question into hops
    2. Execute each hop, using previous results to form next query
    3. Synthesize final answer
    """
    
    DECOMPOSE_PROMPT = """Analyze this complex question and determine if it requires multiple retrieval steps.

Question: {query}

If the question requires multiple steps to answer (e.g., "find X, then use X to find Y"), 
identify the first hop — the first piece of information needed.

Return JSON:
{{
  "is_multi_hop": true/false,
  "first_hop_query": "the first specific question to search for",
  "reasoning": "why this is (or isn't) multi-hop"
}}"""
    
    NEXT_HOP_PROMPT = """You are answering a complex question step by step.

Original question: {original_query}

So far you have found:
{accumulated_facts}

Based on what you've found, what's the NEXT piece of information needed to answer the original question?
Have you already found enough to answer it?

Return JSON:
{{
  "have_enough": true/false,
  "next_query": "next search query (empty if have_enough is true)",
  "extracted_fact": "key fact from the latest search results",
  "reasoning": "what this fact tells us and why we do/don't need more"
}}

Latest search results:
{latest_results}"""
    
    SYNTHESIS_PROMPT = """Answer this question based on the research chain below.

Question: {query}

Research chain:
{chain}

Provide a clear, direct answer based only on the information in the research chain.
If the information is insufficient, say what's missing."""
    
    def __init__(self, client: AsyncOpenAI, retriever, max_hops: int = 4):
        self.client = client
        self.retriever = retriever
        self.max_hops = max_hops
    
    async def run(self, query: str) -> MultiHopResult:
        hops: list[HopResult] = []
        accumulated_facts: list[str] = []
        
        # Step 1: Check if multi-hop is needed and get first query
        first_hop = await self._plan_first_hop(query)
        current_query = first_hop["first_hop_query"]
        
        for hop_idx in range(self.max_hops):
            # Retrieve for current query
            results = await self.retriever.search(current_query, k=3)
            texts = [r.text for r in results]
            
            hop = HopResult(
                hop_index=hop_idx + 1,
                query=current_query,
                retrieved_texts=texts,
            )
            
            # Determine next step
            next_step = await self._plan_next_hop(
                original_query=query,
                accumulated_facts=accumulated_facts,
                latest_results=texts,
            )
            
            hop.extracted_fact = next_step.get("extracted_fact")
            hops.append(hop)
            
            if hop.extracted_fact:
                accumulated_facts.append(f"Hop {hop_idx+1}: {hop.extracted_fact}")
            
            if next_step.get("have_enough") or not next_step.get("next_query"):
                break
            
            current_query = next_step["next_query"]
        
        # Synthesize answer
        answer = await self._synthesize(query, hops)
        
        return MultiHopResult(
            original_query=query,
            hops=hops,
            final_answer=answer,
        )
    
    async def _plan_first_hop(self, query: str) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.DECOMPOSE_PROMPT.format(query=query)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            result = json.loads(response.choices[0].message.content)
            if not result.get("is_multi_hop"):
                result["first_hop_query"] = query  # Use original query directly
            return result
        except json.JSONDecodeError:
            return {"first_hop_query": query}
    
    async def _plan_next_hop(
        self, original_query: str, accumulated_facts: list[str], latest_results: list[str]
    ) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": self.NEXT_HOP_PROMPT.format(
                    original_query=original_query,
                    accumulated_facts="\n".join(accumulated_facts) or "None yet",
                    latest_results="\n\n".join(latest_results[:3]),
                )
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"have_enough": True}
    
    async def _synthesize(self, query: str, hops: list[HopResult]) -> str:
        chain_parts = []
        for hop in hops:
            chain_parts.append(f"Step {hop.hop_index} — Searched: {hop.query}")
            if hop.extracted_fact:
                chain_parts.append(f"Found: {hop.extracted_fact}")
            chain_parts.append(f"Evidence: {hop.retrieved_texts[0][:400] if hop.retrieved_texts else 'No results'}")
        
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": self.SYNTHESIS_PROMPT.format(
                    query=query,
                    chain="\n".join(chain_parts)
                )
            }],
            temperature=0,
        )
        return response.choices[0].message.content
```

---

## Bridge Question Detector

```python
class MultiHopQueryClassifier:
    """
    Classify whether a query requires multi-hop retrieval before running it.
    Use to route queries: simple → standard RAG, complex → multi-hop.
    """
    
    CLASSIFICATION_PROMPT = """Classify this question's retrieval complexity.

Question: {query}

Categories:
- "single_hop": Directly answerable with one search (e.g., "What is X?", "Define Y")
- "multi_query": Parallel searches needed but no chaining (e.g., "Compare X and Y")  
- "multi_hop": Sequential chaining required (e.g., "The X that does Y, what is its Z?")
- "aggregation": Requires collecting and counting/summarizing multiple items

Return JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def classify(self, query: str) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.CLASSIFICATION_PROMPT.format(query=query)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {"category": "single_hop", "confidence": 0.5}


class SmartRAGRouter:
    """Route queries to appropriate retriever based on complexity."""
    
    def __init__(
        self,
        standard_rag,
        multi_query_rag,
        multi_hop_rag,
        classifier: MultiHopQueryClassifier,
    ):
        self.standard_rag = standard_rag
        self.multi_query_rag = multi_query_rag
        self.multi_hop_rag = multi_hop_rag
        self.classifier = classifier
    
    async def run(self, query: str) -> dict:
        classification = await self.classifier.classify(query)
        category = classification.get("category", "single_hop")
        
        if category == "single_hop":
            result = await self.standard_rag.run(query)
            strategy = "standard_rag"
        elif category == "multi_query":
            result = await self.multi_query_rag.run(query)
            strategy = "multi_query"
        else:  # multi_hop or aggregation
            result = await self.multi_hop_rag.run(query)
            strategy = "multi_hop"
        
        return {
            "answer": result.final_answer if hasattr(result, "final_answer") else result,
            "strategy_used": strategy,
            "classification": classification,
        }
```

---

## Subgraph-Based Multi-Hop (with Knowledge Graph)

```python
class GraphMultiHopRetriever:
    """
    Use the knowledge graph for multi-hop retrieval — 
    traverse entity edges instead of re-embedding for each hop.
    Much more efficient than LLM-planned hops for structured domains.
    """
    
    def __init__(self, knowledge_graph, vector_retriever):
        self.kg = knowledge_graph
        self.retriever = vector_retriever
    
    async def retrieve(self, query: str, start_entity: str, hops: int = 2) -> list[str]:
        """
        Start from a known entity, traverse the graph, return text contexts.
        """
        visited = {start_entity}
        current_layer = {start_entity}
        all_contexts = []
        
        # Get initial context for starting entity
        initial = self.kg.get_entity_context(start_entity, depth=1)
        all_contexts.append(self._format_entity_context(initial))
        
        for hop in range(hops):
            next_layer = set()
            for entity in current_layer:
                # Get neighbors from graph
                neighbors = list(self.kg.graph.successors(entity))
                neighbors += list(self.kg.graph.predecessors(entity))
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_layer.add(neighbor)
                        visited.add(neighbor)
                        
                        # Get context for this neighbor
                        ctx = self.kg.get_entity_context(neighbor, depth=1)
                        all_contexts.append(self._format_entity_context(ctx))
            
            current_layer = next_layer
            if not current_layer:
                break
        
        return all_contexts
    
    def _format_entity_context(self, ctx: dict) -> str:
        parts = [f"Entity: {ctx['entity']}"]
        if ctx.get("attributes", {}).get("description"):
            parts.append(f"  {ctx['attributes']['description']}")
        for rel in ctx.get("relationships", [])[:5]:
            parts.append(f"  - [{rel['relation']}] {rel['to']}: {rel['description']}")
        return "\n".join(parts)
```

---

## Evaluation: Multi-Hop Quality

```python
@dataclass
class MultiHopEvalResult:
    query: str
    predicted_answer: str
    gold_answer: str
    supporting_facts_covered: float  # 0-1, fraction of required facts found
    hop_efficiency: float            # optimal_hops / actual_hops (lower is worse)

def evaluate_multi_hop(
    result: MultiHopResult,
    gold_answer: str,
    required_intermediate_facts: list[str],
) -> MultiHopEvalResult:
    """
    Evaluate multi-hop result quality.
    
    Metrics:
    - Answer correctness: Does the final answer match gold?
    - Fact coverage: Were the necessary intermediate facts found?
    - Hop efficiency: Did we use the minimum hops necessary?
    """
    # Check fact coverage (simple substring matching — use NLI in production)
    covered = sum(
        1 for fact in required_intermediate_facts
        if any(fact.lower() in hop.extracted_fact.lower() 
               for hop in result.hops 
               if hop.extracted_fact)
    )
    fact_coverage = covered / len(required_intermediate_facts) if required_intermediate_facts else 1.0
    
    optimal_hops = len(required_intermediate_facts) + 1
    hop_efficiency = optimal_hops / max(len(result.hops), 1)
    
    return MultiHopEvalResult(
        query=result.original_query,
        predicted_answer=result.final_answer,
        gold_answer=gold_answer,
        supporting_facts_covered=fact_coverage,
        hop_efficiency=min(hop_efficiency, 1.0),  # Cap at 1.0
    )
```

---

## When To Use Multi-Hop

- Bridge questions ("The author of X, what else did they write?")
- Dependency chains ("What services depend on the library with CVE-Y?")
- Composition questions ("What team owns the service that handles Z?")
- Medical/legal reasoning chains
- Knowledge graph traversal for entity-rich domains

## When Not To Use

- Simple factual questions → Standard RAG
- When question can be parallelized → Multi-query RAG
- When graph structure is available → Use GraphRAG's graph traversal directly (more efficient)
- Real-time latency requirements → Too slow (3-5 LLM calls minimum)

---

## Common Mistakes

1. **No stopping condition** — Must check "have enough" at each step, not just run fixed N hops
2. **Context explosion** — Each hop adds context; prune to stay within token limits
3. **Bridge entity hallucination** — If hop 1 returns no entity, hop 2 is based on hallucination; always validate
4. **Using multi-hop for simple queries** — Profile query complexity first, route accordingly

---

## Related Concepts

- [14. Multi-Query Retrieval](14-multi-query-retrieval.md)
- [18. Graph RAG](18-graph-rag.md)
- [19. Agentic RAG](19-agentic-rag.md)

---

## Interview Questions

**Q: How is multi-hop retrieval different from multi-query retrieval?**  
A: Multi-query retrieval is *parallel* — generate N query variants, search all simultaneously, merge results. Multi-hop is *sequential* — each hop depends on results from the previous hop. Multi-query handles diversity (different phrasings of the same question). Multi-hop handles reasoning chains (A → B → C where B is needed to find C).

**Q: How do you prevent multi-hop from hallucinating intermediate facts?**  
A: (1) Always retrieve real documents for each hop — never let the LLM "assume" facts without evidence. (2) Extract the specific entity/fact found from the actual retrieved text, not from LLM memory. (3) Set a retrieval score threshold: if hop N returns no confident results, stop and acknowledge uncertainty rather than continuing on a false premise.

---

## References

- Yang, Z. et al. (2018). [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)
- Trivedi, H. et al. (2022). [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509)

---

## Summary

Multi-hop retrieval chains retrieval steps sequentially, where each hop uses findings from the previous hop to form the next query. Unlike multi-query (parallel), multi-hop is sequential and required when the answer depends on finding intermediate bridge entities or facts. The pipeline: classify query complexity → plan first hop → retrieve → extract key fact → plan next hop → repeat until sufficient → synthesize answer. For entity-rich domains, combine with a knowledge graph to traverse relationships directly instead of re-embedding at each hop.
