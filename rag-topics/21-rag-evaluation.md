# 21. RAG Evaluation

## Overview

RAG evaluation measures quality across three axes: **retrieval quality** (did we find the right documents?), **generation quality** (did the LLM use them correctly?), and **end-to-end quality** (did the user get the right answer?). Without evaluation, you're flying blind — you cannot optimize what you cannot measure.

---

## Why Evaluation is Hard

```
"What is the capital of France?" → "Paris" — Easy to evaluate (exact match)

"Summarize our company's leave policy" → A 3-paragraph summary
  - Is it accurate? (Faithfulness)
  - Does it cover everything? (Recall)
  - Is it relevant to what was asked? (Relevance)
  - Does it only use what we retrieved? (Groundedness)

No single metric captures all of this. RAG needs a suite.
```

---

## Core Metrics Framework (RAGAS)

RAGAS (Retrieval Augmented Generation Assessment) defines 4 core metrics:

### 1. Faithfulness
*Is the answer factually grounded in the retrieved context?*

```
Score = (number of claims in answer supported by context) / (total claims in answer)
Score = 1.0 → every claim is in the context (no hallucination)
Score = 0.5 → half the claims are hallucinated
```

### 2. Answer Relevance
*Is the answer actually relevant to the question?*

```
Score = mean_cosine_similarity(generated_questions_from_answer, original_question)

Method: Given the answer, ask "what question does this answer?" → if similar to original, score is high
```

### 3. Context Precision
*Are the retrieved chunks actually useful?*

```
Score = (relevant chunks in top-k) / k
High precision = every retrieved chunk was needed
Low precision = lots of irrelevant chunks retrieved
```

### 4. Context Recall
*Were all the necessary facts present in retrieved chunks?*

```
Score = (ground truth statements attributable to retrieved context) / (total ground truth statements)
Requires a ground truth answer to compute
```

---

## Implementing RAGAS-Style Evaluation

```python
from openai import AsyncOpenAI
from dataclasses import dataclass
import json
import asyncio

@dataclass
class RAGEvalSample:
    question: str
    answer: str           # LLM-generated answer
    contexts: list[str]   # Retrieved chunks
    ground_truth: str | None = None  # Optional for reference-free metrics

@dataclass
class RAGEvalResult:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float | None  # None if no ground_truth
    
    @property
    def overall_score(self) -> float:
        scores = [self.faithfulness, self.answer_relevance, self.context_precision]
        if self.context_recall is not None:
            scores.append(self.context_recall)
        return sum(scores) / len(scores)
    
    def summary(self) -> str:
        parts = [
            f"Faithfulness:       {self.faithfulness:.3f}",
            f"Answer Relevance:   {self.answer_relevance:.3f}",
            f"Context Precision:  {self.context_precision:.3f}",
        ]
        if self.context_recall is not None:
            parts.append(f"Context Recall:     {self.context_recall:.3f}")
        parts.append(f"Overall Score:      {self.overall_score:.3f}")
        return "\n".join(parts)


class RAGEvaluator:
    """LLM-based RAG evaluation suite."""
    
    FAITHFULNESS_PROMPT = """Given a question and answer, extract all factual claims in the answer.
Then determine which claims are supported by the provided context.

Question: {question}
Answer: {answer}
Context: {context}

Return JSON:
{{
  "claims": ["claim 1", "claim 2", ...],
  "supported": [true/false, true/false, ...],
  "score": 0.0-1.0
}}"""
    
    RELEVANCE_PROMPT = """Given the following answer, generate 3 questions that this answer is addressing.

Answer: {answer}

Return JSON: {{"questions": ["q1", "q2", "q3"]}}"""
    
    PRECISION_PROMPT = """Given a question and retrieved context chunks, classify each chunk as:
- "relevant": Directly useful for answering the question
- "somewhat_relevant": Partially useful
- "irrelevant": Not useful for this question

Question: {question}

Chunks:
{chunks}

Return JSON: {{"classifications": ["relevant"/"somewhat_relevant"/"irrelevant", ...]}}"""
    
    RECALL_PROMPT = """Given a ground truth answer and retrieved context chunks, determine what fraction 
of the key facts in the ground truth can be found in the retrieved chunks.

Ground truth: {ground_truth}
Retrieved context: {context}

Return JSON: {{"recall_score": 0.0-1.0, "missing_facts": ["fact1", "fact2", ...]}}"""
    
    def __init__(self, client: AsyncOpenAI, embedder=None):
        self.client = client
        self.embedder = embedder
    
    async def evaluate_sample(self, sample: RAGEvalSample) -> RAGEvalResult:
        """Evaluate a single RAG sample across all metrics."""
        context_str = "\n\n".join(sample.contexts)
        
        tasks = {
            "faithfulness": self._evaluate_faithfulness(sample.question, sample.answer, context_str),
            "relevance": self._evaluate_relevance(sample.question, sample.answer),
            "precision": self._evaluate_precision(sample.question, sample.contexts),
        }
        
        if sample.ground_truth:
            tasks["recall"] = self._evaluate_recall(sample.ground_truth, context_str)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        result_dict = dict(zip(tasks.keys(), results))
        
        return RAGEvalResult(
            faithfulness=self._safe_extract(result_dict, "faithfulness", "score"),
            answer_relevance=self._safe_extract(result_dict, "relevance", "score"),
            context_precision=self._safe_extract(result_dict, "precision", "score"),
            context_recall=self._safe_extract(result_dict, "recall", "recall_score") if sample.ground_truth else None,
        )
    
    async def evaluate_dataset(self, samples: list[RAGEvalSample]) -> dict:
        """Evaluate a full dataset and return aggregate statistics."""
        results = await asyncio.gather(*[self.evaluate_sample(s) for s in samples])
        
        faithfulness_scores = [r.faithfulness for r in results]
        relevance_scores = [r.answer_relevance for r in results]
        precision_scores = [r.context_precision for r in results]
        recall_scores = [r.context_recall for r in results if r.context_recall is not None]
        
        return {
            "n_samples": len(results),
            "faithfulness": {
                "mean": sum(faithfulness_scores) / len(faithfulness_scores),
                "min": min(faithfulness_scores),
                "max": max(faithfulness_scores),
            },
            "answer_relevance": {
                "mean": sum(relevance_scores) / len(relevance_scores),
            },
            "context_precision": {
                "mean": sum(precision_scores) / len(precision_scores),
            },
            "context_recall": {
                "mean": sum(recall_scores) / len(recall_scores),
            } if recall_scores else None,
            "overall_score": {
                "mean": sum(r.overall_score for r in results) / len(results),
            },
        }
    
    async def _evaluate_faithfulness(self, question: str, answer: str, context: str) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.FAITHFULNESS_PROMPT.format(
                question=question, answer=answer, context=context[:3000]
            )}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    
    async def _evaluate_relevance(self, question: str, answer: str) -> dict:
        """Answer relevance: generate questions from answer, compare to original."""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.RELEVANCE_PROMPT.format(answer=answer)}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        generated_questions = data.get("questions", [])
        
        if not generated_questions:
            return {"score": 0.5}
        
        # Use embedding similarity if embedder available, otherwise LLM scoring
        if self.embedder:
            all_texts = [question] + generated_questions
            embeddings = self.embedder.encode(all_texts, normalize_embeddings=True)
            import numpy as np
            q_emb = embeddings[0]
            gen_embs = embeddings[1:]
            similarities = gen_embs @ q_emb
            score = float(similarities.mean())
        else:
            # Fallback: LLM-based similarity
            score = await self._llm_similarity(question, generated_questions)
        
        return {"score": max(0.0, min(1.0, score))}
    
    async def _evaluate_precision(self, question: str, chunks: list[str]) -> dict:
        chunks_text = "\n---\n".join([f"Chunk {i+1}: {c[:300]}" for i, c in enumerate(chunks)])
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.PRECISION_PROMPT.format(
                question=question, chunks=chunks_text
            )}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        classifications = data.get("classifications", [])
        if not classifications:
            return {"score": 0.5}
        
        scores = {"relevant": 1.0, "somewhat_relevant": 0.5, "irrelevant": 0.0}
        score = sum(scores.get(c, 0.5) for c in classifications) / len(classifications)
        return {"score": score}
    
    async def _evaluate_recall(self, ground_truth: str, context: str) -> dict:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": self.RECALL_PROMPT.format(
                ground_truth=ground_truth, context=context[:3000]
            )}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)
    
    async def _llm_similarity(self, question: str, generated_questions: list[str]) -> float:
        prompt = f"""Rate how similar these generated questions are to the original question.
Original: {question}
Generated: {generated_questions}
Return JSON: {{"similarity": 0.0-1.0}}"""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("similarity", 0.5)
    
    def _safe_extract(self, result_dict: dict, key: str, field: str) -> float:
        val = result_dict.get(key)
        if isinstance(val, Exception) or val is None:
            return 0.5
        if isinstance(val, dict):
            return float(val.get(field, 0.5))
        return 0.5
```

---

## Evaluation Dataset Construction

```python
class EvalDatasetBuilder:
    """Build evaluation datasets automatically from documents."""
    
    QA_GENERATION_PROMPT = """Generate {n} question-answer pairs from the following text.
Questions should be diverse: factual, inferential, and comparison questions.

Text: {text}

Return JSON:
{{
  "pairs": [
    {{"question": "...", "answer": "...", "question_type": "factual/inferential/comparison"}},
    ...
  ]
}}"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
    
    async def generate_qa_pairs(self, text: str, n: int = 5) -> list[dict]:
        response = await self.client.chat.completions.create(
            model="gpt-4o",  # Use stronger model for eval data generation
            messages=[{"role": "user", "content": self.QA_GENERATION_PROMPT.format(
                text=text[:4000], n=n
            )}],
            response_format={"type": "json_object"},
            temperature=0.7,  # Some diversity for question generation
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("pairs", [])
    
    async def build_from_documents(self, documents: list[str], pairs_per_doc: int = 3) -> list[dict]:
        """Generate QA pairs from a document corpus for evaluation."""
        all_pairs = []
        tasks = [self.generate_qa_pairs(doc, n=pairs_per_doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        for pairs in results:
            all_pairs.extend(pairs)
        return all_pairs
```

---

## Continuous Evaluation Pipeline

```python
import time
from collections import defaultdict

class ContinuousEvalPipeline:
    """
    Production evaluation pipeline that runs continuously,
    tracking metric drift over time.
    """
    
    def __init__(self, evaluator: RAGEvaluator, rag_pipeline, eval_dataset: list[RAGEvalSample]):
        self.evaluator = evaluator
        self.rag = rag_pipeline
        self.eval_dataset = eval_dataset
        self.history: list[dict] = []
    
    async def run_evaluation_pass(self) -> dict:
        """Run one evaluation pass over the entire dataset."""
        samples_with_answers = []
        
        for sample in self.eval_dataset:
            result = await self.rag.run(sample.question)
            samples_with_answers.append(RAGEvalSample(
                question=sample.question,
                answer=result.answer,
                contexts=result.contexts,
                ground_truth=sample.ground_truth,
            ))
        
        metrics = await self.evaluator.evaluate_dataset(samples_with_answers)
        metrics["timestamp"] = time.time()
        self.history.append(metrics)
        return metrics
    
    def detect_regressions(self, threshold: float = 0.05) -> list[str]:
        """Detect if recent evaluation is worse than baseline by more than threshold."""
        if len(self.history) < 2:
            return []
        
        baseline = self.history[0]
        recent = self.history[-1]
        
        regressions = []
        for metric in ["faithfulness", "answer_relevance", "context_precision"]:
            baseline_score = baseline.get(metric, {}).get("mean", 0)
            recent_score = recent.get(metric, {}).get("mean", 0)
            if baseline_score - recent_score > threshold:
                regressions.append(f"{metric}: {baseline_score:.3f} → {recent_score:.3f} (dropped {baseline_score - recent_score:.3f})")
        
        return regressions
```

---

## Key Metric Targets (Production)

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|-----------|------|-----------|
| Faithfulness | < 0.7 | 0.7-0.8 | 0.8-0.9 | > 0.9 |
| Answer Relevance | < 0.6 | 0.6-0.75 | 0.75-0.85 | > 0.85 |
| Context Precision | < 0.5 | 0.5-0.7 | 0.7-0.85 | > 0.85 |
| Context Recall | < 0.6 | 0.6-0.75 | 0.75-0.9 | > 0.9 |

---

## Common Mistakes

1. **Evaluating only on generated answers** — Always check retrieval quality separately (context precision/recall)
2. **Using the same model to generate answers AND evaluate** — Self-grading inflates scores; use a different model or human evaluation for final validation
3. **Static eval sets** — Update your eval dataset regularly to reflect new document types and user query patterns
4. **Ignoring latency in evaluation** — A perfect-quality RAG that takes 10 seconds is a product failure; evaluate latency alongside quality

---

## Related Concepts

- [02. Information Retrieval](02-information-retrieval.md)
- [22. RAG Observability](./22-rag-observability.md)
- [23. Hallucination Reduction](./23-hallucination-reduction.md)

---

## Interview Questions

**Q: What are the four RAGAS metrics and what does each measure?**  
A: (1) **Faithfulness** — Is every claim in the answer supported by the retrieved context? (2) **Answer Relevance** — Is the answer actually on-topic for the question? (3) **Context Precision** — Are the retrieved chunks useful? (4) **Context Recall** — Did we retrieve all the facts needed to answer correctly? Faithfulness catches hallucination; relevance catches off-topic answers; precision catches noisy retrieval; recall catches incomplete retrieval.

**Q: How do you evaluate RAG without a ground truth dataset?**  
A: Reference-free metrics only: Faithfulness (context vs. answer only), Answer Relevance (question vs. answer), Context Precision (question vs. retrieved chunks). These require no ground truth. Generate a synthetic ground truth using an LLM over your documents for a quick baseline dataset.

---

## References

- Es, S. et al. (2023). [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [RAGAS Python Library](https://docs.ragas.io/)
- [LangSmith Evaluation Docs](https://docs.smith.langchain.com/evaluation)

---

## Summary

RAG evaluation requires measuring retrieval quality (context precision, context recall) and generation quality (faithfulness, answer relevance) separately. RAGAS provides a reference-free framework: faithfulness checks hallucination by comparing answer claims to retrieved context; answer relevance checks topicality; context precision/recall measure retrieval quality. Build evaluation pipelines that run continuously against a golden dataset and alert on metric regressions above 5% threshold.
