# ragjudge

Evaluate Retrieval-Augmented Generation (RAG) systems with a **comprehensive, practical, and multilingual** metric suite — from retrieval quality to generation overlap, citation correctness, faithfulness (NLI), and latency/cost.

* **Batteries included**: classic IR metrics, text-overlap (EM/F1/ROUGE/BLEU), similarity-based scores, citation/evidence checks, NLI-based faithfulness, and performance/cost.
* **Multilingual-friendly**: Unicode-aware normalization & CJK-aware tokenization.
* **Pluggable**: bring your own embeddings, tokenizers, and pricing.
* **Lightweight core** with optional NLP extras (`[nlp]`).

---

## Installation

```bash
# Core metrics
pip install ragjudge

# With optional NLP extras (BERTScore, NLI via transformers/torch)
pip install "ragjudge[nlp]"
```

> Python ≥ 3.11

---

## Metric Overview

| Category                   | Metrics                                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| 🔍 Retrieval               | `Recall@k`, `nDCG@k`, `MRR`, `Diversity@k`, `Redundancy@k`                                         |
| ✍️ Generation Quality      | `ExactMatch (EM)`, `F1`, `ROUGE-N`, `ROUGE-L`, `BLEU`, `BERTScore`, `AnswerRelevancy (similarity)` |
| 📚 Faithfulness & Evidence | `Faithfulness (NLI)`, `Hallucination Rate`, `ContextPrecision@k`, `ContextRecall@k`                |
| 🔗 Citations               | `CitationAccuracy`, `EvidenceDensity`                                                              |
| ⚡ Performance & Cost       | `TTFT`, `E2E Latency`, `Tokens per Request`, `Cost per 1k Requests`                                |

---

## Quick Start

### Python API

```python
from ragjudge.metrics import (
    RecallAtK, RougeL, CitationAccuracy,
    F1Score, BLEU, EvidenceDensity
)

sample = {
    "question": "Who wrote Hamlet?",
    "answer": "Hamlet was written by Shakespeare [1].",
    "reference": ["Hamlet is a tragedy by William Shakespeare."],
    "retrieved_docs": [("d1", 0.95), ("d2", 0.7)],
    "reference_docs": ["d1"],
    "citation_map": {"1": "d1"},
    "doc_text": {"d1": "Shakespeare wrote Hamlet."}
}

print(RecallAtK(k=1).compute(sample))      # {'recall@1': 1.0}
print(RougeL().compute(sample))            # {'rougeL_f1': ..., 'rougeL_precision': ..., 'rougeL_recall': ...}
print(CitationAccuracy(against="gold").compute(sample))
print(F1Score().compute(sample))
print(BLEU().compute(sample))
print(EvidenceDensity(mode="context", k=1).compute(sample))
```

### CLI (batch evaluation)

```bash
# Assume JSONL with one sample per line (see “Data Schema” below)
ragjudge evaluate \
  --input examples/sample.jsonl \
  --metrics recall@3,ndcg@3,rougeL,citation_acc@gold,ttft_ms,e2e_ms \
  --out results.jsonl
```

---

## Data Schema (per-sample keys)

ragjudge is intentionally flexible. You can provide **just what a metric needs**.

Common fields:

* **Retrieval & labels**

  * `retrieved_docs`: `[(doc_id, score), ...]` or `[doc_id, ...]`
  * `relevance`: `{doc_id: graded_score}` (graded labels)
  * `reference_docs`: `[doc_id, ...]` (binary labels)
* **Texts & embeddings**

  * `question`: `str`
  * `answer`: `str`
  * `reference`: `str | List[str]` (gold answers, for EM/F1/ROUGE/BLEU/BERTScore)
  * `reference_texts`: `List[str]` (evidence texts)
  * `doc_text`: `{doc_id: str | List[str]}`
  * `question_embedding`: `List[float]`
  * `answer_embedding`: `List[float]`
  * `reference_embedding`: `List[float] | List[List[float]]`
  * `doc_embedding`: `{doc_id: List[float]}`
* **Citations**

  * `citation_map`: `{"1": "d1", "2": "d2", ...}` (map inline markers to doc IDs)
* **Latency/throughput**

  * `gen_start_time`, `gen_first_token_time`
  * `e2e_start`, `e2e_end` (or `events`: `[{type/name, t}, ...]`)
* **Tokens & cost**

  * `prompt_tokens`, `completion_tokens`, `total_tokens`
  * `model`: `"model-name"` (for pricing registries)

> Many metrics accept fallbacks (e.g., use `reference_docs` when `relevance` is missing). Similarity-based metrics can compute embeddings on-the-fly if you pass an `embedder` callable.

---

## Metric Cheatsheet

**Retrieval**

* `RecallAtK(k)`: fraction of gold docs captured in Top-k.
* `NDCGAtK(k, gain="exp|linear")`: ranked relevance quality with graded labels.
* `MRR(k=None)`: reciprocal rank of the first relevant doc.
* `DiversityAtK(k, mode="cluster|embedding")`: unique clusters (or embedding-based groups) among Top-k.
* `RedundancyAtK`: `1 - DiversityAtK`.

**Generation Overlap**

* `ExactMatch`: strict normalized equality vs reference(s).
* `F1Score`: token-level F1, best-over-multiple references.
* `ROUGE-N (n=1/2/...)`, `ROUGE-L`: standard recall-oriented overlap; returns P/R/F1.
* `BLEU(max_order=4, smoothing="epsilon|add1|none")`: per-sample BLEU with components.
* `BERTScore(model_type=..., agg="max|avg")` *(requires `[nlp]`)*.

**Similarity**

* `AnswerRelevancySim(mode="reference|context", agg="mean|max", k=None)`: cosine over answer vs references or retrieved contexts (uses provided embeddings or a user `embedder`).

**Faithfulness & Evidence**

* `FaithfulnessNLI(mode="reference|context", model_name="roberta-large-mnli")` *(requires `[nlp]`)*: fraction of supported claims; also outputs `hallucination_rate`.
* `ContextPrecisionAtK/ContextRecallAtK`: label-based precision/recall over Top-k contexts.
* `ContextPrecisionSim/ContextRecallSim`: similarity-defined relevance via threshold.

**Citations**

* `CitationAccuracy(against="retrieved|gold", k=None, allow_free_ids=True)`: share of resolved citations inside target set.
* `EvidenceDensity(mode="reference|context", n=1, k=None)`: fraction of answer n-grams covered by evidence vocabulary.

**Performance & Cost**

* `TTFT(unit="ms|s")`: time from generation start to first token.
* `E2ELatency(unit="ms|s")`: end-to-end latency with robust timestamp parsing.
* `TokensPerRequest(...)`: totals from counts or heuristic/tokenizer fallback.
* `CostPer1kRequest(...)`: cost from tokens × price (supports model pricing registries).

---

## Advanced Usage

### Using your own embedder

```python
def my_embedder(x):
    # str -> List[float]; or List[str] -> List[List[float]]
    ...

from ragjudge.metrics import AnswerRelevancySim
m = AnswerRelevancySim(mode="context", k=3, embedder=my_embedder)
m.compute(sample)
```

### Pricing registry for cost

```python
from ragjudge.metrics import CostPer1kRequest
pricing = {"demo-llm": {"prompt": 0.5, "completion": 1.5}}
metric = CostPer1kRequest(pricing=pricing)
metric.compute({"model": "demo-llm", "prompt_tokens": 42, "completion_tokens": 58})
```

### Robust latency from events

```python
from ragjudge.metrics import TTFT, E2ELatency
ttft = TTFT(unit="ms")
e2e  = E2ELatency(unit="ms", decimals=3)
ttft.compute(sample); e2e.compute(sample)
```

---

## JSONL Example (for CLI)

`examples/sample.jsonl`

```json
{"question":"Who wrote Hamlet?","answer":"Hamlet was written by Shakespeare [1].","reference":["Hamlet is a tragedy by William Shakespeare."],"retrieved_docs":[["d1",0.95],["d2",0.7]],"reference_docs":["d1"],"citation_map":{"1":"d1"},"doc_text":{"d1":"Shakespeare wrote Hamlet."}}
{"question":"Who wrote Dream of the Red Chamber?","answer":"It was written by Cao Xueqin [2].","reference":["Dream of the Red Chamber was written by Cao Xueqin."],"retrieved_docs":[["d2",0.88],["d3",0.6]],"reference_docs":["d2"],"citation_map":{"2":"d2"},"doc_text":{"d2":"《红楼梦》作者是曹雪芹。"}}
```

Run:

```bash
ragjudge evaluate \
  --input examples/sample.jsonl \
  --metrics recall@1,rougeL,citation_acc@gold \
  --out results.jsonl
```

---

## Versioning & Stability

* Semantic Versioning starting at **0.1.x**.
* Public API re-exports from `ragjudge.metrics` are the stable surface. Internal modules may change.

---

## Contributing

Issues and PRs are welcome at:
`https://github.com/Lucien1999s/ragjudge`

* Add tests for new metrics.
* Keep outputs **simple dicts** with stable keys.
* Prefer Unicode-safe normalization and multilingual-friendly tokenization.

---

## License

**MIT License** © Lucien Lin

See `LICENSE` for full text.
