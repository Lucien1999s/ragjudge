# test/test_smoke.py
import math
import os
import sys
import time
from datetime import datetime, timezone

import pytest

# Package under test
from ragjudge.metrics import (
    RecallAtK, NDCGAtK, MRR,
    DiversityAtK, RedundancyAtK,
    ExactMatch, F1Score, RougeN, RougeL, BLEU, BERTScore,
    AnswerRelevancySim,
    FaithfulnessNLI,
    ContextPrecisionAtK, ContextPrecisionSim,
    ContextRecallAtK, ContextRecallSim,
    CitationAccuracy,
    EvidenceDensity,
    TTFT,
    E2ELatency,
    TokensPerRequest,
    CostPer1kRequest,
)

# -----------------------------
# Utilities: deterministic toy embedder
# -----------------------------
def toy_embedder(x, dim=16):
    """
    Deterministic embedding:
      - For a str: returns a vector of length `dim`, where each element is a
        simple function of character ordinals and prime-weighted positions.
      - For list[str] or tuple[str]: returns a list of vectors.
    This yields stable cosine similarity that correlates with content.
    """
    def emb_one(s: str):
        v = [0.0] * dim
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        for idx, ch in enumerate(s):
            v[idx % dim] += (ord(ch) % 97) * primes[idx % len(primes)]
        # avoid zero vectors
        if all(abs(t) < 1e-9 for t in v):
            v[0] = 1.0
        return v

    if isinstance(x, str):
        return emb_one(x)
    if isinstance(x, (list, tuple)):
        return [emb_one(s) for s in x]
    raise TypeError("toy_embedder expects str or list[str].")

# ------------------------------------
# Test corpus: near-realistic multilingual RAG scenario
# ------------------------------------
@pytest.fixture
def sample_corpus():
    """
    A multilingual corpus with retrieval ranking, graded/binary relevance,
    reference answers, citation mapping, timing events, token counts, and
    embeddings—intended to cover most metric families.
    """
    # Retrieved list (contains duplicate d1 to test dedup behavior)
    retrieved = [("d1", 0.92), ("d2", 0.88), ("d3", 0.70), ("d1", 0.51), ("d4", 0.40)]

    # Graded relevance (with binary fallback set)
    relevance = {"d1": 3, "d2": 2, "d5": 1}  # d3/d4 not annotated => non-relevant
    reference_docs = ["d1", "d2", "d5"]      # binary fallback

    # Document texts (mixed languages; some multi-paragraph)
    doc_text = {
        "d1": [
            "Shakespeare wrote Hamlet and Macbeth.",
            "He is a renowned English playwright and poet."
        ],
        "d2": "《红楼梦》是清代作家曹雪芹所著。",
        "d3": "夏目漱石の代表作には『吾輩は猫である』がある。",
        "d4": "This is off-topic material about quantum mechanics.",
        "d5": "Additional Chinese context mentioning 曹雪芹 與 紅樓夢。",
    }

    # Question (English) and answer (includes citations and multilingual keywords)
    question = "Who wrote Hamlet and who wrote Dream of the Red Chamber? Provide sources."
    answer = (
        "Hamlet was written by Shakespeare [1]. "
        "《红楼梦》(Dream of the Red Chamber) was written by 曹雪芹 [d5]. "
        "See also [2]."
    )

    # Citation map: map numeric citations to doc ids; the answer also uses [d5]
    citation_map = {"1": "d1", "2": "d2"}

    # Reference answers (for EM/F1/ROUGE/BLEU/BERTScore)
    references = [
        "Shakespeare wrote Hamlet. Dream of the Red Chamber was written by Cao Xueqin.",
        "Hamlet is by Shakespeare; Dream of the Red Chamber is by Cao Xueqin."
    ]

    # Reference texts (for EvidenceDensity / FaithfulnessNLI in 'reference' mode)
    reference_texts = [
        "Hamlet is a tragedy written by William Shakespeare.",
        "Dream of the Red Chamber is a novel authored by Cao Xueqin."
    ]

    # Timeline events (used by TTFT & E2E)
    now = time.time()
    events = [
        {"type": "request_start", "t": now},
        {"name": "gen_start", "t": now + 0.010},
        {"name": "first_token", "t": now + 0.070},
        {"type": "last_token", "t": now + 0.250},
        {"type": "response_end", "t": now + 0.300},
    ]

    # Direct fields for TTFT (fallback to events if absent)
    gen_start_time = now + 0.020
    gen_first_token_time = now + 0.105  # ~85 ms

    # Tokens and model (for TokensPerRequest / CostPer1kRequest)
    prompt_tokens = 42
    completion_tokens = 58
    total_tokens = prompt_tokens + completion_tokens
    model = "demo-llm"

    # Embeddings (can be omitted; some metrics will compute via embedder)
    question_embedding = toy_embedder(question)
    doc_embedding = {
        k: (toy_embedder(v) if isinstance(v, str) else toy_embedder(" ".join(v)))
        for k, v in doc_text.items()
    }

    # Reference embeddings (used by AnswerRelevancySim in reference mode)
    reference_embedding = [toy_embedder(t) for t in reference_texts]

    s = {
        "id": "multi-1",
        "question": question,
        "answer": answer,
        "reference": references,
        "reference_texts": reference_texts,

        "retrieved_docs": retrieved,
        "relevance": relevance,
        "reference_docs": reference_docs,
        "doc_text": doc_text,
        "doc_embedding": doc_embedding,

        "citation_map": citation_map,

        "events": events,
        "gen_start_time": gen_start_time,
        "gen_first_token_time": gen_first_token_time,

        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "model": model,

        "question_embedding": question_embedding,
        "answer_embedding": toy_embedder(answer),
        "reference_embedding": reference_embedding,
    }
    return s


# -----------------------
# Retrieval family: Recall / nDCG / MRR
# -----------------------
def test_retrieval_family(sample_corpus):
    s = sample_corpus
    m1 = RecallAtK(k=3)
    m2 = NDCGAtK(k=3, gain="exp")
    m3 = MRR(k=3)

    r1 = m1.compute(s)
    r2 = m2.compute(s)
    r3 = m3.compute(s)

    assert "recall@3" in r1 and 0.0 <= r1["recall@3"] <= 1.0
    assert "ndcg@3" in r2 and 0.0 <= (r2["ndcg@3"] or 0.0) <= 1.0
    assert "mrr@3" in r3 and 0.0 <= r3["mrr@3"] <= 1.0

# -----------------------
# Diversity / Redundancy
# -----------------------
def test_diversity_and_redundancy(sample_corpus):
    s = sample_corpus
    # cluster mode (when no cluster info is present, cluster defaults to doc_id)
    d = DiversityAtK(k=4, mode="cluster")
    rd = RedundancyAtK(k=4, mode="cluster")
    out_d = d.compute(s)
    out_r = rd.compute(s)
    assert "diversity@4" in out_d and 0.0 <= out_d["diversity@4"] <= 1.0
    assert "redundancy@4" in out_r and math.isclose(
        out_d["diversity@4"] + out_r["redundancy@4"], 1.0, rel_tol=1e-6
    )

# -----------------------
# Text overlap: EM / F1 / ROUGE / BLEU / BERTScore
# -----------------------
def test_text_overlap_family(sample_corpus):
    s = sample_corpus
    em = ExactMatch()
    f1 = F1Score(include_pr=True)
    r1 = RougeN(n=1, include_pr=True)
    r2 = RougeN(n=2, include_pr=True)
    rl = RougeL(include_pr=True)
    bl = BLEU(max_order=4, smoothing="epsilon", include_components=True)

    o_em = em.compute(s)
    o_f1 = f1.compute(s)
    o_r1 = r1.compute(s)
    o_r2 = r2.compute(s)
    o_rl = rl.compute(s)
    o_bl = bl.compute(s)

    assert "em" in o_em and o_em["em"] in (0.0, 1.0)  # likely 0 for non-exact match
    for k in ("f1", "f1_precision", "f1_recall"):
        assert k in o_f1 and 0.0 <= (o_f1[k] or 0.0) <= 1.0
    for o, pref in [(o_r1, "rouge1_"), (o_r2, "rouge2_")]:
        assert pref + "f1" in o and 0.0 <= (o[pref + "f1"] or 0.0) <= 1.0
        assert pref + "precision" in o and pref + "recall" in o
    for k in ("rougeL_f1", "rougeL_precision", "rougeL_recall"):
        assert k in o_rl and 0.0 <= (o_rl[k] or 0.0) <= 1.0

    assert "bleu" in o_bl and 0.0 <= (o_bl["bleu"] or 0.0) <= 1.0
    for k in ("bleu_bp", "bleu_ref_len", "bleu_hyp_len", "bleu_p1", "bleu_p2", "bleu_p3", "bleu_p4"):
        assert k in o_bl

def test_bertscore_optional(sample_corpus):
    # Skip if dependency is not installed; do not fail the entire test suite.
    pytest.importorskip("bert_score", reason="bert-score not installed")
    s = sample_corpus
    bs = BERTScore(model_type="bert-base-uncased", agg="max", normalize_text=False)
    out = bs.compute(s)
    # Do not check exact values; only key presence and value ranges.
    for k in ("bertscore_f1", "bertscore_precision", "bertscore_recall"):
        assert k in out
        assert out[k] is None or (0.0 <= out[k] <= 1.0)

# -----------------------
# Answer similarity: reference / context modes
# -----------------------
def test_answer_relevancy_sim_both_modes(sample_corpus):
    s = sample_corpus

    # mode=reference: uses provided reference_embedding; no embedder required
    m_ref = AnswerRelevancySim(mode="reference", agg="mean")
    out_ref = m_ref.compute(s)
    assert "answer_rel@reference" in out_ref
    assert out_ref["answer_rel@reference"] is None or -1.0 <= out_ref["answer_rel@reference"] <= 1.0

    # mode=context: compute with toy_embedder
    m_ctx = AnswerRelevancySim(mode="context", k=3, embedder=toy_embedder, agg="mean")
    out_ctx = m_ctx.compute(s)
    assert "answer_rel@context@3" in out_ctx
    assert out_ctx["answer_rel@context@3"] is None or -1.0 <= out_ctx["answer_rel@context@3"] <= 1.0

# -----------------------
# Faithfulness NLI (optional; skip if deps missing)
# -----------------------
def test_faithfulness_nli_optional(sample_corpus):
    # Skip if any dependency is missing; do not fail the suite.
    pytest.importorskip("transformers", reason="transformers not installed")
    pytest.importorskip("torch", reason="torch not installed")
    s = sample_corpus
    # Use reference mode to avoid relying on retrieved context; set thresholds higher to reduce noise.
    fnli = FaithfulnessNLI(
        mode="reference",
        threshold_entail=0.4,
        threshold_contra=0.5,
        pool="max",
        max_length=128,
        batch_size=4,
    )
    out = fnli.compute(s)
    keys = {
        "faithfulness_nli", "hallucination_rate", "faith_num_claims",
        "faith_num_supported", "faith_num_contradicted",
        "faith_mean_entail", "faith_mean_contra"
    }
    assert keys.issubset(out.keys())
    # Range checks only
    if out["faithfulness_nli"] is not None:
        assert 0.0 <= out["faithfulness_nli"] <= 1.0
        assert 0.0 <= out["hallucination_rate"] <= 1.0

# -----------------------
# Context Precision/Recall (label and similarity variants)
# -----------------------
def test_context_precision_and_recall_label(sample_corpus):
    s = sample_corpus
    cp = ContextPrecisionAtK(k=3)
    cr = ContextRecallAtK(k=3)
    o1 = cp.compute(s)
    o2 = cr.compute(s)
    assert "context_precision@3" in o1 and (
        o1["context_precision@3"] is None or 0.0 <= o1["context_precision@3"] <= 1.0
    )
    assert "context_recall@3" in o2 and (
        o2["context_recall@3"] is None or 0.0 <= o2["context_recall@3"] <= 1.0
    )

def test_context_precision_and_recall_sim(sample_corpus):
    s = sample_corpus
    cps = ContextPrecisionSim(k=3, threshold=0.2, embedder=toy_embedder)
    crs = ContextRecallSim(k=3, threshold=0.2, embedder=toy_embedder)
    o1 = cps.compute(s)
    o2 = crs.compute(s)
    assert "context_precision_sim@3" in o1 and (
        o1["context_precision_sim@3"] is None or 0.0 <= o1["context_precision_sim@3"] <= 1.0
    )
    assert "context_recall_sim@3" in o2 and (
        o2["context_recall_sim@3"] is None or 0.0 <= o2["context_recall_sim@3"] <= 1.0
    )

# -----------------------
# Citation Accuracy
# -----------------------
def test_citation_accuracy(sample_corpus):
    s = sample_corpus
    # Against retrieved@k
    ca_r = CitationAccuracy(against="retrieved", k=3, allow_free_ids=True)
    out_r = ca_r.compute(s)
    assert "citation_acc[retrieved]@3" in out_r
    assert "citation_num" in out_r and out_r["citation_num"] >= 1
    assert "citation_resolved" in out_r and out_r["citation_resolved"] >= 1
    # Against gold (reference set)
    ca_g = CitationAccuracy(against="gold", k=3, allow_free_ids=True)
    out_g = ca_g.compute(s)
    assert "citation_acc[gold]@gold" in out_g
    v = out_g["citation_acc[gold]@gold"]
    assert (v is None) or (0.0 <= v <= 1.0)

# -----------------------
# Evidence Density (reference & context; including n>1)
# -----------------------
def test_evidence_density(sample_corpus):
    s = sample_corpus
    # context@n=1
    ed_ctx = EvidenceDensity(mode="context", k=3, n=1)
    o1 = ed_ctx.compute(s)
    key1 = "evidence_density@context@3"
    assert key1 in o1 and f"{key1}__matched" in o1 and f"{key1}__total" in o1
    assert (o1[key1] is None) or (0.0 <= o1[key1] <= 1.0)

    # reference@n=2
    ed_ref = EvidenceDensity(mode="reference", n=2)
    o2 = ed_ref.compute(s)
    key2 = "evidence_density@reference[n2]"
    assert key2 in o2 and f"{key2}__matched" in o2 and f"{key2}__total" in o2
    assert (o2[key2] is None) or (0.0 <= o2[key2] <= 1.0)

# -----------------------
# TTFT & E2E Latency
# -----------------------
def test_ttft_and_e2e(sample_corpus):
    s = sample_corpus

    # TTFT using direct fields (ms)
    ttft_ms = TTFT(unit="ms")
    o1 = ttft_ms.compute(s)
    assert "ttft_ms" in o1
    assert (o1["ttft_ms"] is None) or (o1["ttft_ms"] > 0)

    # TTFT using events only (seconds)
    s2 = dict(s)
    s2.pop("gen_start_time", None)
    s2.pop("gen_first_token_time", None)
    ttft_s = TTFT(unit="s")
    o2 = ttft_s.compute(s2)
    assert "ttft_s" in o2
    assert (o2["ttft_s"] is None) or (o2["ttft_s"] > 0)

    # E2E latency mixing direct fields and events (ms)
    e2e = E2ELatency(unit="ms", decimals=3)
    e = e2e.compute(s)
    assert "e2e_ms" in e
    assert (e["e2e_ms"] is None) or (e["e2e_ms"] >= 0)

    # E2E with ISO datetime strings
    s_iso = dict(s)
    start_dt = datetime.now(timezone.utc)
    end_dt = datetime.fromtimestamp(start_dt.timestamp() + 0.5, tz=timezone.utc)
    s_iso["e2e_start"] = start_dt.isoformat()
    s_iso["e2e_end"] = end_dt.isoformat()
    e_iso = e2e.compute(s_iso)
    assert "e2e_ms" in e_iso and (e_iso["e2e_ms"] is None or e_iso["e2e_ms"] > 0)

# -----------------------
# Tokens / Cost
# -----------------------
def test_tokens_and_cost(sample_corpus):
    s = sample_corpus

    # Tokens per Request (direct token counts)
    tpr = TokensPerRequest()
    out_t = tpr.compute(s)
    for k in ("tokens_total", "tokens_prompt", "tokens_completion"):
        assert k in out_t and isinstance(out_t[k], int)

    # Cost via pricing registry
    pricing = {
        "demo-llm": {"prompt": 0.5, "completion": 1.5},  # USD per 1k tokens
        "other-llm": {"total": 1.2}
    }
    cost = CostPer1kRequest(pricing=pricing, decimals=6)
    out_c = cost.compute(s)
    assert "cost_per_request_usd" in out_c and "cost_per_1k_requests_usd" in out_c
    assert out_c["cost_per_request_usd"] is None or out_c["cost_per_request_usd"] >= 0.0

    # Cost with fixed total price, ignoring model
    s2 = dict(s)
    s2["model"] = "other-llm"
    cost2 = CostPer1kRequest(total_price_per_1k=1.2, decimals=6)
    out_c2 = cost2.compute(s2)
    assert out_c2["cost_per_request_usd"] is None or out_c2["cost_per_request_usd"] >= 0.0

# -----------------------
# Edge / degenerate cases
# -----------------------
def test_degenerate_cases():
    # Empty sample
    empty = {}

    assert RecallAtK(k=3).compute(empty)["recall@3"] is None
    assert NDCGAtK(k=3).compute(empty)["ndcg@3"] is None
    assert MRR(k=3).compute(empty)["mrr@3"] is None

    # No retrieval / no text -> EvidenceDensity None + counters still present
    ed = EvidenceDensity(mode="context", k=2)
    o = ed.compute({"answer": "short"})
    key = "evidence_density@context@2"
    assert key in o and o[key] is None

    # TTFT: non-positive durations -> None
    t_bad = TTFT(unit="ms")
    assert t_bad.compute({"gen_start_time": 10.0, "gen_first_token_time": 9.0})["ttft_ms"] is None

    # E2E: end <= start -> None
    e2e = E2ELatency(unit="ms")
    assert e2e.compute({"e2e_start": 100.0, "e2e_end": 99.0})["e2e_ms"] is None

    # Tokens: missing text/token -> None combinations (only total=None guaranteed)
    tpr = TokensPerRequest()
    out = tpr.compute({})
    assert "tokens_total" in out and out["tokens_total"] is None

    # Cost: no pricing -> None
    c = CostPer1kRequest()
    co = c.compute({})
    assert co["cost_per_request_usd"] is None and co["cost_per_1k_requests_usd"] is None
