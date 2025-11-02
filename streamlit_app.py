import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from ragjudge.judge import LLMJudge
from ragjudge.metrics import (
    RecallAtK,
    NDCGAtK,
    MRR,
    DiversityAtK,
    RedundancyAtK,
    ExactMatch,
    F1Score,
    RougeN,
    RougeL,
    BLEU,
    AnswerRelevancySim,
    ContextPrecisionAtK,
    ContextPrecisionSim,
    ContextRecallAtK,
    ContextRecallSim,
    CitationAccuracy,
    EvidenceDensity,
    TTFT,
    E2ELatency,
    TokensPerRequest,
    CostPer1kRequest,
)

# optional extras
try:
    from ragjudge.metrics import BERTScore, FaithfulnessNLI
    HAS_NLP_EXTRAS = True
except ImportError:
    HAS_NLP_EXTRAS = False


# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RAGJudge Studio",
    page_icon="📊",
    layout="wide",
)

st.title("RAGJudge Studio")
st.caption("Minimal-to-advanced UI for evaluating RAG answers. Start with Q&A, add retrieval only if you have it.")


# -----------------------------------------------------------------------------
# PRESETS
# -----------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "Simple QA (Hamlet)": {
        "question": "Who wrote Hamlet?",
        "answer": "Hamlet was written by William Shakespeare [1].",
        "reference": [
            "Hamlet is a tragedy by William Shakespeare.",
            "Shakespeare wrote the play Hamlet.",
        ],
        "retrieved_docs": [["doc1", 0.95], ["doc2", 0.7]],
        "reference_docs": ["doc1"],
        "relevance": {"doc1": 3, "doc2": 1},
        "citation_map": {"1": "doc1"},
        "doc_text": {
            "doc1": "Shakespeare wrote Hamlet.",
            "doc2": "Some irrelevant text.",
        },
    },
    "RAG-heavy (Dream of the Red Chamber)": {
        "question": "Who is the author of Dream of the Red Chamber?",
        "answer": "The novel was written by Cao Xueqin [1].",
        "reference": ["《红楼梦》是曹雪芹所著。"],
        "retrieved_docs": [["doc_zh", 0.88], ["doc_jp", 0.6]],
        "reference_docs": ["doc_zh"],
        "relevance": {"doc_zh": 3},
        "citation_map": {"1": "doc_zh"},
        "doc_text": {
            "doc_zh": "《红楼梦》作者是曹雪芹。",
            "doc_jp": "夏目漱石の代表作には『吾輩は猫である』がある。",
        },
    },
}

EMPTY_SAMPLE: Dict[str, Any] = {
    "question": "",
    "answer": "",
    "reference": [],
    "retrieved_docs": [],
    "reference_docs": [],
    "relevance": {},
    "citation_map": {},
    "doc_text": {},
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "model": "",
    "gen_start_time": "",
    "gen_first_token_time": "",
    "e2e_start": "",
    "e2e_end": "",
}


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def toy_embedder(text: Any, dim: int = 24) -> List[float] | List[List[float]]:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def encode_one(s: str) -> List[float]:
        vec = [0.0] * dim
        s = s or ""
        for idx, ch in enumerate(s):
            vec[idx % dim] += (ord(ch) % 97) * primes[idx % len(primes)]
        if not any(abs(v) > 1e-9 for v in vec):
            vec[0] = 1.0
        return vec

    if isinstance(text, str):
        return encode_one(text)
    if isinstance(text, (list, tuple)):
        return [encode_one(t) for t in text]
    raise TypeError("Embedder expects str or sequence of strings.")


def parse_json_input(label: str, raw: str, fallback: Any) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.warning(f"{label} - JSON parse failed: {exc}", icon="⚠️")
        return fallback


def _maybe_float(val: str) -> Any:
    val = (val or "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return val


def autofill_missing(sample: Dict[str, Any]) -> Dict[str, Any]:
    filled = sample.copy()
    if not filled.get("reference"):
        ans = filled.get("answer") or ""
        filled["reference"] = [ans] if ans else []
    filled.setdefault("retrieved_docs", [])
    filled.setdefault("reference_docs", [])
    filled.setdefault("relevance", {})
    filled.setdefault("citation_map", {})
    filled.setdefault("doc_text", {})
    filled["prompt_tokens"] = filled.get("prompt_tokens") or 0
    filled["completion_tokens"] = filled.get("completion_tokens") or 0
    filled["total_tokens"] = filled.get("total_tokens") or (
        filled["prompt_tokens"] + filled["completion_tokens"]
    )
    return filled


def make_dummy_rag_from_qa(sample: Dict[str, Any], k: int = 3) -> Dict[str, Any]:
    """
    指揮官要的一鍵產生：從 QA 自動生出最小 RAG 結構。
    """
    answer = sample.get("answer") or "No answer."
    doc_id = "doc1"
    retrieved = [[doc_id, 0.95]]
    for i in range(2, k + 1):
        retrieved.append([f"doc{i}", 0.7 - 0.05 * i])
    doc_text = {
        doc_id: answer.replace("[1]", "").strip() or "dummy doc about the answer.",
        "doc2": "This is an additional, less relevant document.",
        "doc3": "Totally unrelated text.",
    }
    sample["retrieved_docs"] = retrieved
    sample["reference_docs"] = [doc_id]
    sample["citation_map"] = {"1": doc_id}
    sample["doc_text"] = doc_text
    return sample


def render_basic_bar(metrics: Dict[str, float]) -> None:
    """Render a simple horizontal bar chart for basic mode."""
    import plotly.express as px

    # 把 None 過濾掉
    data = [{"metric": k, "value": v} for k, v in metrics.items() if v is not None]
    if not data:
        st.info("No generation metrics to display.", icon="ℹ️")
        return

    fig = px.bar(
        data,
        x="value",
        y="metric",
        orientation="h",
        range_x=[0, 1],
        text="value",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=80, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
CategoryEntry = Dict[str, Any]


def metric_specs(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    cat = []

    # Retrieval
    cat.extend(
        [
            (
                "🔍 Retrieval",
                {
                    "label": f"Recall@{cfg['retrieval_k']}",
                    "metric": RecallAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"recall@{cfg['retrieval_k']}",
                },
            ),
            (
                "🔍 Retrieval",
                {
                    "label": f"nDCG@{cfg['retrieval_k']}",
                    "metric": NDCGAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"ndcg@{cfg['retrieval_k']}",
                },
            ),
            (
                "🔍 Retrieval",
                {
                    "label": f"MRR@{cfg['retrieval_k']}",
                    "metric": MRR(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"mrr@{cfg['retrieval_k']}",
                },
            ),
            (
                "🔍 Retrieval",
                {
                    "label": f"Diversity@{cfg['retrieval_k']}",
                    "metric": DiversityAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"diversity@{cfg['retrieval_k']}",
                },
            ),
            (
                "🔍 Retrieval",
                {
                    "label": f"Redundancy@{cfg['retrieval_k']}",
                    "metric": RedundancyAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"redundancy@{cfg['retrieval_k']}",
                },
            ),
        ]
    )

    # Generation
    cat.extend(
        [
            (
                "✍️ Generation Quality",
                {
                    "label": "Exact Match",
                    "metric": ExactMatch(),
                    "primary_key": lambda: "em",
                },
            ),
            (
                "✍️ Generation Quality",
                {
                    "label": "F1 Score",
                    "metric": F1Score(),
                    "primary_key": lambda: "f1",
                },
            ),
            (
                "✍️ Generation Quality",
                {
                    "label": f"ROUGE-{cfg['rouge_n']}",
                    "metric": RougeN(n=cfg["rouge_n"]),
                    "primary_key": lambda: f"rouge{cfg['rouge_n']}_f1",
                },
            ),
            (
                "✍️ Generation Quality",
                {
                    "label": "ROUGE-L",
                    "metric": RougeL(),
                    "primary_key": lambda: "rougeL_f1",
                },
            ),
            (
                "✍️ Generation Quality",
                {
                    "label": "BLEU",
                    "metric": BLEU(),
                    "primary_key": lambda: "bleu",
                },
            ),
            (
                "✍️ Generation Quality",
                {
                    "label": "Answer Relevancy (to context)",
                    "metric": AnswerRelevancySim(
                        mode="context",
                        k=cfg["retrieval_k"],
                        embedder=toy_embedder,
                    ),
                    "primary_key": lambda: "answer_relevancy@context",
                },
            ),
        ]
    )

    if HAS_NLP_EXTRAS and cfg["enable_nlp"]:
        cat.append(
            (
                "✍️ Generation Quality",
                {
                    "label": "BERTScore (F1)",
                    "metric": BERTScore(model_type=cfg["bertscore_model"], agg="max"),
                    "primary_key": lambda: "bertscore_f1",
                },
            )
        )

    # Grounding
    cat.extend(
        [
            (
                "📚 Grounding & Faithfulness",
                {
                    "label": f"Context Precision@{cfg['retrieval_k']}",
                    "metric": ContextPrecisionAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"context_precision@{cfg['retrieval_k']}",
                },
            ),
            (
                "📚 Grounding & Faithfulness",
                {
                    "label": f"Context Recall@{cfg['retrieval_k']}",
                    "metric": ContextRecallAtK(k=cfg["retrieval_k"]),
                    "primary_key": lambda: f"context_recall@{cfg['retrieval_k']}",
                },
            ),
            (
                "📚 Grounding & Faithfulness",
                {
                    "label": "Context Precision (Sim)",
                    "metric": ContextPrecisionSim(
                        k=cfg["retrieval_k"],
                        threshold=cfg["sim_threshold"],
                        embedder=toy_embedder,
                    ),
                    "primary_key": lambda: "context_precision_sim",
                },
            ),
            (
                "📚 Grounding & Faithfulness",
                {
                    "label": "Context Recall (Sim)",
                    "metric": ContextRecallSim(
                        k=cfg["retrieval_k"],
                        threshold=cfg["sim_threshold"],
                        embedder=toy_embedder,
                    ),
                    "primary_key": lambda: "context_recall_sim",
                },
            ),
        ]
    )

    if HAS_NLP_EXTRAS and cfg["enable_nlp"]:
        cat.append(
            (
                "📚 Grounding & Faithfulness",
                {
                    "label": "Faithfulness (NLI)",
                    "metric": FaithfulnessNLI(
                        mode="reference", model_name=cfg["faithfulness_model"]
                    ),
                    "primary_key": lambda: "faithfulness_nli",
                },
            )
        )

    # Citations
    cat.extend(
        [
            (
                "🔗 Citations & Evidence",
                {
                    "label": "Citation Accuracy (vs gold)",
                    "metric": CitationAccuracy(
                        against="gold",
                        k=cfg["retrieval_k"],
                        allow_free_ids=True,
                    ),
                    "primary_key": lambda: "citation_acc[gold]@gold",
                },
            ),
            (
                "🔗 Citations & Evidence",
                {
                    "label": f"Citation Accuracy (vs retrieved@{cfg['retrieval_k']})",
                    "metric": CitationAccuracy(
                        against="retrieved",
                        k=cfg["retrieval_k"],
                        allow_free_ids=True,
                    ),
                    "primary_key": lambda: f"citation_acc[retrieved]@{cfg['retrieval_k']}",
                },
            ),
            (
                "🔗 Citations & Evidence",
                {
                    "label": "Evidence Density (context)",
                    "metric": EvidenceDensity(mode="context", k=cfg["retrieval_k"], n=1),
                    "primary_key": lambda: f"evidence_density@context@{cfg['retrieval_k']}",
                },
            ),
        ]
    )

    # Performance
    cat.extend(
        [
            (
                "⚡ Performance & Cost",
                {
                    "label": "TTFT (ms)",
                    "metric": TTFT(unit="ms"),
                    "primary_key": lambda: "ttft_ms",
                },
            ),
            (
                "⚡ Performance & Cost",
                {
                    "label": "E2E Latency (ms)",
                    "metric": E2ELatency(unit="ms"),
                    "primary_key": lambda: "e2e_ms",
                },
            ),
            (
                "⚡ Performance & Cost",
                {
                    "label": "Tokens per Request",
                    "metric": TokensPerRequest(),
                    "primary_key": lambda: "tokens_total",
                },
            ),
            (
                "⚡ Performance & Cost",
                {
                    "label": "Cost per Request (USD)",
                    "metric": CostPer1kRequest(
                        total_price_per_1k=cfg["cost_per_1k"], decimals=4
                    ),
                    "primary_key": lambda: "cost_per_request_usd",
                },
            ),
        ]
    )

    return cat


def compute_metrics(sample: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[CategoryEntry]]]:
    all_outputs: Dict[str, Any] = {}
    categorized: Dict[str, List[CategoryEntry]] = {
        "🔍 Retrieval": [],
        "✍️ Generation Quality": [],
        "📚 Grounding & Faithfulness": [],
        "🔗 Citations & Evidence": [],
        "⚡ Performance & Cost": [],
    }

    for category, spec in metric_specs(cfg):
        metric = spec["metric"]
        primary_key = spec["primary_key"]()
        try:
            result = metric.compute(sample)
        except Exception as exc:
            result = {primary_key: None, "error": str(exc)}

        if isinstance(result, dict):
            all_outputs.update(result)
            primary_value = result.get(primary_key)
        else:
            primary_value = None

        categorized[category].append(
            {
                "label": spec["label"],
                "primary_key": primary_key,
                "primary_value": primary_value,
                "payload": result,
            }
        )

    return all_outputs, categorized


def aggregated_scores_for_radar(categorized: Dict[str, List[CategoryEntry]]) -> Dict[str, float]:
    summary = {}
    for category, entries in categorized.items():
        values = []
        for entry in entries:
            v = entry["primary_value"]
            key = entry["primary_key"]
            if v is None:
                continue
            if key in ("ttft_ms", "e2e_ms", "tokens_total", "cost_per_request_usd"):
                norm = 1.0 / (1.0 + float(v))
                values.append(norm)
            else:
                fv = float(v)
                if fv <= 1.0:
                    values.append(fv)
                else:
                    values.append(1.0)
        summary[category] = sum(values) / len(values) if values else 0.0
    return summary


def render_category_tab(tab, entries: List[CategoryEntry]) -> None:
    with tab:
        cols = st.columns(2)
        for idx, entry in enumerate(entries):
            with cols[idx % 2]:
                st.metric(entry["label"], value=_fmt_metric(entry["primary_value"]))
        df = pd.DataFrame(
            [
                {
                    "Metric": entry["label"],
                    "Key": entry["primary_key"],
                    "Value": entry["primary_value"],
                    "Raw": json.dumps(entry["payload"], ensure_ascii=False),
                }
                for entry in entries
            ]
        )
        st.dataframe(df, use_container_width=True)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_radar(scores: Dict[str, float]) -> None:
    option = {
        "tooltip": {"trigger": "item"},
        "radar": {
            "indicator": [{"name": k, "max": 1.0} for k in scores],
            "radius": "65%",
        },
        "series": [
            {
                "type": "radar",
                "data": [
                    {
                        "value": [round(scores[k], 4) for k in scores],
                        "name": "Normalized",
                    }
                ],
                "areaStyle": {"opacity": 0.2},
            }
        ],
    }
    st_echarts(option, height="360px")


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Mode")
    ui_mode = st.radio(
        "Select UI mode",
        ["Basic (QA only)", "Advanced (RAG + Telemetry)"],
    )

    st.divider()
    st.subheader("Metrics config")
    retrieval_k = st.slider("Top-K for retrieval", 1, 10, 3)
    rouge_n = st.selectbox("ROUGE-N", [1, 2, 3, 4], index=1)
    sim_threshold = st.slider("Similarity threshold", 0.1, 0.99, 0.75)

    st.divider()
    enable_nlp = st.checkbox(
        "Enable NLP advanced metrics",
        value=False,
        disabled=not HAS_NLP_EXTRAS,
    )
    bertscore_model = st.text_input("BERTScore model", value="bert-base-multilingual-cased")
    faithfulness_model = st.text_input("Faithfulness NLI model", value="roberta-large-mnli")
    cost_per_1k = st.number_input("Cost (USD / 1k requests)", min_value=0.0, value=0.0, step=0.1)

    st.divider()
    st.subheader("Quick start")
    preset_name = st.selectbox("Load sample case", list(PRESETS.keys()))
    if st.button("Load demo case"):
        st.session_state["current_sample"] = PRESETS[preset_name].copy()
        st.success("Demo case loaded.")
    if st.button("Generate dummy RAG from current Q&A"):
        base = st.session_state.get("current_sample", EMPTY_SAMPLE.copy())
        st.session_state["current_sample"] = make_dummy_rag_from_qa(base)
        st.success("Dummy RAG fields generated.")


cfg = {
    "retrieval_k": retrieval_k,
    "rouge_n": rouge_n,
    "sim_threshold": sim_threshold,
    "enable_nlp": enable_nlp,
    "bertscore_model": bertscore_model,
    "faithfulness_model": faithfulness_model,
    "cost_per_1k": cost_per_1k,
}


# -----------------------------------------------------------------------------
# MAIN INPUTS (NO FORM)
# -----------------------------------------------------------------------------
if "current_sample" not in st.session_state:
    st.session_state["current_sample"] = EMPTY_SAMPLE.copy()

sample_state = st.session_state["current_sample"]

st.subheader("1. Question & Answer (required)")
st.markdown("Fill **at least** Question and Model Answer. Reference is recommended.")
question = st.text_area(
    "User Question / Prompt",
    value=sample_state.get("question", ""),
    height=110,
)
answer = st.text_area(
    "Model Answer (LLM output)",
    value=sample_state.get("answer", ""),
    height=120,
)
reference_raw = st.text_area(
    "Reference Answers (JSON array or one per line)",
    value=(
        json.dumps(sample_state.get("reference", []), ensure_ascii=False, indent=2)
        if isinstance(sample_state.get("reference"), list)
        else (sample_state.get("reference") or "")
    ),
    height=130,
)

# Advanced sections (NOW VISIBLE)
if ui_mode == "Advanced (RAG + Telemetry)":
    with st.expander("2. Retrieval & Documents (optional)", expanded=False):
        retrieved_raw = st.text_area(
            "Retrieved Docs (JSON list)",
            value=json.dumps(sample_state.get("retrieved_docs", []), ensure_ascii=False, indent=2)
            if sample_state.get("retrieved_docs")
            else "",
            height=120,
        )
        reference_docs_raw = st.text_area(
            "Gold / Reference Doc IDs (JSON list)",
            value=json.dumps(sample_state.get("reference_docs", []), ensure_ascii=False)
            if sample_state.get("reference_docs")
            else "",
            height=80,
        )
        relevance_raw = st.text_area(
            "Relevance Annotations (JSON dict)",
            value=json.dumps(sample_state.get("relevance", {}), ensure_ascii=False, indent=2)
            if sample_state.get("relevance")
            else "",
            height=80,
        )
        citation_map_raw = st.text_area(
            "Citation Map (JSON dict)",
            value=json.dumps(sample_state.get("citation_map", {}), ensure_ascii=False, indent=2)
            if sample_state.get("citation_map")
            else "",
            height=80,
        )
        doc_text_raw = st.text_area(
            "Document Text (JSON dict: doc_id -> text)",
            value=json.dumps(sample_state.get("doc_text", {}), ensure_ascii=False, indent=2)
            if sample_state.get("doc_text")
            else "",
            height=130,
        )
else:
    retrieved_raw = ""
    reference_docs_raw = ""
    relevance_raw = ""
    citation_map_raw = ""
    doc_text_raw = ""

if ui_mode == "Advanced (RAG + Telemetry)":
    with st.expander("3. Telemetry / Cost (optional)", expanded=False):
        prompt_tokens = st.number_input("Prompt tokens", value=int(sample_state.get("prompt_tokens") or 0))
        completion_tokens = st.number_input("Completion tokens", value=int(sample_state.get("completion_tokens") or 0))
        model_name = st.text_input("Model name", value=sample_state.get("model", ""))
        gen_start_time = st.text_input("gen_start_time", value=str(sample_state.get("gen_start_time", "")))
        gen_first_token_time = st.text_input("gen_first_token_time", value=str(sample_state.get("gen_first_token_time", "")))
        e2e_start = st.text_input("e2e_start", value=str(sample_state.get("e2e_start", "")))
        e2e_end = st.text_input("e2e_end", value=str(sample_state.get("e2e_end", "")))
else:
    prompt_tokens = sample_state.get("prompt_tokens", 0)
    completion_tokens = sample_state.get("completion_tokens", 0)
    model_name = sample_state.get("model", "")
    gen_start_time = sample_state.get("gen_start_time", "")
    gen_first_token_time = sample_state.get("gen_first_token_time", "")
    e2e_start = sample_state.get("e2e_start", "")
    e2e_end = sample_state.get("e2e_end", "")

# single evaluate button
evaluate = st.button("Evaluate", type="primary")


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
if evaluate:
    reference = parse_json_input("Reference Answers", reference_raw, [])
    if isinstance(reference, str):
        reference = [reference]

    retrieved_docs = parse_json_input("Retrieved Docs", retrieved_raw, [])
    reference_docs = parse_json_input("Gold / Reference Doc IDs", reference_docs_raw, [])
    relevance = parse_json_input("Relevance", relevance_raw, {})
    citation_map = parse_json_input("Citation Map", citation_map_raw, {})
    doc_text = parse_json_input("Document Text", doc_text_raw, {})

    st.session_state["current_sample"] = {
        "question": question,
        "answer": answer,
        "reference": reference,
        "retrieved_docs": retrieved_docs,
        "reference_docs": reference_docs,
        "relevance": relevance,
        "citation_map": citation_map,
        "doc_text": doc_text,
        "prompt_tokens": int(prompt_tokens) if prompt_tokens else None,
        "completion_tokens": int(completion_tokens) if completion_tokens else None,
        "total_tokens": int(prompt_tokens + completion_tokens) if (prompt_tokens or completion_tokens) else None,
        "model": model_name,
        "gen_start_time": _maybe_float(gen_start_time),
        "gen_first_token_time": _maybe_float(gen_first_token_time),
        "e2e_start": _maybe_float(e2e_start),
        "e2e_end": _maybe_float(e2e_end),
    }
    st.success("Input saved. Scroll down to see the results.", icon="✅")

# -----------------------------------------------------------------------------
# RESULTS (depends on mode)
# -----------------------------------------------------------------------------
sample = st.session_state["current_sample"]

if sample.get("question") or sample.get("answer"):
    sample = autofill_missing(sample)
    st.session_state["current_sample"] = sample

    # judge 一定要跑
    judge = LLMJudge()
    grade = judge.grade(
        sample.get("question", ""),
        sample.get("answer", ""),
        sample.get("reference"),
    )

    st.subheader("Judge Result")
    judge_cols = st.columns(4)
    judge_cols[0].metric("Judge Score (F1)", _fmt_metric(grade.get("judge_score")))
    judge_cols[1].metric("Precision", _fmt_metric(grade.get("judge_precision")))
    judge_cols[2].metric("Recall", _fmt_metric(grade.get("judge_recall")))
    judge_cols[3].write(f"Best-matched reference: {grade.get('judge_ref_match', '—')}")

    # ------------------------------------------------------
    # BASIC MODE  → only generation-related stuff + bar
    # ------------------------------------------------------
    if ui_mode == "Basic (QA only)":
        # 我們還是用同一套 compute_metrics，然後只挑 generation 那一類
        results, categorized = compute_metrics(sample, cfg)
        gen_entries = categorized.get("✍️ Generation Quality", [])

        # 轉成 {name: value} 給 bar chart
        basic_metrics = {}
        for entry in gen_entries:
            # None 的就先放著不畫
            basic_metrics[entry["label"]] = entry["primary_value"]

        st.divider()
        st.subheader("Generation Metrics")
        # 列表形式
        for entry in gen_entries:
            st.metric(entry["label"], _fmt_metric(entry["primary_value"]))

        st.subheader("Visualization")
        render_basic_bar(basic_metrics)

        # basic 模式就不要顯示雷達 / 其他 tabs / raw json
        # 結束
    else:
        # ------------------------------------------------------
        # ADVANCED MODE → full view
        # ------------------------------------------------------
        results, categorized = compute_metrics(sample, cfg)
        norm_scores = aggregated_scores_for_radar(categorized)

        st.divider()
        st.subheader("Metrics by Category")
        tabs = st.tabs(list(categorized.keys()))
        for tab, (category, entries) in zip(tabs, categorized.items()):
            render_category_tab(tab, entries)

        st.divider()
        st.subheader("Overview Radar (normalized 0–1)")
        render_radar(norm_scores)

        st.divider()
        with st.expander("Raw JSON"):
            st.json(
                {
                    "sample": sample,
                    "judge": grade,
                    "metrics": results,
                    "category_summary_normalized": norm_scores,
                },
                expanded=False,
            )

else:
    st.info("Enter at least a Question and a Model Answer, or click 'Load demo case' in the sidebar.", icon="ℹ️")
