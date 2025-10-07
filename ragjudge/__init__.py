"""
ragjudge
========

A modular and extensible evaluation framework for Retrieval-Augmented Generation (RAG) systems.

This package exposes a unified API for evaluating retrieval, generation, and
faithfulness metrics via a consistent interface:

    from ragjudge import evaluate, LLMJudge, RecallAtK

    results = evaluate(samples, LLMJudge(), [RecallAtK(k=5)])

The framework is designed for easy extension — simply implement new Metric
subclasses or custom judges and register them within your workflow.
"""

__version__ = "0.0.1"

# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

from .judge import LLMJudge  # noqa: F401

from .metrics import (  # noqa: F401
    Metric,
    # Retrieval metrics
    RecallAtK,
    NDCGAtK,
    MRR,
    DiversityAtK,
    RedundancyAtK,
    # Generation metrics
    ExactMatch,
    F1Score,
    RougeL,
    RougeN,
    BLEU,
    BERTScore,
    AnswerRelevancySim,
    # Faithfulness / factuality metrics
    FaithfulnessNLI,
    ContextPrecisionAtK,
    ContextPrecisionSim,
    ContextRecallAtK,
    ContextRecallSim,
    CitationAccuracy,
    EvidenceDensity,
    # Efficiency / cost metrics
    TTFT,
    E2ELatency,
    TokensPerRequest,
    CostPer1kRequest,
)

from .evaluate import evaluate, summarize  # noqa: F401

__all__ = [
    # Core
    "LLMJudge",
    "Metric",
    "evaluate",
    "summarize",
    # Retrieval
    "RecallAtK",
    "NDCGAtK",
    "MRR",
    "DiversityAtK",
    "RedundancyAtK",
    # Generation
    "ExactMatch",
    "F1Score",
    "RougeL",
    "RougeN",
    "BLEU",
    "BERTScore",
    "AnswerRelevancySim",
    # Faithfulness
    "FaithfulnessNLI",
    "ContextPrecisionAtK",
    "ContextPrecisionSim",
    "ContextRecallAtK",
    "ContextRecallSim",
    "CitationAccuracy",
    "EvidenceDensity",
    # Efficiency
    "TTFT",
    "E2ELatency",
    "TokensPerRequest",
    "CostPer1kRequest",
]
