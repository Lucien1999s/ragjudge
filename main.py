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