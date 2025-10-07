from __future__ import annotations
from typing import Any, Dict, Iterable, List
from .judge import LLMJudge
from .metrics import Metric


def evaluate(
    samples: Iterable[Dict[str, Any]],
    judge: LLMJudge,
    metrics: List[Metric],
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of RAG samples using a given judge and metric set.

    This function sequentially:
      1. Invokes the `LLMJudge` to score each (question, answer, reference) triple.
      2. Computes all registered `Metric` objects on each sample.
      3. Aggregates the per-sample results into a flat list of dictionaries.

    Parameters
    ----------
    samples : Iterable[Dict[str, Any]]
        Iterable of input samples. Each sample should contain:
        - "id" : unique identifier
        - "question" : str
        - "answer" : str
        - "reference" : Optional[str]
    judge : LLMJudge
        Judge instance that performs correctness grading.
    metrics : List[Metric]
        List of metric instances to compute on each sample.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, one per sample, containing:
        - original metadata ("id")
        - judge results (e.g., correctness, reasoning)
        - metric scores (e.g., recall, rouge, faithfulness)
    """
    rows: List[Dict[str, Any]] = []

    for s in samples:
        row: Dict[str, Any] = {"id": s.get("id")}
        try:
            row.update(
                judge.grade(
                    s.get("question", ""),
                    s.get("answer", ""),
                    s.get("reference"),
                )
            )
        except Exception as e:
            row["judge_error"] = str(e)

        for m in metrics:
            try:
                row.update(m.compute(s))
            except Exception as e:
                row[f"{m.__class__.__name__}_error"] = str(e)

        rows.append(row)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the mean value for all numeric fields across evaluation rows.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        Output of `evaluate()`, where each entry is a per-sample result.

    Returns
    -------
    Dict[str, Any]
        A dictionary of mean scores.
        Each numeric key is prefixed with "mean_", e.g.:
        {"mean_f1": 0.82, "mean_recall": 0.74}
    """
    if not rows:
        return {}

    keys = set().union(*rows)
    out: Dict[str, Any] = {}

    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[f"mean_{k}"] = sum(vals) / len(vals)

    return out
