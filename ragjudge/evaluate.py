from __future__ import annotations
from typing import Any, Dict, Iterable, List
from .judge import LLMJudge
from .metrics import Metric

def evaluate(samples: Iterable[Dict[str, Any]], judge: LLMJudge, metrics: List[Metric]) -> List[Dict[str, Any]]:
    """Run judge + metrics per-sample and return rows."""
    rows: List[Dict[str, Any]] = []
    for s in samples:
        row: Dict[str, Any] = {"id": s.get("id")}
        row.update(judge.grade(s.get("question",""), s.get("answer",""), s.get("reference")))
        for m in metrics:
            row.update(m.compute(s))
        rows.append(row)
    return rows

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean over numeric fields; ignores None."""
    if not rows:
        return {}
    keys = set().union(*rows)
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if vals:
            out[f"mean_{k}"] = sum(vals) / len(vals)
    return out
