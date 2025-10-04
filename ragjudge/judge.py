from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable
import re
from collections import Counter

class LLMJudge:
    """
    Minimal judge:
      - With reference(s): return token-level F1 (0~1). Supports str or list/tuple of str.
      - No reference: return None (placeholder for future LLM grading).
    """

    def __init__(self, *, lowercase: bool = True, strip_punct: bool = True) -> None:
        self.lowercase = lowercase
        self.strip_punct = strip_punct

    def grade(self, question: str, answer: str, reference: Optional[Iterable[str] | str] = None) -> Dict[str, Any]:
        if reference is None:
            return {"judge_score": None, "judge_mode": "noref"}

        # normalize refs to a list of strings (ignore non-str safely)
        refs: List[str] = []
        if isinstance(reference, (list, tuple)):
            refs = [r for r in reference if isinstance(r, str)]
        elif isinstance(reference, str):
            refs = [reference]

        if not refs:
            # reference given but none valid
            return {"judge_score": None, "judge_mode": "ref", "judge_note": "no valid reference strings"}

        best = {"f1": -1.0, "p": 0.0, "r": 0.0, "ref": None}
        for ref in refs:
            p, r, f1 = self._token_f1(answer or "", ref or "")
            if f1 > best["f1"]:
                best.update({"f1": f1, "p": p, "r": r, "ref": ref})

        return {
            "judge_score": best["f1"],
            "judge_mode": "ref",
            "judge_precision": best["p"],
            "judge_recall": best["r"],
            "judge_ref_match": best["ref"],  # which reference matched best
        }

    # ------------------------ helpers ------------------------

    def _tokenize(self, text: Any) -> List[str]:
        # be robust to non-str inputs
        t = "" if text is None else (text if isinstance(text, str) else str(text))
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = re.sub(r"[^\w\s]", " ", t)
        return [tok for tok in t.split() if tok]

    def _token_f1(self, pred: str, gold: str) -> Tuple[float, float, float]:
        pc, gc = Counter(self._tokenize(pred)), Counter(self._tokenize(gold))
        if not pc and not gc:
            return 1.0, 1.0, 1.0
        if not pc or not gc:
            return 0.0, 0.0, 0.0
        overlap = sum((pc & gc).values())
        p = overlap / max(1, sum(pc.values()))
        r = overlap / max(1, sum(gc.values()))
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return p, r, f1
