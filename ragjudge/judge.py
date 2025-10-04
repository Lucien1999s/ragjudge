# judge.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import unicodedata
from collections import Counter

class LLMJudge:
    """
    Minimal judge (multilingual):
      - With reference(s): token-level F1 (0~1). Supports str or list/tuple of str.
      - No reference: return None.
    """

    def __init__(self, *, lowercase: bool = True, strip_punct: bool = True) -> None:
        self.lowercase = lowercase
        self.strip_punct = strip_punct

    def grade(self, question: str, answer: str, reference=None) -> Dict[str, Any]:
        if reference is None:
            return {"judge_score": None, "judge_mode": "noref"}

        refs: List[str] = []
        if isinstance(reference, (list, tuple)):
            refs = [r for r in reference if isinstance(r, str)]
        elif isinstance(reference, str):
            refs = [reference]

        if not refs:
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
            "judge_ref_match": best["ref"],
        }

    # ------------------------ helpers ------------------------

    def _tokenize(self, text: Any) -> List[str]:
        t = "" if text is None else (text if isinstance(text, str) else str(text))
        return _cjk_aware_tokenize_local(
            t,
            lowercase=self.lowercase,
            strip_punct=self.strip_punct,
            collapse_whitespace=True,
        )

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


def _cjk_aware_tokenize_local(
    text: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = True,
    collapse_whitespace: bool = True,
) -> List[str]:
    # 基於你 metrics 裡的想法的輕量本地版本
    def _unicode_normalize(s: str) -> str:
        t = unicodedata.normalize("NFKC", s or "")
        return t.casefold() if lowercase else t

    def _strip_unicode_punct(s: str) -> str:
        return "".join(ch for ch in s if (unicodedata.category(ch)[0] not in ("P", "S")))

    def _is_cjk(ch: str) -> bool:
        cp = ord(ch)
        return (
            (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or
            (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
            (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
            (0xF900 <= cp <= 0xFAFF) or
            (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF) or (0x31F0 <= cp <= 0x31FF) or
            (0x1100 <= cp <= 0x11FF) or (0x3130 <= cp <= 0x318F) or (0xAC00 <= cp <= 0xD7AF)
        )

    def _is_word_char(ch: str) -> bool:
        cat = unicodedata.category(ch)
        return cat and (cat[0] in ("L", "N"))

    t = _unicode_normalize(text)
    if strip_punct:
        t = _strip_unicode_punct(t)

    tokens, buf = [], []
    def flush():
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in t:
        if ch.isspace():
            flush()
            continue
        if _is_cjk(ch):
            flush()
            tokens.append(ch)
        else:
            if _is_word_char(ch):
                buf.append(ch)
            else:
                flush()
    flush()
    if collapse_whitespace:
        pass
    return tokens
