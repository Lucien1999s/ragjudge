from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Set
import math

Doc = Union[str, int]
Ranked = Union[Doc, Tuple[Doc, float]]

# -------------------------- Base --------------------------

class Metric:
    """Abstract Metric: compute(sample)->dict of scalar(s)."""
    name: str = "metric"
    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# -------------------------- Utilities (local) --------------------------

def _as_id_set(values: Iterable[Union[Doc, Ranked]]) -> Set[Doc]:
    ids: Set[Doc] = set()
    for v in values or []:
        ids.add(v[0] if isinstance(v, tuple) else v)
    return ids

def _as_ranked_list(values: Iterable[Ranked]) -> List[Ranked]:
    if not values:
        return []
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return list(values)

def _ranked_to_ids(ranked: List[Ranked], *, dedup: bool) -> List[Doc]:
    out: List[Doc] = []
    seen: Set[Doc] = set()
    for item in ranked:
        doc_id = item[0] if isinstance(item, tuple) else item
        if dedup:
            if doc_id in seen:
                continue
            seen.add(doc_id)
        out.append(doc_id)
    return out

# -------------------------- Recall@k --------------------------

class RecallAtK(Metric):
    """
    Recall@k for retrieval.

    Sample:
      - gold_field: iterable of relevant doc IDs (default "reference_docs")
      - retrieved_field: ranked retrieved list (IDs or (ID, score); default "retrieved_docs")
    Config:
      - k: cutoff; None -> use full length
      - dedup: de-duplicate retrieved IDs before truncation
    """
    name = "recall_at_k"

    def __init__(self, k: int | None = 5, *, gold_field: str = "reference_docs",
                 retrieved_field: str = "retrieved_docs", dedup: bool = True) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.k = k
        self.gold_field = gold_field
        self.retrieved_field = retrieved_field
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        gold_ids = _as_id_set(sample.get(self.gold_field, []))
        if not gold_ids:
            return {self._key(): None}
        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        k_cut = len(retrieved_ids) if self.k is None else self.k
        topk = set(retrieved_ids[:k_cut])
        hit = len(gold_ids & topk)
        score = hit / len(gold_ids)
        return {self._key(): score}

    def _key(self) -> str:
        return f"recall@{'all' if self.k is None else self.k}"

# -------------------------- nDCG@k --------------------------

class NDCGAtK(Metric):
    """
    nDCG@k for ranked retrieval.

    Sample:
      - graded relevance (preferred): sample[rel_field] = {doc_id: rel>=0}
      - or binary fallback:          sample[binary_fallback_field] = iterable of relevant IDs
      - retrieved ranking:           sample[retrieved_field] = [id | (id,score), ...]
    Config:
      - k: cutoff; None -> full length
      - gain: "exp" (2^rel-1) or "linear" (rel)
      - dedup: de-duplicate retrieved IDs before truncation
    """
    name = "ndcg_at_k"

    def __init__(
        self,
        k: int | None = 5,
        *,
        rel_field: str = "relevance",
        binary_fallback_field: str = "reference_docs",
        retrieved_field: str = "retrieved_docs",
        gain: str = "exp",
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        if gain not in {"exp", "linear"}:
            raise ValueError('gain must be one of {"exp","linear"}')
        self.k = k
        self.rel_field = rel_field
        self.binary_fallback_field = binary_fallback_field
        self.retrieved_field = retrieved_field
        self.gain = gain
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        rel_map = self._build_relevance_map(sample)
        if not rel_map:
            return {self._key(): None}

        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        k_cut = len(retrieved_ids) if self.k is None else self.k

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k_cut], start=1):
            rel = rel_map.get(doc_id, 0.0)
            if rel > 0:
                dcg += self._gain(rel) / self._disc(i)

        # IDCG
        sorted_rels = sorted(rel_map.values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_rels[:k_cut], start=1):
            if rel > 0:
                idcg += self._gain(rel) / self._disc(i)

        score = (dcg / idcg) if idcg > 0 else None
        return {self._key(): score}

    # helpers
    def _key(self) -> str:
        return f"ndcg@{'all' if self.k is None else self.k}"

    def _build_relevance_map(self, sample: Dict[str, Any]) -> Dict[Doc, float]:
        graded = sample.get(self.rel_field)
        rel_map: Dict[Doc, float] = {}
        if isinstance(graded, dict) and graded:
            for k, v in graded.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                if val > 0:
                    rel_map[k] = val
            if rel_map:
                return rel_map
        for gid in _as_id_set(sample.get(self.binary_fallback_field, [])):
            rel_map[gid] = 1.0
        return rel_map

    def _gain(self, rel: float) -> float:
        return (2.0 ** rel - 1.0) if self.gain == "exp" else rel

    @staticmethod
    def _disc(rank: int) -> float:
        return math.log2(rank + 1.0)

# -------------------------- MRR --------------------------

class MRR(Metric):
    """
    MRR (Mean Reciprocal Rank) with optional cutoff k.

    Sample:
      - graded relevance (preferred): sample[rel_field] = {doc_id: rel>0}
      - or binary fallback:           sample[binary_fallback_field]
      - retrieved ranking:            sample[retrieved_field]
    """
    name = "mrr"

    def __init__(
        self,
        k: int | None = None,
        *,
        rel_field: str = "relevance",
        binary_fallback_field: str = "reference_docs",
        retrieved_field: str = "retrieved_docs",
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.k = k
        self.rel_field = rel_field
        self.binary_fallback_field = binary_fallback_field
        self.retrieved_field = retrieved_field
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        relevant = self._relevant_set(sample)
        if not relevant:
            return {self._key(): None}

        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        k_cut = len(retrieved_ids) if self.k is None else self.k

        score = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k_cut], start=1):
            if doc_id in relevant:
                score = 1.0 / i
                break
        return {self._key(): score}

    # helpers
    def _key(self) -> str:
        return f"mrr@{'all' if self.k is None else self.k}"

    def _relevant_set(self, sample: Dict[str, Any]) -> Set[Doc] | None:
        graded = sample.get(self.rel_field)
        if isinstance(graded, dict) and graded:
            rel = {k for k, v in graded.items() if self._is_pos(v)}
            if rel:
                return rel
        gold_ids = _as_id_set(sample.get(self.binary_fallback_field, []))
        return gold_ids if gold_ids else None

    @staticmethod
    def _is_pos(v: Any) -> bool:
        try:
            return float(v) > 0.0
        except Exception:
            return False

# -------------------------- Diversity / Redundancy --------------------------

class DiversityAtK(Metric):
    """
    Diversity@k for retrieved results.

    Modes:
      - "cluster": sample[cluster_field] = {doc_id: cluster_label}
      - "embedding": sample[embedding_field] = {doc_id: [float,...]} with cosine >= threshold grouped
    """
    name = "diversity_at_k"

    def __init__(
        self,
        k: int | None = 5,
        *,
        mode: str = "cluster",
        cluster_field: str = "doc_cluster",
        embedding_field: str = "doc_embedding",
        retrieved_field: str = "retrieved_docs",
        similarity_thresh: float = 0.9,
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        if mode not in {"cluster", "embedding"}:
            raise ValueError('mode must be one of {"cluster","embedding"}')
        if not (0.0 < similarity_thresh <= 1.0):
            raise ValueError("similarity_thresh must be in (0,1].")
        self.k = k
        self.mode = mode
        self.cluster_field = cluster_field
        self.embedding_field = embedding_field
        self.retrieved_field = retrieved_field
        self.similarity_thresh = similarity_thresh
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        if not retrieved_ids:
            return {self._key(): None}
        k_cut = len(retrieved_ids) if self.k is None else self.k
        topk = retrieved_ids[:k_cut]

        if self.mode == "cluster":
            clusters = sample.get(self.cluster_field, {}) or {}
            unique_clusters = set(clusters.get(doc_id, doc_id) for doc_id in topk)
            c = len(unique_clusters)
        else:
            # Greedy clustering by cosine similarity threshold
            emb_map = sample.get(self.embedding_field, {}) or {}
            reps: List[List[float]] = []
            c = 0
            for doc_id in topk:
                vec = emb_map.get(doc_id)
                if not (isinstance(vec, list) and vec):
                    c += 1
                    reps.append([1.0])  # dummy to keep shape
                    continue
                placed = False
                for rep in reps:
                    if _cosine_sim(vec, rep) >= self.similarity_thresh:
                        placed = True
                        break
                if not placed:
                    reps.append(vec)
                    c += 1

        score = c / max(1, len(topk))
        return {self._key(): score}

    def _key(self) -> str:
        return f"diversity@{'all' if self.k is None else self.k}"

class RedundancyAtK(Metric):
    """Redundancy@k = 1 - Diversity@k (same configuration)."""
    name = "redundancy_at_k"

    def __init__(self, **kwargs: Any) -> None:
        self._div = DiversityAtK(**kwargs)
        self.k = self._div.k

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        d = self._div.compute(sample)
        (key, val), = d.items()
        if val is None:
            return {self._key(): None}
        return {self._key(): 1.0 - float(val)}

    def _key(self) -> str:
        return f"redundancy@{'all' if self.k is None else self.k}"

class ExactMatch(Metric):
    """
    Exact Match (EM) on generated answer vs reference answer(s).

    Sample:
      - answer_field:    sample[answer_field] -> predicted text (default "answer")
      - reference_field: sample[reference_field] -> gold text or list[str] (default "reference")

    Normalization (configurable):
      - lowercase:           lower-case text (default True)
      - strip_punct:         remove punctuation (default True)
      - collapse_whitespace: collapse multiple spaces to one (default True)
      - remove_articles:     drop "a", "an", "the" (English-only; default False)
      - extra_strip_chars:   additional characters to strip (e.g., "[]()") (default "")

    Returns:
      { "em": 1.0 or 0.0 } ; if no reference provided -> { "em": None }.
    """
    name = "em"

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
    ) -> None:
        self.answer_field = answer_field
        self.reference_field = reference_field
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pred = sample.get(self.answer_field)
        gold = sample.get(self.reference_field)

        if gold is None:
            return {"em": None}

        pred_n = self._normalize(pred if isinstance(pred, str) else "")
        # 支援多參考答案
        if isinstance(gold, (list, tuple)):
            gold_norm = [self._normalize(g) for g in gold if isinstance(g, str)]
            score = 1.0 if gold_norm and pred_n in gold_norm else (0.0 if gold_norm else None)
        else:
            gold_n = self._normalize(gold if isinstance(gold, str) else "")
            score = 1.0 if (gold is not None and pred_n == gold_n) else 0.0

        return {"em": score}

    # ------------------------ helpers ------------------------

    def _normalize(self, s: str) -> str:
        import re, string
        t = s or ""
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            keep = "".join(ch for ch in t if ch not in string.punctuation)
            t = keep
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t

class F1Score(Metric):
    """
    Token-level F1 between generated answer and reference answer(s).

    Sample:
      - answer_field:    sample[answer_field] -> predicted text (default "answer")
      - reference_field: sample[reference_field] -> gold text or list[str] (default "reference")

    Normalization (configurable):
      - lowercase (bool, default True)
      - strip_punct (bool, default True)
      - collapse_whitespace (bool, default True)
      - remove_articles (bool, default False)  # English-only
      - extra_strip_chars (str, default "")

    Multi-reference handling:
      - When reference is a list/tuple, compute F1 vs each and take the best.
    
    Returns:
      { "f1": float in [0,1] or None,
        "f1_precision": float or None,
        "f1_recall": float or None }
    """
    name = "f1"

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
        include_pr: bool = True,   # whether to output precision/recall alongside f1
    ) -> None:
        self.answer_field = answer_field
        self.reference_field = reference_field
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars
        self.include_pr = include_pr

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pred = sample.get(self.answer_field)
        gold = sample.get(self.reference_field)

        if gold is None:
            out = {"f1": None}
            if self.include_pr:
                out.update({"f1_precision": None, "f1_recall": None})
            return out

        # normalize to strings
        pred_tokens = self._tokenize(pred if isinstance(pred, str) else "")

        # support multiple references
        if isinstance(gold, (list, tuple)):
            candidates = [g for g in gold if isinstance(g, str)]
        else:
            candidates = [gold if isinstance(gold, str) else ""]

        best = {"f1": -1.0, "p": 0.0, "r": 0.0}
        for g in candidates:
            p, r, f1 = self._prf1(pred_tokens, self._tokenize(g))
            if f1 > best["f1"]:
                best.update({"f1": f1, "p": p, "r": r})

        out = {"f1": max(0.0, best["f1"])}  # guard if list empty
        if self.include_pr:
            out.update({"f1_precision": best["p"], "f1_recall": best["r"]})
        return out

    # ------------------------ helpers ------------------------

    def _tokenize(self, s: str) -> List[str]:
        import re, string, unicodedata
        t = s or ""
        t = unicodedata.normalize("NFKC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = "".join(ch for ch in t if ch not in string.punctuation)
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t.split() if t else []

    def _prf1(self, pred_toks: List[str], gold_toks: List[str]) -> Tuple[float, float, float]:
        from collections import Counter
        if not pred_toks and not gold_toks:
            return 1.0, 1.0, 1.0
        if not pred_toks or not gold_toks:
            return 0.0, 0.0, 0.0
        pc, gc = Counter(pred_toks), Counter(gold_toks)
        overlap = sum((pc & gc).values())
        p = overlap / max(1, sum(pc.values()))
        r = overlap / max(1, sum(gc.values()))
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return p, r, f1

class RougeN(Metric):
    """
    ROUGE-N (recall-oriented) between generated answer and reference(s).
    Computes precision, recall, and F1 on n-gram overlap.

    Sample:
      - answer_field:    sample[answer_field] -> predicted text (default "answer")
      - reference_field: sample[reference_field] -> gold text or list[str] (default "reference")

    Config:
      - n:                   size of n-grams (default 1)
      - lowercase, strip_punct, collapse_whitespace, remove_articles, extra_strip_chars:
                             normalization flags (same semantics as ExactMatch/F1Score)
      - include_pr:          whether to output precision/recall alongside F1 (default True)
      - agg:                 aggregation when multiple references are provided:
                               "max" (take best over refs) or "avg" (average over refs). Default "max".

    Returns:
      { f"rouge{n}_f1": float or None,
        f"rouge{n}_precision": float or None,
        f"rouge{n}_recall": float or None }
    """
    name = "rouge_n"

    def __init__(
        self,
        n: int = 1,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
        include_pr: bool = True,
        agg: str = "max",   # "max" or "avg"
    ) -> None:
        if n <= 0:
            raise ValueError("n must be >= 1.")
        if agg not in {"max", "avg"}:
            raise ValueError('agg must be one of {"max","avg"}')
        self.n = n
        self.answer_field = answer_field
        self.reference_field = reference_field
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars
        self.include_pr = include_pr
        self.agg = agg

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pred = sample.get(self.answer_field)
        gold = sample.get(self.reference_field)
        key_prefix = f"rouge{self.n}_"

        if gold is None:
            out = {key_prefix + "f1": None}
            if self.include_pr:
                out.update({key_prefix + "precision": None, key_prefix + "recall": None})
            return out

        pred_tokens = self._tokenize(pred if isinstance(pred, str) else "")

        # collect references
        refs = []
        if isinstance(gold, (list, tuple)):
            refs = [g for g in gold if isinstance(g, str)]
        elif isinstance(gold, str):
            refs = [gold]

        if not refs:
            out = {key_prefix + "f1": None}
            if self.include_pr:
                out.update({key_prefix + "precision": None, key_prefix + "recall": None})
            return out

        # compute per-ref then aggregate
        triples = []
        for g in refs:
            p, r, f1 = self._ngram_prf1(pred_tokens, self._tokenize(g), self.n)
            triples.append((p, r, f1))

        if self.agg == "avg":
            P = sum(t[0] for t in triples) / len(triples)
            R = sum(t[1] for t in triples) / len(triples)
            F = sum(t[2] for t in triples) / len(triples)
        else:  # max
            P, R, F = max(triples, key=lambda t: t[2])

        out = {key_prefix + "f1": F}
        if self.include_pr:
            out.update({key_prefix + "precision": P, key_prefix + "recall": R})
        return out

    # ------------------------ helpers ------------------------

    def _tokenize(self, s: str) -> List[str]:
        import re, string, unicodedata
        t = s or ""
        t = unicodedata.normalize("NFKC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = "".join(ch for ch in t if ch not in string.punctuation)
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t.split() if t else []

    def _ngram_counts(self, toks: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        from collections import Counter
        if n <= 0 or not toks or len(toks) < n:
            return Counter()
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        return Counter(grams)

    def _ngram_prf1(self, pred_toks: List[str], gold_toks: List[str], n: int) -> Tuple[float, float, float]:
        from collections import Counter
        pc = self._ngram_counts(pred_toks, n)
        gc = self._ngram_counts(gold_toks, n)
        if not pc and not gc:
            return 1.0, 1.0, 1.0
        if not pc or not gc:
            return 0.0, 0.0, 0.0
        overlap = sum((pc & gc).values())
        p = overlap / max(1, sum(pc.values()))
        r = overlap / max(1, sum(gc.values()))
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return p, r, f1


class RougeL(Metric):
    """
    ROUGE-L (LCS-based) between generated answer and reference(s).
    Computes precision, recall, and F1 using the length of the longest common subsequence.

    Sample / Config:
      - Same fields and normalization flags as RougeN (answer_field, reference_field, ...).
      - agg: "max" (default) or "avg" over multiple references.

    Returns:
      { "rougeL_f1": float or None,
        "rougeL_precision": float or None,
        "rougeL_recall": float or None }
    """
    name = "rouge_l"

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
        include_pr: bool = True,
        agg: str = "max",
    ) -> None:
        if agg not in {"max", "avg"}:
            raise ValueError('agg must be one of {"max","avg"}')
        self.answer_field = answer_field
        self.reference_field = reference_field
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars
        self.include_pr = include_pr
        self.agg = agg

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        pred = sample.get(self.answer_field)
        gold = sample.get(self.reference_field)

        if gold is None:
            out = {"rougeL_f1": None}
            if self.include_pr:
                out.update({"rougeL_precision": None, "rougeL_recall": None})
            return out

        pred_tokens = self._tokenize(pred if isinstance(pred, str) else "")

        # refs
        refs = []
        if isinstance(gold, (list, tuple)):
            refs = [g for g in gold if isinstance(g, str)]
        elif isinstance(gold, str):
            refs = [gold]
        if not refs:
            out = {"rougeL_f1": None}
            if self.include_pr:
                out.update({"rougeL_precision": None, "rougeL_recall": None})
            return out

        triples = []
        for g in refs:
            g_tokens = self._tokenize(g)
            p, r, f1 = self._lcs_prf1(pred_tokens, g_tokens)
            triples.append((p, r, f1))

        if self.agg == "avg":
            P = sum(t[0] for t in triples) / len(triples)
            R = sum(t[1] for t in triples) / len(triples)
            F = sum(t[2] for t in triples) / len(triples)
        else:
            P, R, F = max(triples, key=lambda t: t[2])

        out = {"rougeL_f1": F}
        if self.include_pr:
            out.update({"rougeL_precision": P, "rougeL_recall": R})
        return out

    # ------------------------ helpers ------------------------

    def _tokenize(self, s: str) -> List[str]:
        import re, string, unicodedata
        t = s or ""
        t = unicodedata.normalize("NFKC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = "".join(ch for ch in t if ch not in string.punctuation)
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t.split() if t else []

    def _lcs_prf1(self, pred_toks: List[str], gold_toks: List[str]) -> Tuple[float, float, float]:
        # LCS length
        lcs_len = self._lcs_length(pred_toks, gold_toks)
        if len(pred_toks) == 0 and len(gold_toks) == 0:
            return 1.0, 1.0, 1.0
        if len(pred_toks) == 0 or len(gold_toks) == 0:
            return 0.0, 0.0, 0.0
        p = lcs_len / max(1, len(pred_toks))
        r = lcs_len / max(1, len(gold_toks))
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        return p, r, f1

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        # O(len(a)*len(b)) DP；長度通常很短，夠用
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0
        dp = [0] * (lb + 1)
        for i in range(1, la + 1):
            prev = 0
            for j in range(1, lb + 1):
                tmp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = tmp
        return dp[lb]

class BLEU(Metric):
    """
    Corpus-independent (per-sample) BLEU with multiple references.

    Definition (per sample):
      BLEU = BP * exp( sum_{n=1..N} w_n * log p_n )
      - p_n: modified n-gram precision with clipping against refs
      - BP (brevity penalty): 1 if c>r else exp(1 - r/c), c=len(hyp), r=closest_ref_len
      - N: max n-gram order (default 4)
      - w_n: uniform weights (1/N) by default

    Sample:
      - answer_field:    sample[answer_field] -> hypothesis text (default "answer")
      - reference_field: sample[reference_field] -> gold text or list[str] (default "reference")

    Normalization (same options as EM/F1/ROUGE):
      - lowercase, strip_punct, collapse_whitespace, remove_articles, extra_strip_chars

    Config:
      - max_order: int, default 4
      - weights: Optional[List[float]] (len == max_order); if None -> uniform
      - smoothing: {"none","epsilon","add1"}  default "epsilon"
         * "none":    if any p_n==0, BLEU becomes 0 (log(0))
         * "epsilon": replace 0 precisions with tiny value (1e-9)
         * "add1":    Laplace smoothing (add-1) on counts for every n
      - include_components: whether to output p1..pN, BP, ref/hyp length (default True)

    Returns:
      {
        "bleu": float in [0,1] or None,
        "bleu_bp": ...,
        "bleu_ref_len": int,
        "bleu_hyp_len": int,
        "bleu_p1": float, ..., "bleu_pN": float
      }
      If no valid reference -> {"bleu": None, ... components also None}
    """
    name = "bleu"

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
        max_order: int = 4,
        weights: list[float] | None = None,
        smoothing: str = "epsilon",         # "none" | "epsilon" | "add1"
        include_components: bool = True,
    ) -> None:
        if max_order <= 0:
            raise ValueError("max_order must be >= 1")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != max_order:
                raise ValueError("weights must be a list with length == max_order")
            s = sum(weights)
            if s <= 0:
                raise ValueError("weights must sum to > 0")
            self.weights = [w / s for w in weights]
        else:
            self.weights = [1.0 / max_order] * max_order

        if smoothing not in {"none", "epsilon", "add1"}:
            raise ValueError('smoothing must be one of {"none","epsilon","add1"}')

        self.answer_field = answer_field
        self.reference_field = reference_field
        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars
        self.max_order = max_order
        self.smoothing = smoothing
        self.include_components = include_components

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        hyp = sample.get(self.answer_field)
        refs = sample.get(self.reference_field)

        # normalize references to list[str]
        ref_texts: List[str] = []
        if isinstance(refs, (list, tuple)):
            ref_texts = [r for r in refs if isinstance(r, str)]
        elif isinstance(refs, str):
            ref_texts = [refs]

        if not ref_texts:
            return self._empty_out(None)

        hyp_tokens = self._tokenize(hyp if isinstance(hyp, str) else "")
        ref_token_lists = [self._tokenize(r) for r in ref_texts]

        # lengths
        c = len(hyp_tokens)
        r = self._closest_ref_length(c, [len(t) for t in ref_token_lists])

        # modified precisions p_n
        p_list: List[float] = []
        for n in range(1, self.max_order + 1):
            match, total = self._modified_precision(hyp_tokens, ref_token_lists, n)
            if self.smoothing == "add1":
                match += 1.0
                total += 1.0
            if total == 0:
                pn = 0.0
            else:
                pn = match / total
            if pn == 0.0 and self.smoothing == "epsilon":
                pn = 1e-9
            p_list.append(pn)

        # brevity penalty
        if c == 0 and r == 0:
            bp = 1.0
        elif c == 0:
            bp = 0.0
        else:
            bp = 1.0 if c > r else math.exp(1.0 - float(r) / float(c))

        # geometric mean with logs
        from math import log, exp
        sum_wlogp = sum(w * log(p) for w, p in zip(self.weights, p_list))
        bleu = bp * (0.0 if any(p <= 0.0 for p in p_list) and self.smoothing == "none" else exp(sum_wlogp))

        out = {"bleu": float(bleu)}
        if self.include_components:
            out.update({
                "bleu_bp": bp,
                "bleu_ref_len": int(r),
                "bleu_hyp_len": int(c),
            })
            for i, p in enumerate(p_list, start=1):
                out[f"bleu_p{i}"] = float(p)
        return out

    # ------------------------ helpers ------------------------

    def _tokenize(self, s: str) -> List[str]:
        import re, string, unicodedata
        t = s or ""
        t = unicodedata.normalize("NFKC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = "".join(ch for ch in t if ch not in string.punctuation)
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t.split() if t else []

    def _ngram_counts(self, toks: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        from collections import Counter
        if n <= 0 or len(toks) < n:
            return Counter()
        grams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
        return Counter(grams)

    def _modified_precision(self, hyp_toks: List[str], ref_tok_lists: List[List[str]], n: int) -> Tuple[int, int]:
        """Return (match_count, total_hyp_ngrams) with clipping by max ref counts."""
        hyp_counts = self._ngram_counts(hyp_toks, n)
        if not hyp_counts:
            return (0, 0)
        # max reference counts
        from collections import Counter
        max_ref_counts: Counter = Counter()
        for ref in ref_tok_lists:
            rc = self._ngram_counts(ref, n)
            for g, c in rc.items():
                if c > max_ref_counts[g]:
                    max_ref_counts[g] = c
        # clipped matches
        match = 0
        for g, c in hyp_counts.items():
            match += min(c, max_ref_counts.get(g, 0))
        total = sum(hyp_counts.values())
        return (match, total)

    @staticmethod
    def _closest_ref_length(c: int, ref_lengths: List[int]) -> int:
        """Closest reference length to c; tie -> choose the shortest (Papineni et al.)."""
        if not ref_lengths:
            return 0
        # sort by (abs diff, length) and take first
        ref_lengths = sorted(ref_lengths, key=lambda rl: (abs(rl - c), rl))
        return ref_lengths[0]

    def _empty_out(self, val: Any) -> Dict[str, Any]:
        out = {"bleu": val}
        if self.include_components:
            out.update({
                "bleu_bp": val,
                "bleu_ref_len": val,
                "bleu_hyp_len": val,
            })
            for i in range(1, self.max_order + 1):
                out[f"bleu_p{i}"] = val
        return out

class BERTScore(Metric):
    """
    BERTScore between generated answer and reference(s).

    Notes:
      - Requires optional dependency: `pip install bert-score torch transformers`
      - If dependency is missing, returns None and an error hint.

    Sample:
      - answer_field:    sample[answer_field] -> hypothesis text (default "answer")
      - reference_field: sample[reference_field] -> gold text or list[str] (default "reference")

    Config:
      - model_type: e.g., "microsoft/deberta-xlarge-mnli", "bert-base-uncased" (default "bert-base-uncased")
      - lang: language code used by bert-score baselines (default None)
      - num_layers: which layer to use (default None -> library default)
      - rescale_with_baseline: bool, default False (set True if you want baseline-rescaled scores)
      - device: "cuda" | "cpu" | "mps" (default: auto)
      - batch_size: int (default 32)
      - idf: bool to use IDF weighting (default False)
      - agg: "max" (best over multiple references) or "avg" (average), default "max"
      - normalize_text: whether to normalize text like EM/F1 (default False)

    Returns:
      {
        "bertscore_f1": float or None,
        "bertscore_precision": float or None,
        "bertscore_recall": float or None,
        # If dependency missing:
        "bertscore_error": str (present only on error)
      }
    """
    name = "bertscore"
    _scorer_cache = None  # type: ignore

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        reference_field: str = "reference",
        model_type: str = "bert-base-uncased",
        lang: str | None = None,
        num_layers: int | None = None,
        rescale_with_baseline: bool = False,
        device: str | None = None,
        batch_size: int = 32,
        idf: bool = False,
        agg: str = "max",            # "max" | "avg"
        normalize_text: bool = False
    ) -> None:
        if agg not in {"max", "avg"}:
            raise ValueError('agg must be one of {"max","avg"}')

        # ★ 新增：若要 baseline rescale，必須指定 lang
        if rescale_with_baseline and (lang is None or not isinstance(lang, str) or not lang.strip()):
            raise ValueError('BERTScore: rescale_with_baseline=True 時必須指定 lang，例如 lang="en"。')

        self.answer_field = answer_field
        self.reference_field = reference_field
        self.model_type = model_type
        self.lang = lang
        self.num_layers = num_layers
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device
        self.batch_size = batch_size
        self.idf = idf
        self.agg = agg
        self.normalize_text = normalize_text

    # 下面 compute/_get_scorer_class/_get_or_build_scorer/_normalize 保持你原本的實作


    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        hyp = sample.get(self.answer_field)
        refs = sample.get(self.reference_field)

        # collect references
        ref_texts: List[str] = []
        if isinstance(refs, (list, tuple)):
            ref_texts = [r for r in refs if isinstance(r, str)]
        elif isinstance(refs, str):
            ref_texts = [refs]

        if not ref_texts:
            return {"bertscore_f1": None, "bertscore_precision": None, "bertscore_recall": None}

        hyp_text = hyp if isinstance(hyp, str) else ""
        if self.normalize_text:
            hyp_text = self._normalize(hyp_text)
            ref_texts = [self._normalize(r) for r in ref_texts]

        # lazy import & scorer build
        try:
            Scorer = self._get_scorer_class()
        except ImportError as e:
            return {
                "bertscore_f1": None,
                "bertscore_precision": None,
                "bertscore_recall": None,
                "bertscore_error": f"{e}. Install via: pip install bert-score torch transformers",
            }

        scorer = self._get_or_build_scorer(Scorer)

        # score against multiple refs: repeat hyp for each ref, then aggregate
        hyps = [hyp_text] * len(ref_texts)
        P, R, F = scorer.score(hyps, ref_texts)
        # tensors -> floats
        p_list = [float(x) for x in P]
        r_list = [float(x) for x in R]
        f_list = [float(x) for x in F]

        if self.agg == "avg":
            p = sum(p_list) / len(p_list)
            r = sum(r_list) / len(r_list)
            f1 = sum(f_list) / len(f_list)
        else:  # max
            idx = max(range(len(f_list)), key=lambda i: f_list[i])
            p, r, f1 = p_list[idx], r_list[idx], f_list[idx]

        return {"bertscore_f1": f1, "bertscore_precision": p, "bertscore_recall": r}

    # ------------------------ helpers ------------------------

    def _get_scorer_class(self):
        try:
            from bert_score import BERTScorer  # type: ignore
            return BERTScorer
        except Exception as e:
            raise ImportError(str(e))

    def _get_or_build_scorer(self, Scorer):
        # cache key by (model_type, lang, num_layers, rescale_with_baseline, device, batch_size, idf)
        key = (
            self.model_type,
            self.lang,
            self.num_layers,
            self.rescale_with_baseline,
            self.device,
            self.batch_size,
            self.idf,
        )
        cache = getattr(BERTScore, "_scorer_cache", None)
        if cache and cache.get("key") == key:
            return cache["scorer"]

        scorer = Scorer(
            model_type=self.model_type,
            lang=self.lang,
            num_layers=self.num_layers,
            rescale_with_baseline=self.rescale_with_baseline,
            device=self.device,
            batch_size=self.batch_size,
            idf=self.idf,
        )
        BERTScore._scorer_cache = {"key": key, "scorer": scorer}
        return scorer

    def _normalize(self, s: str) -> str:
        # Optional lightweight normalization; off by default for BERTScore
        import unicodedata, string
        t = unicodedata.normalize("NFKC", s or "")
        t = t.lower()
        t = "".join(ch for ch in t if ch not in string.punctuation)
        t = " ".join(t.split())
        return t

class AnswerRelevancySim(Metric):
    """
    Answer Relevancy (similarity-based).

    Two modes:
      - mode="reference": cosine(answer_emb, reference_emb[or text]) aggregated over refs.
      - mode="context"  : cosine(answer_emb, each top-k doc embedding) aggregated over docs.

    Embedding sources (prefer vectors; else use an optional embedder callable):
      - answer_embedding_field: sample[answer_embedding_field] -> List[float]
      - reference_embedding_field: sample[reference_embedding_field] -> List[float] or List[List[float]]
      - doc_embedding_field: sample[doc_embedding_field] -> Dict[doc_id, List[float]]
      - If vectors are missing and `embedder` is provided, we will compute embeddings from text:
          * answer_field (str)
          * reference_field (str or List[str])
          * context texts (if you store them and your embedder accepts them) – otherwise rely on doc embeddings.

    Aggregation:
      - agg: "max" or "mean" over refs/contexts (default "mean")

    Config:
      - mode: "reference" | "context"
      - k: cutoff for context mode (None => all retrieved)
      - retrieved_field: key of ranked list for context mode (default "retrieved_docs")
      - dedup: de-duplicate retrieved IDs before truncation (default True)
      - keys:
          * answer_embedding_field="answer_embedding"
          * reference_embedding_field="reference_embedding"
          * doc_embedding_field="doc_embedding"
          * answer_field="answer", reference_field="reference"
      - embedder: Optional callable that returns vector(s). It can accept:
          * str -> List[float], or
          * List[str] -> List[List[float]]
        If not provided and vectors missing => returns None with an error hint.
      - l2_normalize: whether to L2-normalize embeddings before cosine (default True)

    Returns:
      - mode="reference": { "answer_rel@reference": float in [-1,1] or None, ...maybe error key }
      - mode="context"  : { f"answer_rel@context@{k_str}": float in [-1,1] or None, ...maybe error key }
    """
    name = "answer_relevancy_sim"

    def __init__(
        self,
        *,
        mode: str = "reference",
        agg: str = "mean",                 # "mean" | "max"
        k: int | None = 5,
        retrieved_field: str = "retrieved_docs",
        dedup: bool = True,
        # fields for vectors
        answer_embedding_field: str = "answer_embedding",
        reference_embedding_field: str = "reference_embedding",
        doc_embedding_field: str = "doc_embedding",
        # fields for raw text (only used if embedder is provided)
        answer_field: str = "answer",
        reference_field: str = "reference",
        # embedding options
        embedder: Any | None = None,       # callable or None
        l2_normalize: bool = True,
    ) -> None:
        if mode not in {"reference", "context"}:
            raise ValueError('mode must be one of {"reference","context"}')
        if agg not in {"mean", "max"}:
            raise ValueError('agg must be one of {"mean","max"}')
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")

        self.mode = mode
        self.agg = agg
        self.k = k
        self.retrieved_field = retrieved_field
        self.dedup = dedup

        self.answer_embedding_field = answer_embedding_field
        self.reference_embedding_field = reference_embedding_field
        self.doc_embedding_field = doc_embedding_field

        self.answer_field = answer_field
        self.reference_field = reference_field

        self.embedder = embedder
        self.l2_normalize = l2_normalize

    # ------------------------------- main -------------------------------

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # get / build answer embedding
        a_vec = self._get_answer_vec(sample)
        if a_vec is None:
            return {self._key(): None, "answer_rel_error": "no answer embedding and no usable embedder"}

        if self.mode == "reference":
            ref_vecs = self._get_reference_vecs(sample)
            if not ref_vecs:
                return {self._key(): None, "answer_rel_error": "no reference embeddings/texts"}
            sims = [self._cos(a_vec, rv) for rv in ref_vecs]
        else:
            # context mode
            ctx_vecs = self._get_context_vecs(sample)
            if not ctx_vecs:
                return {self._key(): None, "answer_rel_error": "no context embeddings for retrieved docs"}
            sims = [self._cos(a_vec, cv) for cv in ctx_vecs]

        if not sims:
            return {self._key(): None}

        score = (sum(sims) / len(sims)) if self.agg == "mean" else max(sims)
        return {self._key(): float(score)}

    # ----------------------------- helpers ------------------------------

    def _key(self) -> str:
        if self.mode == "reference":
            return "answer_rel@reference"
        else:
            k_str = "all" if self.k is None else str(self.k)
            return f"answer_rel@context@{k_str}"

    def _cos(self, a: List[float] | None, b: List[float] | None) -> float:
        if a is None or b is None:
            return 0.0
        if self.l2_normalize:
            a = self._l2(a); b = self._l2(b)
        # reuse file-level cosine if present; else simple implementation
        try:
            return _cosine_sim(a, b)  # type: ignore  # uses your existing helper if defined
        except NameError:
            # fallback
            dot = sum(x*y for x, y in zip(a, b))
            na = sum(x*x for x in a) ** 0.5
            nb = sum(y*y for y in b) ** 0.5
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        import math
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x / n for x in v]

    # ---- fetch embeddings or build via embedder ----

    def _get_answer_vec(self, sample: Dict[str, Any]) -> List[float] | None:
        vec = sample.get(self.answer_embedding_field)
        if isinstance(vec, list) and vec:
            return vec
        # try to embed from text
        if self.embedder is not None:
            text = sample.get(self.answer_field)
            if isinstance(text, str):
                return self._embed_one(text)
        return None

    def _get_reference_vecs(self, sample: Dict[str, Any]) -> List[List[float]]:
        # priority 1: provided vectors
        ref_vec = sample.get(self.reference_embedding_field)
        out: List[List[float]] = []
        if isinstance(ref_vec, list) and ref_vec and all(isinstance(x, (float, int)) for x in ref_vec):
            out = [list(map(float, ref_vec))]
        elif isinstance(ref_vec, list) and ref_vec and all(isinstance(x, list) for x in ref_vec):
            out = [list(map(float, v)) for v in ref_vec if isinstance(v, list) and v]

        if out:
            return out

        # priority 2: embed from text if embedder provided
        if self.embedder is not None:
            gold = sample.get(self.reference_field)
            if isinstance(gold, str):
                v = self._embed_one(gold)
                return [v] if v is not None else []
            elif isinstance(gold, (list, tuple)):
                texts = [g for g in gold if isinstance(g, str)]
                vecs = self._embed_many(texts) if texts else []
                return [v for v in vecs if isinstance(v, list) and v]
        return []

    def _get_context_vecs(self, sample: Dict[str, Any]) -> List[List[float]]:
        # require doc embeddings
        emb_map = sample.get(self.doc_embedding_field, {}) or {}
        ranked = sample.get(self.retrieved_field, []) or []
        ranked = _as_ranked_list(ranked) if " _as_ranked_list" in globals() else ranked  # safe if helper exists
        ids = _ranked_to_ids(ranked, dedup=self.dedup) if "_ranked_to_ids" in globals() else [i[0] if isinstance(i, tuple) else i for i in ranked]
        if not ids:
            return []
        k_cut = len(ids) if self.k is None else self.k
        topk = ids[:k_cut]
        vecs = [emb_map.get(doc_id) for doc_id in topk]
        return [v for v in vecs if isinstance(v, list) and v]

    # ---- embedder adapters ----

    def _embed_one(self, text: str) -> List[float] | None:
        try:
            v = self.embedder(text)  # type: ignore
            if isinstance(v, list):
                return [float(x) for x in v]
            if hasattr(v, "__iter__"):
                return [float(x) for x in list(v)]
        except Exception:
            return None
        return None

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        try:
            v = self.embedder(texts)  # type: ignore
            # expect list of vectors
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [[float(x) for x in vec] for vec in v]
        except Exception:
            # fallback: map one-by-one
            outs: List[List[float]] = []
            for t in texts:
                vt = self._embed_one(t)
                if vt is not None:
                    outs.append(vt)
            return outs
        return []

class FaithfulnessNLI(Metric):
    """
    Faithfulness via NLI, and Hallucination Rate derived from it.

    For each answer sentence (claim), we check if it's supported by evidence using an NLI model.
    - premise  = evidence text (reference or retrieved contexts)
    - hypothesis = claim (one sentence from answer)

    Modes:
      - mode="reference": evidence from sample["reference"] or sample["reference_texts"] (str or List[str]).
      - mode="context"  : evidence from top-k retrieved docs via sample["doc_text"][doc_id].
                          sample["doc_text"][doc_id] can be str or List[str] (already-chunked).

    Output:
      {
        "faithfulness_nli": float in [0,1] or None,
        "hallucination_rate": float in [0,1] or None,
        # diagnostics (helpful but optional to read):
        "faith_num_claims": int, "faith_num_supported": int, "faith_num_contradicted": int,
        "faith_mean_entail": float or None, "faith_mean_contra": float or None,
        # on failure:
        "faith_error": str (only if something went wrong)
      }

    Dependencies (lazy import; only when used):
      pip install transformers torch

    Config:
      - mode: "reference" | "context" (default "context")
      - k: cutoff for context mode; None => use all retrieved (default 5)
      - retrieved_field: key for ranked docs (default "retrieved_docs")
      - doc_text_field: key for doc id -> text mapping (default "doc_text")
      - reference_field: key for reference text(s) (default "reference"); or "reference_texts"
      - sentence_splitter: optional callable str->List[str]; else use built-in regex
      - model_name: NLI classifier (default "roberta-large-mnli")
      - device: "cuda" | "mps" | "cpu" | None (auto)
      - batch_size: int, default 16 (pairs per batch)
      - threshold_entail: float in [0,1], default 0.5
      - threshold_contra: float in [0,1], default 0.5
      - pool: "max" or "mean" to aggregate claim vs multi-evidence (default "max")
      - max_length: tokenizer max_length (default 256)
      - dedup: de-duplicate retrieved IDs before truncation (default True)
    """
    name = "faithfulness_nli"

    _nli_cache = None  # {"key": (...), "tok": tokenizer, "mdl": model, "device": torch.device}

    def __init__(
        self,
        *,
        mode: str = "context",
        k: int | None = 5,
        retrieved_field: str = "retrieved_docs",
        doc_text_field: str = "doc_text",
        reference_field: str = "reference",
        sentence_splitter: Any | None = None,
        model_name: str = "roberta-large-mnli",
        device: str | None = None,
        batch_size: int = 16,
        threshold_entail: float = 0.5,
        threshold_contra: float = 0.5,
        pool: str = "max",
        max_length: int = 256,
        dedup: bool = True,
    ) -> None:
        if mode not in {"reference", "context"}:
            raise ValueError('mode must be one of {"reference","context"}')
        if pool not in {"max", "mean"}:
            raise ValueError('pool must be one of {"max","mean"}')
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.mode = mode
        self.k = k
        self.retrieved_field = retrieved_field
        self.doc_text_field = doc_text_field
        self.reference_field = reference_field
        self.sentence_splitter = sentence_splitter
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.threshold_entail = float(threshold_entail)
        self.threshold_contra = float(threshold_contra)
        self.pool = pool
        self.max_length = int(max_length)
        self.dedup = dedup

    # ------------------------------ main ------------------------------

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        answer = sample.get("answer") or ""
        claims = self._split_sentences(answer)
        if not claims:
            # no claims → undefined; choose to return None to be ignored by summarize
            return {
                "faithfulness_nli": None,
                "hallucination_rate": None,
                "faith_num_claims": 0,
                "faith_num_supported": 0,
                "faith_num_contradicted": 0,
                "faith_mean_entail": None,
                "faith_mean_contra": None,
            }

        evidences = self._collect_evidences(sample)
        if not evidences:
            return {
                "faithfulness_nli": None,
                "hallucination_rate": None,
                "faith_num_claims": len(claims),
                "faith_num_supported": 0,
                "faith_num_contradicted": 0,
                "faith_mean_entail": None,
                "faith_mean_contra": None,
                "faith_error": "no evidences available",
            }

        try:
            tok, mdl, dev, idx_entail, idx_contra = self._get_or_build_nli()
        except Exception as e:
            return {
                "faithfulness_nli": None,
                "hallucination_rate": None,
                "faith_num_claims": len(claims),
                "faith_num_supported": 0,
                "faith_num_contradicted": 0,
                "faith_mean_entail": None,
                "faith_mean_contra": None,
                "faith_error": f"{e}. Install transformers/torch or check model.",
            }

        # Build all (claim, evidence) pairs
        pairs: list[tuple[str, str]] = []
        claim_index: list[int] = []  # map from pair -> claim id
        for ci, c in enumerate(claims):
            for ev in evidences:
                pairs.append((ev, c))  # premise=ev, hypothesis=c
                claim_index.append(ci)

        # Batched inference
        import math
        from torch.nn.functional import softmax  # type: ignore
        import torch  # type: ignore

        entail_scores = [0.0] * len(claims)
        contra_scores = [0.0] * len(claims)

        # Aggregate per claim with max or mean across evidences
        agg_init = -1.0 if self.pool == "max" else 0.0
        e_collect = [[agg_init, 0] for _ in claims]  # [agg_val, count]
        c_collect = [[agg_init, 0] for _ in claims]

        bs = max(1, int(self.batch_size))
        for start in range(0, len(pairs), bs):
            batch = pairs[start:start + bs]
            enc = tok(
                [p for p, h in batch],
                [h for p, h in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(**enc).logits  # [B, num_labels]
                probs = softmax(logits, dim=-1)  # [B, L]
            pe = probs[:, idx_entail].detach().cpu().tolist()
            pc = probs[:, idx_contra].detach().cpu().tolist()

            for i, (e, c) in enumerate(zip(pe, pc)):
                ci = claim_index[start + i]
                if self.pool == "max":
                    # max over evidences
                    e_collect[ci][0] = max(e_collect[ci][0], float(e))
                    c_collect[ci][0] = max(c_collect[ci][0], float(c))
                else:
                    # mean over evidences
                    e_collect[ci][0] += float(e); e_collect[ci][1] += 1
                    c_collect[ci][0] += float(c); c_collect[ci][1] += 1

        # finalize per-claim scores
        for ci in range(len(claims)):
            if self.pool == "mean":
                if e_collect[ci][1] > 0:
                    entail_scores[ci] = e_collect[ci][0] / e_collect[ci][1]
                    contra_scores[ci] = c_collect[ci][0] / c_collect[ci][1]
                else:
                    entail_scores[ci] = 0.0
                    contra_scores[ci] = 0.0
            else:
                entail_scores[ci] = max(0.0, e_collect[ci][0])
                contra_scores[ci] = max(0.0, c_collect[ci][0])

        # Decide supported / contradicted
        supported = sum(1 for e in entail_scores if e >= self.threshold_entail)
        contrad  = sum(1 for c in contra_scores if c >= self.threshold_contra)
        num_claims = len(claims)
        faith = supported / num_claims if num_claims > 0 else None
        halluc = (1.0 - faith) if faith is not None else None

        mean_entail = sum(entail_scores)/num_claims if num_claims>0 else None
        mean_contra = sum(contra_scores)/num_claims if num_claims>0 else None

        return {
            "faithfulness_nli": faith,
            "hallucination_rate": halluc,
            "faith_num_claims": num_claims,
            "faith_num_supported": supported,
            "faith_num_contradicted": contrad,
            "faith_mean_entail": mean_entail,
            "faith_mean_contra": mean_contra,
        }

    # ------------------------------ helpers ------------------------------

    def _collect_evidences(self, sample: Dict[str, Any]) -> list[str]:
        evs: list[str] = []
        if self.mode == "reference":
            # reference / reference_texts
            ref = sample.get(self.reference_field)
            if isinstance(ref, str) and ref.strip():
                evs.append(ref.strip())
            elif isinstance(ref, (list, tuple)):
                evs.extend([x for x in ref if isinstance(x, str) and x.strip()])
            # optional alias field
            ref2 = sample.get("reference_texts")
            if isinstance(ref2, str) and ref2.strip():
                evs.append(ref2.strip())
            elif isinstance(ref2, (list, tuple)):
                evs.extend([x for x in ref2 if isinstance(x, str) and x.strip()])
            return evs

        # context mode
        ranked = sample.get(self.retrieved_field, []) or []
        try:
            ranked = _as_ranked_list(ranked)  # uses your existing helper
            ids = _ranked_to_ids(ranked, dedup=self.dedup)
        except Exception:
            # light fallback
            ids = [it[0] if isinstance(it, tuple) else it for it in ranked]

        if not ids:
            return []

        k_cut = len(ids) if self.k is None else self.k
        topk = ids[:k_cut]
        text_map = sample.get(self.doc_text_field, {}) or {}
        for did in topk:
            t = text_map.get(did)
            if isinstance(t, str) and t.strip():
                evs.append(t.strip())
            elif isinstance(t, (list, tuple)):
                # already chunked list of passages
                evs.extend([x for x in t if isinstance(x, str) and x.strip()])
        return evs

    def _split_sentences(self, text: str) -> list[str]:
        if self.sentence_splitter:
            try:
                out = self.sentence_splitter(text)
                if isinstance(out, list) and out:
                    return [s.strip() for s in out if isinstance(s, str) and s.strip()]
            except Exception:
                pass
        import re
        # simple multilingual-ish splitter: ., !, ?, ;, newlines, and common CJK terminators
        parts = re.split(r"[。！？!?；;。\n]+|(?<=[\.!?;])\s+", text or "")
        return [p.strip() for p in parts if isinstance(p, str) and p.strip()]

    def _get_or_build_nli(self):
        # lazy import
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            raise ImportError(str(e))

        # device
        dev = None
        if self.device:
            dev = torch.device(self.device)
        else:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")

        key = (self.model_name, str(dev))
        cache = getattr(FaithfulnessNLI, "_nli_cache", None)
        if cache and cache.get("key") == key:
            tok, mdl, dev_cached, idx_entail, idx_contra = cache["tok"], cache["mdl"], cache["dev"], cache["idx_e"], cache["idx_c"]
            return tok, mdl, dev_cached, idx_entail, idx_contra

        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        mdl.to(dev)
        mdl.eval()

        # map label indices
        id2label = {int(k): str(v).lower() for k, v in getattr(mdl.config, "id2label", {}).items()}
        idx_entail = self._find_label_index(id2label, "entail")
        idx_contra = self._find_label_index(id2label, "contrad")
        if idx_entail is None or idx_contra is None:
            # fallback to MNLI common ordering: 0=contradiction, 1=neutral, 2=entailment
            idx_contra = 0 if idx_contra is None else idx_contra
            idx_entail = 2 if idx_entail is None else idx_entail

        FaithfulnessNLI._nli_cache = {
            "key": key, "tok": tok, "mdl": mdl, "dev": dev, "idx_e": idx_entail, "idx_c": idx_contra
        }
        return tok, mdl, dev, idx_entail, idx_contra

    @staticmethod
    def _find_label_index(id2label: Dict[int, str], keyword: str) -> int | None:
        for i, name in id2label.items():
            if keyword in name:
                return i
        return None

class ContextPrecisionAtK(Metric):
    """
    Context Precision@k (label-based).

    Measures precision among top-k retrieved contexts using provided relevance labels.

    Labels:
      - Preferred graded: sample[rel_field] = {doc_id: rel}, rel>0 => relevant
      - Fallback binary:  sample[binary_fallback_field] = iterable of relevant IDs

    Config:
      - k: int | None (None => use full retrieved length)
      - rel_field: str = "relevance"
      - binary_fallback_field: str = "reference_docs"
      - retrieved_field: str = "retrieved_docs"
      - dedup: bool = True

    Returns:
      { f"context_precision@{k_str}": float in [0,1] or None }.
      None when no relevance labels are available.
    """
    name = "context_precision_at_k"

    def __init__(
        self,
        k: int | None = 5,
        *,
        rel_field: str = "relevance",
        binary_fallback_field: str = "reference_docs",
        retrieved_field: str = "retrieved_docs",
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.k = k
        self.rel_field = rel_field
        self.binary_fallback_field = binary_fallback_field
        self.retrieved_field = retrieved_field
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        relevant = self._relevant_set(sample)
        if relevant is None:
            return {self._key(): None}

        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        if not retrieved_ids:
            return {self._key(): None}

        k_cut = len(retrieved_ids) if self.k is None else self.k
        topk = retrieved_ids[:k_cut]
        hits = sum(1 for d in topk if d in relevant)
        prec = hits / max(1, len(topk))
        return {self._key(): prec}

    def _relevant_set(self, sample: Dict[str, Any]) -> Set[Doc] | None:
        graded = sample.get(self.rel_field)
        if isinstance(graded, dict) and graded:
            rel = {k for k, v in graded.items() if self._is_pos(v)}
            if rel:
                return rel
        gold_ids = _as_id_set(sample.get(self.binary_fallback_field, []))
        return gold_ids if gold_ids else None

    @staticmethod
    def _is_pos(v: Any) -> bool:
        try:
            return float(v) > 0.0
        except Exception:
            return False

    def _key(self) -> str:
        return f"context_precision@{'all' if self.k is None else self.k}"


class ContextPrecisionSim(Metric):
    """
    Context Precision@k (similarity-based).

    Measures precision among top-k retrieved contexts using cosine(question, doc).
    A doc is relevant if cosine >= threshold.

    Inputs:
      - question_embedding_field: sample[question_embedding_field] -> List[float]
      - doc_embedding_field:      sample[doc_embedding_field] -> Dict[doc_id, List[float]]
      - If embeddings missing and `embedder` is provided, will try to embed from text:
          * question_field (str)
          * doc_text_field: Dict[doc_id, str or List[str]] -> we embed the doc text (str) or join list.

    Config:
      - k: int | None (None => use full retrieved length)
      - threshold: float in [-1,1], default 0.3
      - agg_doc: "mean" | "max"  (when doc_text provides List[str], how to aggregate multiple chunks; default "max")
      - retrieved_field: str = "retrieved_docs"
      - question_field: str = "question"
      - doc_text_field: str = "doc_text"
      - question_embedding_field: str = "question_embedding"
      - doc_embedding_field: str = "doc_embedding"
      - embedder: Optional[callable] converting str or List[str] -> vector(s)
      - l2_normalize: bool = True
      - dedup: bool = True

    Returns:
      { f"context_precision_sim@{k_str}": float in [0,1] or None }.
      None when embeddings/texts are insufficient.
    """
    name = "context_precision_sim"

    def __init__(
        self,
        k: int | None = 5,
        *,
        threshold: float = 0.3,
        agg_doc: str = "max",
        retrieved_field: str = "retrieved_docs",
        question_field: str = "question",
        doc_text_field: str = "doc_text",
        question_embedding_field: str = "question_embedding",
        doc_embedding_field: str = "doc_embedding",
        embedder: Any | None = None,
        l2_normalize: bool = True,
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        if agg_doc not in {"mean", "max"}:
            raise ValueError('agg_doc must be one of {"mean","max"}')
        self.k = k
        self.threshold = float(threshold)
        self.agg_doc = agg_doc
        self.retrieved_field = retrieved_field
        self.question_field = question_field
        self.doc_text_field = doc_text_field
        self.question_embedding_field = question_embedding_field
        self.doc_embedding_field = doc_embedding_field
        self.embedder = embedder
        self.l2_normalize = l2_normalize
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        q = self._get_question_vec(sample)
        if q is None:
            return {self._key(): None, "context_prec_error": "no question embedding and no usable embedder"}

        # 拿 top-k doc embeddings（必要時用 embedder 對文本向量化）
        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        doc_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        if not doc_ids:
            return {self._key(): None}
        k_cut = len(doc_ids) if self.k is None else self.k
        doc_ids = doc_ids[:k_cut]

        doc_emb_map = sample.get(self.doc_embedding_field, {}) or {}
        sims: List[float] = []
        for did in doc_ids:
            vec = doc_emb_map.get(did)
            if isinstance(vec, list) and vec:
                sims.append(self._cos(q, vec))
                continue

            # 沒有 doc 向量：嘗試從文本計算（需要 embedder）
            if self.embedder is None:
                continue
            doc_text = sample.get(self.doc_text_field, {}).get(did) if isinstance(sample.get(self.doc_text_field), dict) else None
            if isinstance(doc_text, str):
                v = self._embed_one(doc_text)
                if v is not None:
                    sims.append(self._cos(q, v))
            elif isinstance(doc_text, (list, tuple)):
                # 多段文本 → 先各自嵌入，再依 agg_doc 聚合成單一分數
                segs = [t for t in doc_text if isinstance(t, str) and t.strip()]
                vecs = self._embed_many(segs) if segs else []
                if vecs:
                    seg_sims = [self._cos(q, v) for v in vecs]
                    agg = max(seg_sims) if self.agg_doc == "max" else (sum(seg_sims)/len(seg_sims))
                    sims.append(agg)

        if not sims:
            return {self._key(): None, "context_prec_error": "no usable doc embeddings/texts"}

        hits = sum(1 for s in sims if s >= self.threshold)
        prec = hits / max(1, len(sims))
        return {self._key(): float(prec)}

    # ------------------------ helpers ------------------------

    def _key(self) -> str:
        return f"context_precision_sim@{'all' if self.k is None else self.k}"

    def _cos(self, a: List[float], b: List[float]) -> float:
        if self.l2_normalize:
            a = self._l2(a); b = self._l2(b)
        try:
            return _cosine_sim(a, b)  # 若前面已定義共用 cosine
        except NameError:
            dot = sum(x*y for x, y in zip(a, b))
            na = sum(x*x for x in a) ** 0.5
            nb = sum(y*y for y in b) ** 0.5
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        import math
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_question_vec(self, sample: Dict[str, Any]) -> List[float] | None:
        vec = sample.get(self.question_embedding_field)
        if isinstance(vec, list) and vec:
            return vec
        if self.embedder is not None:
            text = sample.get(self.question_field)
            if isinstance(text, str):
                return self._embed_one(text)
        return None

    # ---- embedder adapters ----
    def _embed_one(self, text: str) -> List[float] | None:
        try:
            v = self.embedder(text)  # type: ignore
            if isinstance(v, list):
                return [float(x) for x in v]
            if hasattr(v, "__iter__"):
                return [float(x) for x in list(v)]
        except Exception:
            return None
        return None

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        try:
            v = self.embedder(texts)  # type: ignore
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [[float(x) for x in vec] for vec in v]
        except Exception:
            outs: List[List[float]] = []
            for t in texts:
                vt = self._embed_one(t)
                if vt is not None:
                    outs.append(vt)
            return outs
        return []

class ContextRecallAtK(Metric):
    """
    Context Recall@k (label-based).

    Uses provided relevance labels to compute recall among top-k retrieved docs.

    Labels:
      - Preferred graded: sample[rel_field] = {doc_id: rel}, rel>0 => relevant
      - Fallback binary:  sample[binary_fallback_field] = iterable of relevant IDs

    Config:
      - k: int | None (None => use full retrieved length)
      - rel_field: str = "relevance"
      - binary_fallback_field: str = "reference_docs"
      - retrieved_field: str = "retrieved_docs"
      - dedup: bool = True

    Returns:
      { f"context_recall@{k_str}": float in [0,1] or None }.
      None when no relevance labels are available.
    """
    name = "context_recall_at_k"

    def __init__(
        self,
        k: int | None = 5,
        *,
        rel_field: str = "relevance",
        binary_fallback_field: str = "reference_docs",
        retrieved_field: str = "retrieved_docs",
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.k = k
        self.rel_field = rel_field
        self.binary_fallback_field = binary_fallback_field
        self.retrieved_field = retrieved_field
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        relevant = self._relevant_set(sample)
        if not relevant:
            return {self._key(): None}

        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        if not retrieved_ids:
            return {self._key(): None}

        k_cut = len(retrieved_ids) if self.k is None else self.k
        topk = retrieved_ids[:k_cut]

        hits = sum(1 for d in topk if d in relevant)
        rec = hits / max(1, len(relevant))
        return {self._key(): rec}

    def _relevant_set(self, sample: Dict[str, Any]) -> Set[Doc] | None:
        graded = sample.get(self.rel_field)
        if isinstance(graded, dict) and graded:
            rel = {k for k, v in graded.items() if self._is_pos(v)}
            if rel:
                return rel
        gold_ids = _as_id_set(sample.get(self.binary_fallback_field, []))
        return gold_ids if gold_ids else None

    @staticmethod
    def _is_pos(v: Any) -> bool:
        try:
            return float(v) > 0.0
        except Exception:
            return False

    def _key(self) -> str:
        return f"context_recall@{'all' if self.k is None else self.k}"


class ContextRecallSim(Metric):
    """
    Context Recall@k (similarity-based).

    Define relevant docs via cosine(question, doc) >= threshold over the WHOLE retrieved list
    (after de-dup). Recall@k is the fraction of those 'relevant' docs captured in Top-k.

    Inputs:
      - question_embedding_field: sample[question_embedding_field] -> List[float]
      - doc_embedding_field:      sample[doc_embedding_field] -> Dict[doc_id, List[float]]
      - If embeddings missing and `embedder` is provided, will try to embed from text:
          * question_field (str)
          * doc_text_field: Dict[doc_id, str or List[str]] -> we embed the doc text (str) or join/aggregate list.

    Config:
      - k: int | None (None => use full retrieved length)
      - threshold: float in [-1,1], default 0.3
      - agg_doc: "mean" | "max" for aggregating multiple chunks per doc (default "max")
      - retrieved_field: str = "retrieved_docs"
      - question_field: str = "question"
      - doc_text_field: str = "doc_text"
      - question_embedding_field: str = "question_embedding"
      - doc_embedding_field: str = "doc_embedding"
      - embedder: Optional[callable] converting str or List[str] -> vector(s)
      - l2_normalize: bool = True
      - dedup: bool = True

    Returns:
      { f"context_recall_sim@{k_str}": float in [0,1] or None }.
      None when no doc meets the threshold (denominator=0) or embeddings/texts insufficient.
    """
    name = "context_recall_sim"

    def __init__(
        self,
        k: int | None = 5,
        *,
        threshold: float = 0.3,
        agg_doc: str = "max",
        retrieved_field: str = "retrieved_docs",
        question_field: str = "question",
        doc_text_field: str = "doc_text",
        question_embedding_field: str = "question_embedding",
        doc_embedding_field: str = "doc_embedding",
        embedder: Any | None = None,
        l2_normalize: bool = True,
        dedup: bool = True,
    ) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        if agg_doc not in {"mean", "max"}:
            raise ValueError('agg_doc must be one of {"mean","max"}')
        self.k = k
        self.threshold = float(threshold)
        self.agg_doc = agg_doc
        self.retrieved_field = retrieved_field
        self.question_field = question_field
        self.doc_text_field = doc_text_field
        self.question_embedding_field = question_embedding_field
        self.doc_embedding_field = doc_embedding_field
        self.embedder = embedder
        self.l2_normalize = l2_normalize
        self.dedup = dedup

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # question vector
        q = self._get_question_vec(sample)
        if q is None:
            return {self._key(): None, "context_recall_error": "no question embedding and no usable embedder"}

        # full retrieved ids (for defining relevant set R)
        ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
        all_ids = _ranked_to_ids(ranked, dedup=self.dedup)
        if not all_ids:
            return {self._key(): None}

        doc_emb_map = sample.get(self.doc_embedding_field, {}) or {}
        # compute similarity for every deduped retrieved doc
        sims_all: Dict[Doc, float] = {}
        for did in all_ids:
            vec = doc_emb_map.get(did)
            if isinstance(vec, list) and vec:
                sims_all[did] = self._cos(q, vec)
                continue
            # try embed from text if available
            if self.embedder is not None:
                text_map = sample.get(self.doc_text_field, {}) or {}
                t = text_map.get(did)
                if isinstance(t, str):
                    v = self._embed_one(t)
                    if v is not None:
                        sims_all[did] = self._cos(q, v)
                elif isinstance(t, (list, tuple)):
                    segs = [s for s in t if isinstance(s, str) and s.strip()]
                    vecs = self._embed_many(segs) if segs else []
                    if vecs:
                        seg_sims = [self._cos(q, v) for v in vecs]
                        sims_all[did] = max(seg_sims) if self.agg_doc == "max" else (sum(seg_sims)/len(seg_sims))

        if not sims_all:
            return {self._key(): None, "context_recall_error": "no usable doc embeddings/texts"}

        # define 'relevant' docs by threshold over the WHOLE retrieved list
        relevant_ids = {did for did, s in sims_all.items() if s >= self.threshold}
        if not relevant_ids:
            return {self._key(): None}  # denominator 0 -> undefined

        # take top-k and compute recall@k
        k_cut = len(all_ids) if self.k is None else self.k
        topk = all_ids[:k_cut]
        hits = sum(1 for d in topk if d in relevant_ids)
        rec = hits / max(1, len(relevant_ids))
        return {self._key(): float(rec)}

    # ------------------------ helpers ------------------------

    def _key(self) -> str:
        return f"context_recall_sim@{'all' if self.k is None else self.k}"

    def _cos(self, a: List[float], b: List[float]) -> float:
        if self.l2_normalize:
            a = self._l2(a); b = self._l2(b)
        try:
            return _cosine_sim(a, b)
        except NameError:
            dot = sum(x*y for x, y in zip(a, b))
            na = sum(x*x for x in a) ** 0.5
            nb = sum(y*y for y in b) ** 0.5
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

    @staticmethod
    def _l2(v: List[float]) -> List[float]:
        import math
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_question_vec(self, sample: Dict[str, Any]) -> List[float] | None:
        vec = sample.get(self.question_embedding_field)
        if isinstance(vec, list) and vec:
            return vec
        if self.embedder is not None:
            text = sample.get(self.question_field)
            if isinstance(text, str):
                return self._embed_one(text)
        return None

    def _embed_one(self, text: str) -> List[float] | None:
        try:
            v = self.embedder(text)  # type: ignore
            if isinstance(v, list):
                return [float(x) for x in v]
            if hasattr(v, "__iter__"):
                return [float(x) for x in list(v)]
        except Exception:
            return None
        return None

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        try:
            v = self.embedder(texts)  # type: ignore
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [[float(x) for x in vec] for vec in v]
        except Exception:
            outs: List[List[float]] = []
            for t in texts:
                vt = self._embed_one(t)
                if vt is not None:
                    outs.append(vt)
            return outs
        return []

class CitationAccuracy(Metric):
    """
    Citation Accuracy on inline citations in the model answer.

    Goal:
      Measure how many cited document IDs in the answer actually point to the intended target set:
        - against="retrieved": Top-k retrieved IDs (default)
        - against="gold":      gold relevant set (graded rel>0 or binary reference_docs)

    Sample fields:
      - answer: str, containing inline citations like "[1]", "[1,2]", "[d3]" (comma/space separated)
      - citation_map (optional): Dict[str, Doc] mapping numeric markers to doc IDs, e.g. {"1":"d2","2":"d5"}
      - retrieved_docs: ranked list of retrieved items [id] or [(id,score)]  (for against="retrieved")
      - relevance / reference_docs: gold labels (for against="gold")

      Optional universe hints (用於辨識直接以 doc_id 引用時的可用全集，非必需):
      - doc_text: Dict[doc_id, str or List[str]]  (keys provide doc_id universe)
      - doc_embedding: Dict[doc_id, List[float]]  (keys provide doc_id universe)

    Config:
      - against: "retrieved" | "gold" (default "retrieved")
      - k: int | None (cutoff for Top-k when against="retrieved"; None => full list)
      - retrieved_field: str = "retrieved_docs"
      - rel_field: str = "relevance"
      - binary_fallback_field: str = "reference_docs"
      - citation_map_field: str = "citation_map"
      - dedup: bool = True
      - allow_free_ids: bool = True
          If True, citations like "[d7]" are accepted even without citation_map, as long as 'd7'
          appears in any known universe (retrieved ids, doc_text keys, doc_embedding keys).

    Returns:
      {
        f"citation_acc[{against}]@{k_str}": float in [0,1] or None,
        "citation_num": int,            # total raw citation markers found
        "citation_resolved": int,       # markers resolved to doc IDs
        "citation_in_target": int       # resolved & inside target set
      }
      If nothing to evaluate (no citations resolved or no target), metric returns None.
    """
    name = "citation_accuracy"

    def __init__(
        self,
        *,
        against: str = "retrieved",
        k: int | None = 5,
        retrieved_field: str = "retrieved_docs",
        rel_field: str = "relevance",
        binary_fallback_field: str = "reference_docs",
        citation_map_field: str = "citation_map",
        dedup: bool = True,
        allow_free_ids: bool = True,
    ) -> None:
        if against not in {"retrieved", "gold"}:
            raise ValueError('against must be one of {"retrieved","gold"}')
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.against = against
        self.k = k
        self.retrieved_field = retrieved_field
        self.rel_field = rel_field
        self.binary_fallback_field = binary_fallback_field
        self.citation_map_field = citation_map_field
        self.dedup = dedup
        self.allow_free_ids = allow_free_ids

    # ------------------------------- main -------------------------------

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        answer = sample.get("answer") or ""
        raw_tokens = self._extract_citation_tokens(answer)
        total_raw = len(raw_tokens)

        # Universe of known doc IDs (for resolving free-form IDs like [d3])
        universe = self._collect_universe(sample)

        # Resolve tokens -> doc_ids
        c_map = sample.get(self.citation_map_field, {}) or {}
        resolved: List[Doc] = []
        for tok in raw_tokens:
            # 1) citation_map wins if available (numeric or custom key)
            if isinstance(c_map, dict) and tok in c_map:
                resolved.append(c_map[tok])
                continue
            # 2) allow free-form IDs like [d3], [DOC_12] if present in universe
            if self.allow_free_ids and tok in universe:
                resolved.append(tok)
                continue
            # 3) also try numeric-in-string fallback like leading/trailing spaces
            if self.allow_free_ids and tok.strip() in universe:
                resolved.append(tok.strip())

        num_resolved = len(resolved)
        if num_resolved == 0:
            return {self._key(): None, "citation_num": total_raw, "citation_resolved": 0, "citation_in_target": 0}

        # Build target set
        target: Set[Doc] | None = None
        if self.against == "retrieved":
            ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
            retrieved_ids = _ranked_to_ids(ranked, dedup=self.dedup)
            if retrieved_ids:
                k_cut = len(retrieved_ids) if self.k is None else self.k
                target = set(retrieved_ids[:k_cut])
        else:  # gold
            target = self._gold_set(sample)

        if not target:
            return {self._key(): None, "citation_num": total_raw, "citation_resolved": num_resolved, "citation_in_target": 0}

        in_target = sum(1 for d in resolved if d in target)
        prec = in_target / max(1, num_resolved)

        return {
            self._key(): float(prec),
            "citation_num": total_raw,
            "citation_resolved": num_resolved,
            "citation_in_target": in_target,
        }

    # ----------------------------- helpers ------------------------------

    def _key(self) -> str:
        k_str = "all" if (self.against == "retrieved" and self.k is None) else (str(self.k) if self.against == "retrieved" else "gold")
        return f"citation_acc[{self.against}]@{k_str}"

    def _extract_citation_tokens(self, text: str) -> List[str]:
        """
        Extract tokens inside square brackets that look like citations.
        Examples matched: "[1]", "[1,2]", "[d3]", "[ d4 ; d5 ]"
        We split by comma/semicolon/space and keep alnum/underscore/dash tokens.
        """
        import re
        tokens: List[str] = []
        # find bracketed segments
        for seg in re.findall(r"\[([^\[\]]+)\]", text or ""):
            # split by commas/semicolons/spaces
            parts = re.split(r"[,\s;]+", seg)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # keep only reasonable tokens (alnum + _ -)
                if re.fullmatch(r"[A-Za-z0-9_\-]+", p):
                    tokens.append(p)
        return tokens

    def _collect_universe(self, sample: Dict[str, Any]) -> Set[Doc]:
        """
        Build a universe of known doc IDs from retrieved list and optional maps (doc_text/doc_embedding).
        Helps resolve free-form IDs like [d3] without a citation_map.
        """
        uni: Set[Doc] = set()
        # retrieved ids
        try:
            ranked = _as_ranked_list(sample.get(self.retrieved_field, []))
            uni.update(_ranked_to_ids(ranked, dedup=True))
        except Exception:
            pass
        # doc_text keys
        dt = sample.get("doc_text", {})
        if isinstance(dt, dict):
            uni.update(dt.keys())
        # doc_embedding keys
        de = sample.get("doc_embedding", {})
        if isinstance(de, dict):
            uni.update(de.keys())
        # gold ids (for against="gold")
        uni.update(self._gold_set(sample) or set())
        return uni

    def _gold_set(self, sample: Dict[str, Any]) -> Set[Doc] | None:
        graded = sample.get(self.rel_field)
        if isinstance(graded, dict) and graded:
            pos = {k for k, v in graded.items() if self._is_pos(v)}
            if pos:
                return pos
        gold_ids = _as_id_set(sample.get(self.binary_fallback_field, []))
        return gold_ids if gold_ids else None

    @staticmethod
    def _is_pos(v: Any) -> bool:
        try:
            return float(v) > 0.0
        except Exception:
            return False

class EvidenceDensity(Metric):
    """
    Evidence Density via lexical overlap (n-gram coverage).

    Definition:
      Build an evidence vocabulary from the evidence texts, then compute the fraction of
      answer n-grams that appear in the evidence vocabulary.

      density = matched_answer_ngrams / total_answer_ngrams

    Evidence modes:
      - mode="reference": use sample["reference"] or ["reference_texts"] as evidence (str or List[str])
      - mode="context"  : use top-k of sample["retrieved_docs"], and take texts from sample["doc_text"][doc_id]
                          where values can be str or List[str] (pre-chunked passages)

    Config:
      - mode: "reference" | "context" (default "context")
      - n: n-gram size (int >= 1, default 1)
      - k: cutoff for context mode (None => all retrieved)
      - retrieved_field: str = "retrieved_docs"
      - doc_text_field: str = "doc_text"
      - reference_field: str = "reference"  (also accepts "reference_texts")
      - dedup: de-duplicate retrieved IDs before truncation (default True)

      Normalization of text (applied to both answer and evidence):
      - lowercase (bool, default True)
      - strip_punct (bool, default True)
      - collapse_whitespace (bool, default True)
      - remove_articles (bool, default False)  # English-only
      - extra_strip_chars (str, default "")

    Returns:
      {
        key: float in [0,1] or None,
        "evid_density_matched": int,  # matched answer n-grams
        "evid_density_total": int     # total answer n-grams
      }
      If no evidence texts or no answer n-grams (e.g., empty answer), returns None and counts.
    """
    name = "evidence_density"

    def __init__(
        self,
        *,
        mode: str = "context",
        n: int = 1,
        k: int | None = 5,
        retrieved_field: str = "retrieved_docs",
        doc_text_field: str = "doc_text",
        reference_field: str = "reference",
        dedup: bool = True,
        # normalization
        lowercase: bool = True,
        strip_punct: bool = True,
        collapse_whitespace: bool = True,
        remove_articles: bool = False,
        extra_strip_chars: str = "",
    ) -> None:
        if mode not in {"reference", "context"}:
            raise ValueError('mode must be one of {"reference","context"}')
        if n <= 0:
            raise ValueError("n must be >= 1")
        if k is not None and k <= 0:
            raise ValueError("k must be positive or None.")
        self.mode = mode
        self.n = int(n)
        self.k = k
        self.retrieved_field = retrieved_field
        self.doc_text_field = doc_text_field
        self.reference_field = reference_field
        self.dedup = dedup

        self.lowercase = lowercase
        self.strip_punct = strip_punct
        self.collapse_whitespace = collapse_whitespace
        self.remove_articles = remove_articles
        self.extra_strip_chars = extra_strip_chars

    # ------------------------------ main ------------------------------

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        answer = sample.get("answer") or ""
        ans_tokens = self._tokenize(answer)
        ans_ngrams = self._ngrams(ans_tokens, self.n)
        total = len(ans_ngrams)

        if total == 0:
            return {self._key(): None, "evid_density_matched": 0, "evid_density_total": 0}

        evid_texts = self._collect_evidence_texts(sample)
        if not evid_texts:
            return {self._key(): None, "evid_density_matched": 0, "evid_density_total": total}

        # build evidence n-gram set
        evid_set: set[tuple[str, ...]] = set()
        for t in evid_texts:
            toks = self._tokenize(t)
            evid_set.update(self._ngrams(toks, self.n))

        if not evid_set:
            return {self._key(): None, "evid_density_matched": 0, "evid_density_total": total}

        matched = sum(1 for g in ans_ngrams if g in evid_set)
        density = matched / max(1, total)
        return {self._key(): float(density), "evid_density_matched": matched, "evid_density_total": total}

    # ------------------------------ helpers ------------------------------

    def _key(self) -> str:
        base = f"evidence_density@{self.mode}"
        if self.mode == "context":
            base += f"@{'all' if self.k is None else self.k}"
        if self.n != 1:
            base += f"[n{self.n}]"
        return base

    def _collect_evidence_texts(self, sample: Dict[str, Any]) -> list[str]:
        evs: list[str] = []
        # reference mode
        if self.mode == "reference":
            ref = sample.get(self.reference_field)
            if isinstance(ref, str) and ref.strip():
                evs.append(ref.strip())
            elif isinstance(ref, (list, tuple)):
                evs.extend([x for x in ref if isinstance(x, str) and x.strip()])
            # alias: reference_texts
            ref2 = sample.get("reference_texts")
            if isinstance(ref2, str) and ref2.strip():
                evs.append(ref2.strip())
            elif isinstance(ref2, (list, tuple)):
                evs.extend([x for x in ref2 if isinstance(x, str) and x.strip()])
            return evs

        # context mode
        ranked = sample.get(self.retrieved_field, []) or []
        try:
            rlist = _as_ranked_list(ranked)
            ids = _ranked_to_ids(rlist, dedup=self.dedup)
        except Exception:
            ids = [it[0] if isinstance(it, tuple) else it for it in ranked]

        if not ids:
            return []

        k_cut = len(ids) if self.k is None else self.k
        topk = ids[:k_cut]
        text_map = sample.get(self.doc_text_field, {}) or {}
        for did in topk:
            t = text_map.get(did)
            if isinstance(t, str) and t.strip():
                evs.append(t.strip())
            elif isinstance(t, (list, tuple)):
                evs.extend([x for x in t if isinstance(x, str) and x.strip()])
        return evs

    def _tokenize(self, s: str) -> List[str]:
        import re, string, unicodedata
        t = s or ""
        t = unicodedata.normalize("NFKC", t)
        if self.lowercase:
            t = t.lower()
        if self.strip_punct:
            t = "".join(ch for ch in t if ch not in string.punctuation)
        if self.extra_strip_chars:
            t = t.translate(str.maketrans("", "", self.extra_strip_chars))
        if self.remove_articles:
            t = re.sub(r"\b(a|an|the)\b", " ", t)
        if self.collapse_whitespace:
            t = " ".join(t.split())
        return t.split() if t else []

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        if n <= 0 or len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# -------------------------- local math utils --------------------------

def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(y*y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
