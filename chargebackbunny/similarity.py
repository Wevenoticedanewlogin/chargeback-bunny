"""Lightweight text similarity for repeat-claimant candidate retrieval.

TF-IDF cosine over word shingles — good enough to shortlist similar past
disputes before spending an LLM judge call, with zero dependencies.

"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _shingles(tokens: list[str], n: int = 2) -> list[str]:
    if len(tokens) < n:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def jaccard_from_sets(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_k_similar(query: str, docs: dict[str, str], k: int = 3) -> list[tuple[str, float]]:
    """Return [(doc_id, score)] sorted desc. Score blends TF-IDF cosine on
    bigram shingles with unigram Jaccard."""
    if not docs:
        return []
    q_tokens = tokenize(query)
    q_shingles = Counter(_shingles(q_tokens))
    q_unigrams = set(q_tokens)

    doc_tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
    df: Counter[str] = Counter()
    for toks in doc_tokens.values():
        for sh in set(_shingles(toks)):
            df[sh] += 1
    n_docs = len(docs)

    def tfidf_vec(shingles: Counter[str]) -> dict[str, float]:
        total = sum(shingles.values()) or 1
        return {
            sh: (count / total) * math.log((n_docs + 1) / (df[sh] + 1) + 1)
            for sh, count in shingles.items()
        }

    q_vec = tfidf_vec(q_shingles)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored: list[tuple[str, float]] = []
    for doc_id, toks in doc_tokens.items():
        d_vec = tfidf_vec(Counter(_shingles(toks)))
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(v * d_vec.get(sh, 0.0) for sh, v in q_vec.items())
        cosine = dot / (q_norm * d_norm)
        jac = jaccard_from_sets(q_unigrams, set(toks))
        scored.append((doc_id, 0.7 * cosine + 0.3 * jac))

    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [(doc_id, round(score, 4)) for doc_id, score in scored[:k] if score > 0.05]
