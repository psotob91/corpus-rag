"""Multi-query retrieval (production): decompose a query into sub-queries, retrieve
each with the hybrid retriever, and RRF-fuse — casts a wider net for synthesis /
global questions (pilot A/B: +0.19 global recall). Deterministic and token-free:
the decomposition is keyword-based, and an AGENT can also pass its own richer
sub-questions via `extra_queries` (the best path, at no extra API cost). ASCII-only.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from corpus_rag import config as _config
from corpus_rag.retrieve.search import search_corpus

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "that", "this", "these", "those", "as", "at",
    "from", "we", "our", "its", "it", "not", "but", "which", "than", "then", "their",
}
# Synthesis scaffolding to strip so the core CONCEPTS become sub-queries.
_SCAFFOLD = {
    "compare", "summarize", "summary", "across", "studies", "study", "overall",
    "synthesize", "synthesis", "between", "differ", "differences", "consensus",
}


def _terms(text: str, n: int) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text or "")]
    toks = [t for t in toks if t not in _STOP and t not in _SCAFFOLD]
    return [w for w, _ in Counter(toks).most_common(n)]


def subqueries(query: str, n_sub: int = 4) -> list[str]:
    """Original query + its core content terms (synthesis scaffolding stripped)."""
    subs = [query]
    for t in _terms(query, n_sub * 3):
        subs.append(t)
        if len(subs) >= n_sub:
            break
    return subs


def rrf(ranked_lists: list[list[str]], rrf_k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over several ranked id-lists -> one ordered id-list."""
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] += 1.0 / (rrf_k + rank + 1)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def multi_query_search(
    query: str,
    top_k: int | None = None,
    n_sub: int = 4,
    config_path: str = "rag.config.yaml",
    route: bool = False,
    extra_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve with several sub-queries and RRF-fuse into top_k SearchHit dicts.

    extra_queries: agent-authored reformulations (used instead of auto-decomposition).
    """
    cfg = _config.load(config_path)
    retr = cfg.get("retrieve", {}) or {}
    k = int(top_k) if top_k else int(retr.get("top_k", 8))
    rrf_k = int(retr.get("rrf_k", 60))

    subs = [query] + list(extra_queries) if extra_queries else subqueries(query, n_sub)
    cand = max(k * 2, k)

    lists: list[list[str]] = []
    hit_by_id: dict[str, dict] = {}
    for q in subs:
        ids = []
        for h in search_corpus(q, top_k=cand, config_path=config_path, route=route):
            cid = h["chunk"]["chunk_id"]
            ids.append(cid)
            hit_by_id.setdefault(cid, h)
        lists.append(ids)

    scores: dict[str, float] = defaultdict(float)
    for lst in lists:
        for rank, cid in enumerate(lst):
            scores[cid] += 1.0 / (rrf_k + rank + 1)

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    out = []
    for cid in ordered[:k]:
        hit = dict(hit_by_id[cid])
        hit["score"] = round(scores[cid], 6)
        out.append(hit)
    return out
