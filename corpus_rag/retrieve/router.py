"""Query router (P2): local (single-source) vs global (multi-source / synthesis).

Routes a query so the retriever can go deeper/wider on synthesis questions. Two
tiers, dependency-free (reuses the fastembed embedder already in core deps):
embedding similarity to frozen LOCAL/GLOBAL exemplars, with a lexical-marker
fallback when the embedding margin is small or the embedder is unavailable.
Deterministic (constants only). ASCII-only.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from corpus_rag.index.embed import get_embedder

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Frozen exemplars (biostat/epi flavour). LOCAL = answerable from one chunk;
# GLOBAL = needs evidence gathered across many chunks/docs.
LOCAL_EXEMPLARS = (
    "what is the value of the .632 estimator",
    "definition of the hazard ratio",
    "what sample size does Riley 2020 recommend",
    "the equation for the c-statistic",
    "what does table 2 report",
    "which optimism-correction did Harrell use",
    "the formula for net benefit",
    "what is the bootstrap .632+ rule",
)
GLOBAL_EXEMPLARS = (
    "compare bootstrap and cross-validation across studies",
    "summarize the methods for sample size in prediction models",
    "how do calibration approaches differ overall",
    "what is the consensus on optimism correction",
    "synthesize the recommendations across these papers",
    "trends in prediction-model validation over time",
    "relationship between shrinkage and overfitting across methods",
    "overall, how should model performance be assessed",
)

_GLOBAL_MARKERS = (
    "compare", "across", "overall", "summary", "summarize", "synthesize", "trend",
    "differ", "differences", "versus", " vs ", "relationship", "consensus",
    "in general", "all studies", "between",
)
_LOCAL_MARKERS = (
    "define", "definition", "what is", "value of", "equation", "formula",
    "table", "figure",
)


def _cos(a, b) -> float:
    num = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return num / (na * nb) if na and nb else 0.0


@lru_cache(maxsize=4)
def _exemplar_vecs(model: str):
    emb = get_embedder(model)
    return emb.encode(list(LOCAL_EXEMPLARS)), emb.encode(list(GLOBAL_EXEMPLARS))


def _lexical_scope(query: str) -> tuple[str, int]:
    q = f" {query.lower()} "
    g = sum(1 for m in _GLOBAL_MARKERS if m in q)
    loc = sum(1 for m in _LOCAL_MARKERS if m in q)
    if g > loc:
        return "global", g - loc
    if loc > g:
        return "local", loc - g
    return "local", 0  # tie -> default to the fast (local) path


def classify_query(
    query: str, model: str = _DEFAULT_MODEL, margin: float = 0.03
) -> dict[str, Any]:
    """Return {'scope': 'local'|'global', 'method': 'exemplar'|'lexical', 'score': float}."""
    lex_scope, lex_strength = _lexical_scope(query)
    try:
        loc_vecs, glo_vecs = _exemplar_vecs(model)
        qv = get_embedder(model).encode([query])[0]
        loc_sim = max(_cos(qv, v) for v in loc_vecs)
        glo_sim = max(_cos(qv, v) for v in glo_vecs)
        diff = glo_sim - loc_sim
        if abs(diff) >= margin:
            return {
                "scope": "global" if diff > 0 else "local",
                "method": "exemplar",
                "score": round(diff, 4),
            }
    except Exception:
        pass
    return {"scope": lex_scope, "method": "lexical", "score": float(lex_strength)}
