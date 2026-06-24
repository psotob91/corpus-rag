"""Hybrid search (dense + BM25, RRF) + optional rerank -> chunks + citations + artifacts.

SCAFFOLD — not implemented. RESCUE: new (LanceDB hybrid_search; rerank OFF by default)
"""
from __future__ import annotations


def search_corpus(query: str, top_k: int = 8, rerank: bool = False) -> list[dict]:
    raise NotImplementedError("scaffold — implement per ARCHITECTURE.md")
