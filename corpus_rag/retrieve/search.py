"""Hybrid search (dense + BM25, RRF) + optional rerank -> chunks + citations.

Embeds the query, runs a dense vector search AND a full-text/BM25 search against
the LanceDB 'chunks' table, and fuses the two ranked lists with Reciprocal Rank
Fusion (RRF). Returns the top_k results as SearchHit dicts, each carrying the
Chunk and a Citation (doc_id, chunk_id, section_path parsed back to a list, page,
artifact_id).

Rerank is OFF by default (evidence: cross-encoders can degrade retrieval on
text+table corpora). When rerank=True and the optional 'rerank' extra is
installed, a cross-encoder reorders the fused candidates; otherwise it is skipped.

ASCII-only output.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from corpus_rag import config as _config
from corpus_rag.index.embed import get_embedder

TABLE_NAME = "chunks"


def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    index = cfg.get("index", {}) or {}
    embedder_cfg = index.get("embedder", {}) or {}
    retrieve = cfg.get("retrieve", {}) or {}
    rerank_cfg = retrieve.get("rerank", {}) or {}
    routing_cfg = retrieve.get("routing", {}) or {}
    return {
        "index_path": index.get("path", ".rag/corpus.lance"),
        "model": embedder_cfg.get("model", "BAAI/bge-small-en-v1.5"),
        "rrf_k": int(retrieve.get("rrf_k", 60)),
        "top_k": int(retrieve.get("top_k", 8)),
        "rerank_enabled": bool(rerank_cfg.get("enabled", False)),
        "rerank_model": rerank_cfg.get("model", "BAAI/bge-reranker-base"),
        "routing_enabled": bool(routing_cfg.get("enabled", False)),
        "routing_margin": float(routing_cfg.get("margin", 0.03)),
        "global_candidate_k": int(routing_cfg.get("global_candidate_k", 60)),
        "global_top_k": int(routing_cfg.get("global_top_k", 20)),
    }


@lru_cache(maxsize=4)
def _open_table(index_path: str):
    """Open the LanceDB 'chunks' table read-only (cached per index path)."""
    import lancedb

    db = lancedb.connect(index_path)
    return db.open_table(TABLE_NAME)


def _resolve_index_path(cfg_index_path: str, config_path: str) -> str:
    p = Path(cfg_index_path)
    if not p.is_absolute():
        p = Path(config_path).resolve().parent / p
    return str(p)


def _rrf_fuse(
    ranked_lists: list[list[str]], rrf_k: int
) -> dict[str, float]:
    """Reciprocal Rank Fusion: id -> summed 1/(rrf_k + rank) over the lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def _row_to_hit(row: dict[str, Any], score: float) -> dict[str, Any]:
    """Build a SearchHit dict (chunk + score + citation) from a table row."""
    try:
        section_path = json.loads(row.get("section_path") or "[]")
        if not isinstance(section_path, list):
            section_path = []
    except (json.JSONDecodeError, TypeError):
        section_path = []

    page = row.get("page")
    page = int(page) if page is not None and int(page) >= 0 else None
    artifact_id = row.get("artifact_id") or None

    chunk: dict[str, Any] = {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "text": row.get("text"),
        "kind": row.get("kind"),
        "section_path": section_path,
    }
    if page is not None:
        chunk["page"] = page
    if artifact_id:
        chunk["artifact_id"] = artifact_id

    citation: dict[str, Any] = {
        "doc_id": row.get("doc_id"),
        "chunk_id": row.get("chunk_id"),
        "section_path": section_path,
    }
    if page is not None:
        citation["page"] = page
    if artifact_id:
        citation["artifact_id"] = artifact_id

    return {"chunk": chunk, "score": float(score), "citation": citation}


def _maybe_rerank(
    query: str, hits: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Reorder hits with a cross-encoder if the 'rerank' extra is installed."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError:
        # 'rerank' extra not installed -> silently keep the fused order.
        return hits
    ce = CrossEncoder(model)
    pairs = [[query, h["chunk"].get("text") or ""] for h in hits]
    scores = ce.predict(pairs)
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def search_corpus(
    query: str,
    top_k: int | None = None,
    rerank: bool = False,
    config_path: str = "rag.config.yaml",
    route: bool | None = None,
) -> list[dict[str, Any]]:
    """Hybrid (dense + BM25, RRF) search. Returns top_k SearchHit dicts.

    top_k=None uses retrieve.top_k from config. route=None reads
    retrieve.routing.enabled (default off, so behaviour is unchanged). When
    routing is on and the query is classified GLOBAL, the candidate pool deepens
    and -- only if top_k was not passed explicitly -- the returned set widens to
    retrieve.routing.global_top_k (keeps eval A/B apples-to-apples).
    """
    cfg = _config.load(config_path)
    resolved = _resolve(cfg)
    rrf_k = resolved["rrf_k"]

    top_k_explicit = top_k is not None
    if top_k is None:
        top_k = resolved["top_k"]

    index_path = _resolve_index_path(resolved["index_path"], config_path)
    table = _open_table(index_path)

    candidate_k = max(top_k * 4, top_k)

    do_route = resolved["routing_enabled"] if route is None else route
    route_info: dict[str, Any] | None = None
    if do_route:
        from corpus_rag.retrieve.router import classify_query
        route_info = classify_query(
            query, model=resolved["model"], margin=resolved["routing_margin"]
        )
        if route_info["scope"] == "global":
            candidate_k = max(candidate_k, resolved["global_candidate_k"])
            if not top_k_explicit:
                top_k = resolved["global_top_k"]

    qvec = get_embedder(resolved["model"]).encode([query])[0]

    # Dense (vector) search.
    vector_rows = table.search(qvec).limit(candidate_k).to_list()
    # Full-text / BM25 search.
    try:
        fts_rows = table.search(query, query_type="fts").limit(candidate_k).to_list()
    except Exception:
        # FTS index missing or query unparseable -> dense-only fallback.
        fts_rows = []

    by_id: dict[str, dict[str, Any]] = {}
    for row in vector_rows + fts_rows:
        cid = row.get("chunk_id")
        if cid is not None and cid not in by_id:
            by_id[cid] = row

    vector_ids = [r["chunk_id"] for r in vector_rows if r.get("chunk_id") is not None]
    fts_ids = [r["chunk_id"] for r in fts_rows if r.get("chunk_id") is not None]

    fused = _rrf_fuse([vector_ids, fts_ids], rrf_k)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    hits = [_row_to_hit(by_id[cid], score) for cid, score in ordered if cid in by_id]

    do_rerank = rerank or resolved["rerank_enabled"]
    if do_rerank and hits:
        hits = _maybe_rerank(query, hits, resolved["rerank_model"])

    hits = hits[:top_k]
    if route_info and hits:
        hits[0]["_route"] = {**route_info, "candidate_k": candidate_k, "returned_top_k": top_k}
    return hits
