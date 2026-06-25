"""Pilot retrieval experiments: chunk-type stratification + a multi-query x agentic factorial.

Deterministic and LLM-free (free to run, reproducible). In PRODUCTION an agent
(Claude) authors the sub-queries / follow-ups with reasoning; this pilot decomposes
a query by its content terms to measure the retrieval-COVERAGE ceiling cheaply.
All cells return the same budget B, so the factorial isolates the strategy effect.
ASCII-only output.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from corpus_rag import config as _config
from corpus_rag.eval.benchmark import (
    _keywords,
    _load_chunks,
    _resolve,
    generate_probes,
)
from corpus_rag.retrieve.search import search_corpus

# Synthesis scaffolding to strip when decomposing a query into sub-queries.
_SCAFFOLD = {
    "compare", "summarize", "summary", "across", "studies", "study", "overall",
    "synthesize", "synthesis", "between", "differ", "differences", "consensus",
}


def _outputs_dir(config_path: str) -> Path:
    r = _resolve(_config.load(config_path))
    d = Path(r["outputs_dir"])
    return d if d.is_absolute() else Path(config_path).resolve().parent / d


def _ids(query: str, k: int, config_path: str) -> list[dict[str, Any]]:
    return search_corpus(query, top_k=k, config_path=config_path, route=False)


def _rrf(ranked_lists: list[list[str]], rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] += 1.0 / (rrf_k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _subqueries(query: str, n_sub: int) -> list[str]:
    """Original query + its core content terms (synthesis scaffolding stripped)."""
    subs = [query]
    for t in _keywords(query, n_sub * 3):
        if t not in _SCAFFOLD:
            subs.append(t)
        if len(subs) >= n_sub:
            break
    return subs


# ── Chunk-type stratified eval: can we retrieve tables / figures / formulas? ──

def eval_by_chunk_kind(
    config_path: str = "rag.config.yaml", k: int = 10, per_kind: int = 25
) -> dict[str, Any]:
    chunks = _load_chunks(_outputs_dir(config_path))
    by_kind: dict[str, list] = defaultdict(list)
    for c in chunks:
        by_kind[c.get("kind", "text")].append(c)

    out: dict[str, Any] = {}
    for kind, cs in by_kind.items():
        cs = sorted(cs, key=lambda c: str(c.get("chunk_id", "")))[:per_kind]
        recalls = []
        for c in cs:
            kws = _keywords(c.get("text", "") or "", 5)
            if len(kws) < 3:  # too little searchable text (e.g. a bare caption)
                continue
            ids = [h["chunk"]["chunk_id"] for h in _ids(" ".join(kws), k, config_path)]
            recalls.append(1.0 if c["chunk_id"] in ids else 0.0)
        out[kind] = {
            "n_probed": len(recalls),
            "n_total": len(cs),
            "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else None,
        }
    return {"k": k, "by_kind": out}


# ── Factorial: single vs multi-query x non-agentic vs agentic ──

def expand_retrieve(
    query: str, B: int, config_path: str,
    multi: bool = False, agentic: bool = False, n_sub: int = 4, rounds: int = 2,
) -> list[str]:
    """Return up to B chunk ids gathered by the chosen strategy (RRF-fused)."""
    id_text: dict[str, str] = {}
    lists: list[list[str]] = []

    def run(q: str) -> None:
        ids = []
        for h in _ids(q, B, config_path):
            cid = h["chunk"]["chunk_id"]
            ids.append(cid)
            id_text[cid] = h["chunk"].get("text", "") or ""
        lists.append(ids)

    for q in (_subqueries(query, n_sub) if multi else [query]):
        run(q)
    fused = _rrf(lists)

    if agentic:
        for _ in range(max(0, rounds - 1)):
            seen = " ".join(id_text.get(cid, "") for cid in fused[:B])
            terms = [t for t in _keywords(seen, n_sub * 2) if t not in _SCAFFOLD][:n_sub]
            followup = " ".join(terms) or query
            for q in (_subqueries(followup, n_sub) if multi else [followup]):
                run(q)
            fused = _rrf(lists)
    return fused[:B]


def factorial(
    config_path: str = "rag.config.yaml", B: int = 20, n_sub: int = 4, rounds: int = 2
) -> dict[str, Any]:
    chunks = _load_chunks(_outputs_dir(config_path))
    r = _resolve(_config.load(config_path))
    probes = [p for p in generate_probes(chunks, r["n_local"], B) if p["scope"] == "global"]

    cells = {
        "single": (False, False),
        "multiquery": (True, False),
        "agentic": (False, True),
        "both": (True, True),
    }
    res: dict[str, Any] = {}
    for name, (m, a) in cells.items():
        recalls, sizes = [], []
        for p in probes:
            ids = expand_retrieve(p["question"], B, config_path, multi=m, agentic=a,
                                  n_sub=n_sub, rounds=rounds)
            gold = set(p["gold_chunk_ids"])
            recalls.append(len(gold.intersection(ids)) / len(gold) if gold else 0.0)
            sizes.append(len(ids))
        res[name] = {
            "recall_at_B": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "avg_returned": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        }
    return {"B": B, "n_global_probes": len(probes), "n_sub": n_sub, "rounds": rounds, "cells": res}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else "rag.config.yaml"

    ck = eval_by_chunk_kind(config_path)
    print(f"== chunk-type retrieval (recall@{ck['k']}) ==")
    for kind, d in sorted(ck["by_kind"].items()):
        print(f"  {kind:8} recall={d['recall_at_k']}  (probed {d['n_probed']}/{d['n_total']})")

    fac = factorial(config_path)
    print(f"\n== factorial multi-query x agentic (recall@B, B={fac['B']}, "
          f"n_sub={fac['n_sub']}, rounds={fac['rounds']}, {fac['n_global_probes']} global probes) ==")
    base = fac["cells"]["single"]["recall_at_B"]
    for name in ("single", "multiquery", "agentic", "both"):
        c = fac["cells"][name]
        delta = "" if base is None else f"  ({c['recall_at_B'] - base:+.3f} vs single)"
        print(f"  {name:11} recall={c['recall_at_B']}  returned~{c['avg_returned']}{delta}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
