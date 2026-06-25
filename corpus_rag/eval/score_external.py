"""Score an EXTERNAL/independent validation set (questions + relevance gold)
authored by ANOTHER model (e.g. Gemini/ChatGPT), keeping the gold blind.

Run this YOURSELF on your machine; it reads the gold file you hold and prints ONLY
aggregate metrics (recall / hit_rate by scope) — never per-item gold. So you can
report the aggregates to whoever built the retriever WITHOUT exposing the answer
key, which prevents test-set leakage / tuning-to-the-test bias.

Usage:
  python -m corpus_rag.eval.score_external <gold.json> [config.yaml] \\
      [--route] [--k K] [--sample N] [--seed S]

gold.json: a JSON array (or {"items": [...]}) of objects:
  {"id","scope":"local|global","question","gold_chunk_ids":[...],"answerable":bool}
Repeated disjoint --sample draws (different --seed) give use-once held-out sets.
ASCII-only output.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from corpus_rag.retrieve.search import search_corpus


def _load_gold(path: str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) and "items" in data else data


def score(
    gold_path: str,
    config_path: str = "rag.config.yaml",
    top_k: int | None = None,
    route: bool | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    items = _load_gold(gold_path)
    answerable = [it for it in items if it.get("answerable", True) and it.get("gold_chunk_ids")]
    if sample and sample < len(answerable):
        answerable = random.Random(seed).sample(answerable, sample)

    by_scope: dict[str, list] = defaultdict(list)
    for it in answerable:
        hits = search_corpus(it["question"], top_k=top_k, config_path=config_path, route=route)
        ids = [h["chunk"]["chunk_id"] for h in hits]
        gold = set(it["gold_chunk_ids"])
        recall = len(gold.intersection(ids)) / len(gold)
        by_scope[it.get("scope", "?")].append((recall, 1.0 if gold.intersection(ids) else 0.0))

    out: dict[str, Any] = {}
    allrows: list = []
    for scope, rows in by_scope.items():
        allrows += rows
        n = len(rows)
        out[scope] = {
            "n": n,
            "recall": round(sum(r for r, _ in rows) / n, 4),
            "hit_rate": round(sum(h for _, h in rows) / n, 4),
        }
    if allrows:
        n = len(allrows)
        out["overall"] = {
            "n": n,
            "recall": round(sum(r for r, _ in allrows) / n, 4),
            "hit_rate": round(sum(h for _, h in allrows) / n, 4),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m corpus_rag.eval.score_external <gold.json> [config] "
              "[--route] [--k K] [--sample N] [--seed S]")
        return 2

    gold = argv[0]
    config, route, sample, seed, k = "rag.config.yaml", None, None, 0, None
    rest, i = argv[1:], 0
    while i < len(rest):
        a = rest[i]
        if a == "--route":
            route = True
        elif a == "--sample":
            i += 1
            sample = int(rest[i])
        elif a == "--seed":
            i += 1
            seed = int(rest[i])
        elif a == "--k":
            i += 1
            k = int(rest[i])
        elif not a.startswith("--"):
            config = a
        i += 1

    res = score(gold, config_path=config, top_k=k, route=route, sample=sample, seed=seed)
    print("EXTERNAL validation (aggregate only; the gold stays in your file):")
    for scope in sorted(res):
        d = res[scope]
        print(f"  {scope:8} n={d['n']:3}  recall={d['recall']}  hit_rate={d['hit_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
