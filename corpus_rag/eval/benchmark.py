"""BenchmarkQED-lite: decide hybrid vs routing vs graph from YOUR corpus, cheaply.

The cheap, LLM-free signal the research recommends BEFORE paying to build a graph:
generate LOCAL (single-source) and GLOBAL (multi-source / cross-cutting) probe
queries from the indexed corpus, run the hybrid retriever, and measure the
local-vs-global retrieval GAP. A large gap (local retrieval works, global does
not) is the signal that single-vector hybrid retrieval misses distributed
evidence -> query routing (P2) / a concept graph (P3) become candidates.

Deterministic by default (reproducible, no API key). Probes are TEMPLATED from
chunk content, so they approximate -- not replace -- real user queries; the
local-vs-global *gap* is the robust signal, not the absolute scores. For a
decision with real budget on the line, regenerate probes from actual user
questions (or an LLM) before committing to a graph.

ASCII-only output.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from corpus_rag import config as _config
from corpus_rag.retrieve.search import search_corpus

DECISIONS = (
    "HYBRID_SUFFICIENT",
    "CONSIDER_GRAPH_OR_ROUTING",
    "TUNE_RETRIEVAL_FIRST",
    "MONITOR",
    "INSUFFICIENT_DATA",
)

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "that", "this", "these", "those",
    "as", "at", "from", "we", "our", "using", "used", "use", "can", "may", "also",
    "which", "than", "then", "between", "their", "its", "it", "not", "but", "each",
    "per", "via", "into", "over", "under", "both", "all", "more", "most", "such",
    "when", "while", "after", "before", "during", "results", "methods", "study",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    ev = cfg.get("eval", {}) or {}
    th = ev.get("thresholds", {}) or {}
    corpus = cfg.get("corpus", {}) or {}
    retrieve = cfg.get("retrieve", {}) or {}
    return {
        "outputs_dir": corpus.get("outputs_dir", "outputs"),
        "top_k": int(ev.get("top_k", retrieve.get("top_k", 10))),
        "n_local": int(ev.get("n_local", 30)),
        "local_recall_ok": float(th.get("local_recall_ok", 0.6)),
        "global_recall_ok": float(th.get("global_recall_ok", 0.7)),
        "gap": float(th.get("gap", 0.25)),
    }


def _load_chunks(outputs_dir: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not outputs_dir.is_dir():
        return chunks
    for slug in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        cj = slug / "chunks.jsonl"
        if not cj.is_file():
            continue
        for line in cj.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return chunks


def _keywords(text: str, n: int) -> list[str]:
    toks = [t.lower() for t in _WORD.findall(text or "")]
    toks = [t for t in toks if t not in _STOP]
    return [w for w, _ in Counter(toks).most_common(n)]


def _term_index(text_chunks: list[dict[str, Any]]) -> dict[str, set]:
    """Content term -> set(chunk_id), to surface cross-cutting concepts."""
    idx: dict[str, set] = {}
    for c in text_chunks:
        cid = c.get("chunk_id")
        for t in set(_keywords(c.get("text", ""), 12)):
            idx.setdefault(t, set()).add(cid)
    return idx


def generate_probes(
    chunks: list[dict[str, Any]], n_local: int, k: int
) -> list[dict[str, Any]]:
    """Deterministic LOCAL (single-source) + GLOBAL (multi-source) probes.

    LOCAL: a keyword query from one chunk's own salient terms; gold = that chunk.
    GLOBAL: a cross-cutting term spread across >=2 distinct (doc, section) groups;
    gold = every chunk carrying it. Global gold is capped at k so recall can reach
    1.0 (avoids a mechanical |gold|>k penalty).
    """
    text_chunks = [c for c in chunks if c.get("kind") == "text" and c.get("text")]
    text_chunks.sort(key=lambda c: str(c.get("chunk_id", "")))  # deterministic
    probes: list[dict[str, Any]] = []

    for c in text_chunks[:n_local]:
        kws = _keywords(c["text"], 5)
        if len(kws) < 3:
            continue
        probes.append({
            "id": f"local::{c['chunk_id']}",
            "question": " ".join(kws),
            "scope": "local",
            "gold_chunk_ids": [c["chunk_id"]],
        })

    tindex = _term_index(text_chunks)
    meta = {
        c["chunk_id"]: (c.get("doc_id"), tuple(c.get("section_path") or []))
        for c in text_chunks
    }
    n_global = 0
    for term, cids in sorted(tindex.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if not (2 <= len(cids) <= k):
            continue
        groups = {meta.get(cid) for cid in cids}
        if len(groups) < 2:  # must be genuinely multi-source
            continue
        probes.append({
            "id": f"global::{term}",
            "question": term,
            "scope": "global",
            "gold_chunk_ids": sorted(cids),
        })
        n_global += 1
        if n_global >= n_local:
            break
    return probes


def _score(probe: dict[str, Any], retrieved_ids: list[str]) -> dict[str, float]:
    gold = set(probe["gold_chunk_ids"])
    inter = gold.intersection(retrieved_ids)
    recall = len(inter) / len(gold) if gold else 0.0
    rr = 0.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold:
            rr = 1.0 / rank
            break
    return {"recall": recall, "hit": 1.0 if inter else 0.0, "rr": rr}


def run_eval(config_path: str = "rag.config.yaml", top_k: int | None = None) -> dict[str, Any]:
    cfg = _config.load(config_path)
    r = _resolve(cfg)
    k = int(top_k) if top_k else r["top_k"]

    config_dir = Path(config_path).resolve().parent
    outputs_dir = Path(r["outputs_dir"])
    if not outputs_dir.is_absolute():
        outputs_dir = config_dir / outputs_dir

    chunks = _load_chunks(outputs_dir)
    probes = generate_probes(chunks, r["n_local"], k)

    per: list[dict[str, Any]] = []
    for p in probes:
        hits = search_corpus(p["question"], top_k=k, config_path=config_path)
        rids = [h["chunk"]["chunk_id"] for h in hits]
        per.append({
            "id": p["id"], "scope": p["scope"],
            "n_gold": len(p["gold_chunk_ids"]), **_score(p, rids),
        })

    def agg(scope: str) -> dict[str, Any]:
        rows = [x for x in per if x["scope"] == scope]
        n = len(rows)
        if not n:
            return {"n": 0, "recall_at_k": None, "hit_rate": None, "mrr": None}
        return {
            "n": n,
            "recall_at_k": round(sum(x["recall"] for x in rows) / n, 4),
            "hit_rate": round(sum(x["hit"] for x in rows) / n, 4),
            "mrr": round(sum(x["rr"] for x in rows) / n, 4),
        }

    return {
        "k": k,
        "n_chunks": len(chunks),
        "n_probes": len(probes),
        "local": agg("local"),
        "global": agg("global"),
        "thresholds": {kk: r[kk] for kk in ("local_recall_ok", "global_recall_ok", "gap")},
        "per_probe": per,
    }


def decide(result: dict[str, Any]) -> dict[str, Any]:
    """Map metrics -> an explicit, thresholded recommendation (with the numbers)."""
    loc = result["local"]["recall_at_k"]
    glo = result["global"]["recall_at_k"]
    th = result["thresholds"]
    k = result["k"]

    if result["local"]["n"] == 0 or loc is None:
        return {"decision": "INSUFFICIENT_DATA",
                "rationale": "No local probes could be generated; ingest more documents."}
    if loc < th["local_recall_ok"]:
        return {"decision": "TUNE_RETRIEVAL_FIRST",
                "rationale": (f"Local recall@{k}={loc:.2f} < {th['local_recall_ok']}: even "
                              "single-source retrieval is weak. Fix chunking/embeddings "
                              "before routing or a graph.")}
    if result["global"]["n"] == 0 or glo is None:
        return {"decision": "HYBRID_SUFFICIENT",
                "rationale": (f"Local recall@{k}={loc:.2f} is healthy and no multi-source "
                              "probes were found in this corpus; a graph is not justified.")}
    gap = round(loc - glo, 4)
    if glo >= th["global_recall_ok"]:
        return {"decision": "HYBRID_SUFFICIENT",
                "rationale": (f"Global recall@{k}={glo:.2f} >= {th['global_recall_ok']}: hybrid "
                              "already gathers distributed evidence. Do NOT build a graph; "
                              "revisit if the corpus grows.")}
    if gap >= th["gap"]:
        return {"decision": "CONSIDER_GRAPH_OR_ROUTING",
                "rationale": (f"Gap {gap:.2f} (local {loc:.2f} vs global {glo:.2f}) >= {th['gap']}: "
                              "hybrid finds single chunks but misses distributed evidence for "
                              "cross-cutting concepts. Candidate for query routing (P2) and/or "
                              "LazyGraphRAG (P3) -- validate on real user queries first.")}
    return {"decision": "MONITOR",
            "rationale": (f"Global recall@{k}={glo:.2f} is below target but the gap {gap:.2f} is "
                          "small; hybrid is borderline. Add more eval queries (ideally "
                          "LLM/user-authored) before deciding.")}


def _report_md(result: dict[str, Any], decision: dict[str, Any]) -> str:
    def row(name: str, d: dict[str, Any]) -> str:
        return f"| {name} | {d['n']} | {d['recall_at_k']} | {d['hit_rate']} | {d['mrr']} |"

    return "\n".join([
        "# corpus-rag eval (BenchmarkQED-lite)",
        "",
        f"- chunks indexed: {result['n_chunks']}  |  probes: {result['n_probes']}  |  k = {result['k']}",
        f"- thresholds: {result['thresholds']}",
        "",
        "| scope | n | recall@k | hit_rate | MRR |",
        "|---|---|---|---|---|",
        row("local (single-source)", result["local"]),
        row("global (multi-source)", result["global"]),
        "",
        f"## Decision: {decision['decision']}",
        "",
        decision["rationale"],
        "",
        "> Probes are templated from chunk content (deterministic, no LLM). The "
        "local-vs-global *gap* is the robust signal; absolute scores are approximate. "
        "For a budget decision, regenerate probes from real user questions before "
        "building a graph.",
    ])


def evaluate(
    config_path: str = "rag.config.yaml",
    top_k: int | None = None,
    report_dir: str | None = None,
) -> dict[str, Any]:
    result = run_eval(config_path, top_k)
    decision = decide(result)
    config_dir = Path(config_path).resolve().parent
    rdir = Path(report_dir) if report_dir else config_dir / ".rag" / "eval"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "eval.json").write_text(
        json.dumps({"result": result, "decision": decision}, indent=2), encoding="utf-8"
    )
    (rdir / "eval.md").write_text(_report_md(result, decision), encoding="utf-8")
    return {"result": result, "decision": decision, "report_dir": str(rdir)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else "rag.config.yaml"
    out = evaluate(config_path)
    r, d = out["result"], out["decision"]
    print(f"probes={r['n_probes']} k={r['k']} "
          f"local_recall={r['local']['recall_at_k']} global_recall={r['global']['recall_at_k']}")
    print(f"DECISION: {d['decision']}")
    print(d["rationale"])
    print(f"report: {out['report_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
