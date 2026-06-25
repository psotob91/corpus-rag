"""Honest 4-arm benchmark of the graph retriever on the pilot GLOBAL probes.

Arms (all return <=20 chunks):
  (a) hybrid   = search_corpus
  (b) multi    = multi_query_search
  (c) graph    = graph_search
  (d) graph+hybrid = RRF-fuse(graph, hybrid) via multi.rrf

Metric: recall@20 of gold_chunk_ids per GLOBAL probe (k=20). Reports mean per arm
and a one-line verdict vs multi-query. No faked numbers; graph errors are surfaced.
ASCII-only output.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

CONFIG = str(Path(".rag/pilot/rag.config.yaml").resolve())
K = 20


def _ascii(s: str) -> str:
    return (s or "").encode("ascii", errors="replace").decode("ascii")


def main() -> int:
    from corpus_rag.eval.benchmark import generate_probes, _load_chunks, _resolve
    from corpus_rag import config as _config
    from corpus_rag.retrieve.search import search_corpus
    from corpus_rag.retrieve.multi import multi_query_search, rrf
    from corpus_rag.retrieve import graph as graph_mod

    cfg = _config.load(CONFIG)
    r = _resolve(cfg)
    config_dir = Path(CONFIG).resolve().parent
    outputs_dir = Path(r["outputs_dir"])
    if not outputs_dir.is_absolute():
        outputs_dir = config_dir / outputs_dir

    chunks = _load_chunks(outputs_dir)
    probes = generate_probes(chunks, r["n_local"], K)
    global_probes = [p for p in probes if p["scope"] == "global"]
    print(_ascii(f"[bench] chunks={len(chunks)} global_probes={len(global_probes)} k={K}"))

    # Build the graph fresh on the pilot.
    print("[bench] building graph on pilot ...")
    graph_mod.build_graph(CONFIG)

    def recall(gold: list[str], retrieved_ids: list[str]) -> float:
        g = set(gold)
        if not g:
            return 0.0
        return len(g.intersection(retrieved_ids)) / len(g)

    arms = ("hybrid", "multi", "graph", "graph+hybrid")
    sums = {a: 0.0 for a in arms}
    per_probe = []
    graph_errors = 0
    first_graph_err = None

    for p in global_probes:
        q = p["question"]
        gold = p["gold_chunk_ids"]

        hybrid_hits = search_corpus(q, top_k=K, config_path=CONFIG)
        hybrid_ids = [h["chunk"]["chunk_id"] for h in hybrid_hits]

        multi_hits = multi_query_search(q, top_k=K, config_path=CONFIG)
        multi_ids = [h["chunk"]["chunk_id"] for h in multi_hits]

        try:
            graph_hits = graph_mod.graph_search(q, top_k=K, config_path=CONFIG)
            graph_ids = [h["chunk"]["chunk_id"] for h in graph_hits]
        except Exception as e:  # surface, do not fake
            graph_errors += 1
            if first_graph_err is None:
                first_graph_err = "".join(
                    traceback.format_exception_only(type(e), e)
                ).strip()
            graph_ids = []

        # graph+hybrid: RRF-fuse the two ranked id-lists, take top K.
        fused_ids = rrf([graph_ids, hybrid_ids], rrf_k=int(r.get("rrf_k", 60) or 60))[:K]

        rec = {
            "hybrid": recall(gold, hybrid_ids[:K]),
            "multi": recall(gold, multi_ids[:K]),
            "graph": recall(gold, graph_ids[:K]),
            "graph+hybrid": recall(gold, fused_ids),
        }
        for a in arms:
            sums[a] += rec[a]
        per_probe.append({"id": p["id"], "n_gold": len(gold), **rec})

    n = len(global_probes)
    means = {a: (sums[a] / n if n else 0.0) for a in arms}

    print("")
    print("[bench] per-probe recall@%d (global probes):" % K)
    for row in per_probe:
        print(_ascii(
            "  %-28s gold=%2d hybrid=%.3f multi=%.3f graph=%.3f graph+hybrid=%.3f"
            % (row["id"][:28], row["n_gold"], row["hybrid"], row["multi"],
               row["graph"], row["graph+hybrid"])
        ))

    print("")
    print("[bench] MEAN recall@%d over %d global probes:" % (K, n))
    for a in arms:
        print("  %-14s %.4f" % (a, means[a]))

    if graph_errors:
        print(_ascii("[bench] graph_search errored on %d/%d probes; first error: %s"
                     % (graph_errors, n, first_graph_err)))

    g, m = means["graph"], means["multi"]
    eps = 1e-9
    if g > m + eps:
        verdict = "graph BEATS multi-query (%.4f vs %.4f, +%.4f)" % (g, m, g - m)
    elif g < m - eps:
        verdict = "graph LOSES to multi-query (%.4f vs %.4f, %.4f)" % (g, m, g - m)
    else:
        verdict = "graph TIES multi-query (%.4f vs %.4f)" % (g, m)
    print("")
    print(_ascii("[bench] VERDICT: " + verdict))

    # machine-readable line for the harness to parse
    import json
    print("RESULT_JSON=" + json.dumps({
        "means": means, "n_global": n, "graph_errors": graph_errors,
        "first_graph_err": first_graph_err, "verdict": verdict,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
