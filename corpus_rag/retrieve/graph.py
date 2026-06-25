"""LazyGraphRAG retriever ARM (token-free): concept co-occurrence graph + Leiden.

A genuine, LLM-free retriever faithful to Microsoft's LazyGraphRAG, adapted to
this local stack (spaCy noun-chunk concepts + networkx co-occurrence graph +
Leiden communities + fastembed concept vectors). It attacks the weak spot of a
single hybrid search on GLOBAL / synthesis queries: evidence split ACROSS
documents, where each source chunk is near only its own facet and none is near
the blended query vector. The graph DECOMPOSES the query into seed concepts,
follows CROSS-DOCUMENT co-occurrence edges, confines expansion to the query's
Leiden community under a hard relevance budget, and scores highest the chunks
that BRIDGE multiple query-relevant concepts (distributed evidence).

Two public functions, mirroring the other arms:

- build_graph(config_path) -> dict   [THE RECIPE, offline]
- graph_search(query, top_k, config_path) -> list[SearchHit]   [RUNTIME, LLM-free]

SearchHit dicts are built in the EXACT search._row_to_hit shape so they are
RRF-fusable (multi.rrf) and MCP-usable. NOTE the one difference: in chunks.jsonl
`section_path` is ALREADY a list -> it is passed through directly and NEVER
json.loads'd (that step is only for the lance-row path in search.py).

Import-safe: ONLY stdlib + corpus_rag.config at module load. spacy / networkx /
igraph / leidenalg / numpy / fastembed are imported lazily INSIDE functions.
ASCII-only output (Windows cp1252). Deterministic: fixed Leiden seed, sorted
iteration everywhere, lexicographic tie-breaks. Artifacts persist under .rag/
(gitignored) -> "commit the recipe, not the index".
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from corpus_rag import config as _config

CODE_VERSION = "graph-1.0"
DIM = 384

# Single-word generic concepts to drop (allowed INSIDE multiword concepts).
# benchmark._STOP plus a domain-generic stoplist.
_BENCH_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "that", "this", "these", "those",
    "as", "at", "from", "we", "our", "using", "used", "use", "can", "may", "also",
    "which", "than", "then", "between", "their", "its", "it", "not", "but", "each",
    "per", "via", "into", "over", "under", "both", "all", "more", "most", "such",
    "when", "while", "after", "before", "during", "results", "methods", "study",
}
_GENERIC = _BENCH_STOP | {
    "model", "method", "result", "study", "approach", "figure", "table", "paper",
    "value", "data", "analysis",
}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the graph.* block with real-corpus defaults (see SMALL-CORPUS note)."""
    corpus = cfg.get("corpus", {}) or {}
    index = cfg.get("index", {}) or {}
    index_emb = index.get("embedder", {}) or {}
    g = cfg.get("graph", {}) or {}
    g_emb = g.get("embedder", {}) or {}
    return {
        "outputs_dir": corpus.get("outputs_dir", "outputs"),
        "enabled": bool(g.get("enabled", False)),
        "path": g.get("path", ".rag/graph"),
        "embedder_model": g_emb.get("model", index_emb.get("model", "BAAI/bge-small-en-v1.5")),
        "min_concept_freq": int(g.get("min_concept_freq", 2)),
        "min_concept_chars": int(g.get("min_concept_chars", 3)),
        "max_concept_tokens": int(g.get("max_concept_tokens", 4)),
        "max_concepts": int(g.get("max_concepts", 4000)),
        "max_concepts_per_chunk": int(g.get("max_concepts_per_chunk", 40)),
        "drop_generic": bool(g.get("drop_generic", True)),
        "min_edge_weight": int(g.get("min_edge_weight", 2)),
        "max_node_degree": g.get("max_node_degree", None),
        "leiden_resolution": float(g.get("leiden_resolution", 1.0)),
        "leiden_seed": int(g.get("leiden_seed", 42)),
        "seed_sim_min": float(g.get("seed_sim_min", 0.30)),
        "seed_z": float(g.get("seed_z", 2.0)),
        "seed_top_n": int(g.get("seed_top_n", 12)),
        "seed_concepts": list(g.get("seed_concepts", []) or []),
        "relevant_communities": int(g.get("relevant_communities", 5)),
        "community_prior_w": float(g.get("community_prior_w", 0.5)),
        "max_hops": int(g.get("max_hops", 2)),
        "hop_decay": float(g.get("hop_decay", 0.5)),
        "budget_concepts": int(g.get("budget_concepts", 200)),
        "budget_chunks": int(g.get("budget_chunks", 600)),
        "fusion": str(g.get("fusion", "accum")),
        "rrf_k": int(g.get("rrf_k", 60)),
        "alpha": float(g.get("alpha", 0.5)),
        "multi_hit_w": float(g.get("multi_hit_w", 0.3)),
        "fallback_hybrid": bool(g.get("fallback_hybrid", False)),
    }


def _resolve_path(rel_or_abs: str, config_path: str) -> Path:
    """Resolve graph.path relative to config_path.parent (mirrors _resolve_index_path)."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = Path(config_path).resolve().parent / p
    return p


def _resolve_outputs_dir(outputs_dir: str, config_path: str) -> Path:
    p = Path(outputs_dir)
    if not p.is_absolute():
        p = Path(config_path).resolve().parent / p
    return p


# ---------------------------------------------------------------------------
# Chunk loading (replicate benchmark._load_chunks EXACTLY)
# ---------------------------------------------------------------------------
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


def _ascii(s: str) -> str:
    """Strip to ASCII for cp1252-safe print()."""
    return (s or "").encode("ascii", errors="replace").decode("ascii")


# ---------------------------------------------------------------------------
# Concept extraction (pure-ish helper, unit-tested with spaCy)
# ---------------------------------------------------------------------------
def _clean_concept(span, drop_generic: bool, min_chars: int, max_tokens: int) -> str | None:
    """Clean one spaCy noun_chunk span -> a canonical concept string, or None.

    1. Drop leading DET/PRON/NUM/stopword tokens.
    2. Keep alpha (hyphen-internal ok), non-stop, non-punct tokens; lemmatize+lower.
    3. Join with single spaces.
    4. Reject if too short, too many tokens, empty, or a bare generic single token.
    """
    toks = list(span)
    # 1. strip leading determiners / pronouns / numbers / stopwords
    start = 0
    while start < len(toks):
        t = toks[start]
        if t.pos_ in {"DET", "PRON", "NUM"} or t.is_stop:
            start += 1
        else:
            break
    kept: list[str] = []
    for t in toks[start:]:
        if t.is_punct or t.is_stop:
            continue
        # allow internal hyphen: token text alpha when hyphens removed
        txt = t.text.replace("-", "")
        if not txt.isalpha():
            continue
        lemma = (t.lemma_ or t.text).lower().strip()
        if lemma:
            kept.append(lemma)
    if not kept:
        return None
    if len(kept) > max_tokens:
        return None
    concept = " ".join(kept)
    concept = " ".join(concept.split())  # collapse whitespace
    if len(concept) < min_chars:
        return None
    if drop_generic and len(kept) == 1 and kept[0] in _GENERIC:
        return None
    return concept


def _extract_concepts(
    text_chunks: list[dict[str, Any]], resolved: dict[str, Any]
) -> tuple[dict[str, set], dict[str, set], int]:
    """Run spaCy over chunk texts -> (concept->chunk_ids, concept->doc_ids, n_raw_spans).

    Processes chunks in sorted(chunk_id) order for determinism.
    """
    try:
        import spacy
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "spacy not installed. Install the [graph] extra: "
            "uv pip install -e .[graph] && python -m spacy download en_core_web_sm"
        ) from e

    try:
        # CRITICAL: only disable 'ner'. noun_chunks needs the parser; we need lemmas.
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it: "
            "python -m spacy download en_core_web_sm (needs the [graph] extra)."
        ) from e

    ordered = sorted(text_chunks, key=lambda c: str(c.get("chunk_id", "")))
    texts = [c.get("text", "") or "" for c in ordered]
    concept_chunks: dict[str, set] = {}
    concept_docs: dict[str, set] = {}
    n_raw = 0
    for chunk, doc in zip(ordered, nlp.pipe(texts, batch_size=64)):
        cid = chunk.get("chunk_id")
        did = chunk.get("doc_id")
        if not cid:
            continue
        for span in doc.noun_chunks:
            n_raw += 1
            concept = _clean_concept(
                span,
                resolved["drop_generic"],
                resolved["min_concept_chars"],
                resolved["max_concept_tokens"],
            )
            if concept is None:
                continue
            concept_chunks.setdefault(concept, set()).add(cid)
            concept_docs.setdefault(concept, set()).add(did)
    return concept_chunks, concept_docs, n_raw


# ---------------------------------------------------------------------------
# Co-occurrence graph
# ---------------------------------------------------------------------------
def _cooccurrence(
    text_chunks: list[dict[str, Any]],
    concept_chunks: dict[str, set],
    surviving: set,
    resolved: dict[str, Any],
):
    """Build a pruned, weighted networkx co-occurrence graph over surviving concepts."""
    import networkx as nx

    freq = {c: len(concept_chunks[c]) for c in surviving}
    # chunk_id -> sorted list of surviving concepts in it
    chunk_concepts: dict[str, set] = {}
    for c in surviving:
        for cid in concept_chunks[c]:
            chunk_concepts.setdefault(cid, set()).add(c)

    cooccur: dict[tuple, int] = {}
    cooccur_docs: dict[tuple, set] = {}
    cid_to_doc = {c.get("chunk_id"): c.get("doc_id") for c in text_chunks}
    cap = resolved["max_concepts_per_chunk"]
    for cid in sorted(chunk_concepts):
        concepts = chunk_concepts[cid]
        # cap |S|: keep lowest-freq (most informative), tie-break on name
        s_sorted = sorted(concepts, key=lambda x: (freq[x], x))[:cap]
        did = cid_to_doc.get(cid)
        for i in range(len(s_sorted)):
            for j in range(i + 1, len(s_sorted)):
                a, b = s_sorted[i], s_sorted[j]
                pair = (a, b) if a < b else (b, a)
                cooccur[pair] = cooccur.get(pair, 0) + 1
                cooccur_docs.setdefault(pair, set()).add(did)

    G = nx.Graph()
    for c in sorted(surviving):
        G.add_node(c, freq=freq[c], doc_count=0)

    min_w = resolved["min_edge_weight"]
    for pair in sorted(cooccur):
        w = cooccur[pair]
        if w < min_w:
            continue
        a, b = pair
        G.add_edge(a, b, weight=int(w), doc_w=int(len(cooccur_docs[pair])))

    # drop isolated nodes (degree 0): they cannot carry cross-document evidence
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    # optional hub pruning
    max_deg = resolved["max_node_degree"]
    if max_deg is not None:
        hubs = [n for n in G.nodes() if G.degree(n) > int(max_deg)]
        G.remove_nodes_from(hubs)
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        G.remove_nodes_from(isolated)

    return G


# ---------------------------------------------------------------------------
# Leiden communities
# ---------------------------------------------------------------------------
def _leiden(G, resolved: dict[str, Any]) -> dict[str, int]:
    """Partition G into communities. Returns concept -> community id (int)."""
    nodes = sorted(G.nodes())
    if not nodes or G.number_of_edges() == 0:
        # degenerate: each surviving node is its own community
        return {n: i for i, n in enumerate(nodes)}

    import igraph as ig
    import leidenalg

    idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph()
    ig_g.add_vertices(len(nodes))
    edges = sorted(G.edges())
    ig_g.add_edges([(idx[a], idx[b]) for (a, b) in edges])
    weights = [int(G[a][b]["weight"]) for (a, b) in edges]

    part = leidenalg.find_partition(
        ig_g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolved["leiden_resolution"],
        seed=resolved["leiden_seed"],
        n_iterations=-1,
    )
    membership = part.membership
    return {nodes[i]: int(membership[i]) for i in range(len(nodes))}


# ---------------------------------------------------------------------------
# build_graph (THE RECIPE)
# ---------------------------------------------------------------------------
def build_graph(config_path: str = "rag.config.yaml") -> dict[str, Any]:
    """Build + persist the concept co-occurrence graph. Returns a meta dict."""
    import numpy as np
    from networkx.readwrite import json_graph

    from corpus_rag.index.embed import get_embedder

    t0 = time.time()
    cfg = _config.load(config_path)
    resolved = _resolve(cfg)

    outputs_dir = _resolve_outputs_dir(resolved["outputs_dir"], config_path)
    graph_dir = _resolve_path(resolved["path"], config_path)

    chunks = _load_chunks(outputs_dir)
    text_chunks = [c for c in chunks if c.get("kind") == "text" and (c.get("text") or "").strip()]

    concept_chunks, concept_docs, n_raw = _extract_concepts(text_chunks, resolved)
    n_concepts_raw = len(concept_chunks)

    # frequency pruning + optional cap
    min_freq = resolved["min_concept_freq"]
    surviving = {c for c, cids in concept_chunks.items() if len(cids) >= min_freq}
    if len(surviving) > resolved["max_concepts"]:
        ranked = sorted(surviving, key=lambda c: (-len(concept_chunks[c]), c))
        surviving = set(ranked[: resolved["max_concepts"]])

    G = _cooccurrence(text_chunks, concept_chunks, surviving, resolved)
    # nodes that survived isolation/hub pruning are the final concept set
    concepts_sorted = sorted(G.nodes())

    community_of = _leiden(G, resolved)

    # embeddings: one batched call, L2-normalized rows aligned to concepts_sorted
    if concepts_sorted:
        raw = get_embedder(resolved["embedder_model"]).encode(concepts_sorted)
        emb = np.asarray(raw, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
    else:
        emb = np.zeros((0, DIM), dtype=np.float32)

    # community centroids (L2-normalized mean of member embeddings), ascending id
    comm_ids = sorted(set(community_of.values())) if community_of else []
    cid_to_row = {c: i for i, c in enumerate(concepts_sorted)}
    if comm_ids:
        centroids = np.zeros((max(comm_ids) + 1, DIM), dtype=np.float32)
        for comm in comm_ids:
            members = [cid_to_row[c] for c in concepts_sorted if community_of.get(c) == comm]
            if members:
                m = emb[members].mean(axis=0)
                n = np.linalg.norm(m)
                centroids[comm] = m / n if n > 0 else m
    else:
        centroids = np.zeros((0, DIM), dtype=np.float32)

    # ---- persist ----
    graph_dir.mkdir(parents=True, exist_ok=True)

    def _wjson(name: str, obj: Any) -> None:
        (graph_dir / name).write_text(
            json.dumps(obj, ensure_ascii=True, sort_keys=False), encoding="utf-8"
        )

    _wjson("concepts.json", concepts_sorted)
    _wjson(
        "concept_chunks.json",
        {c: sorted(concept_chunks[c]) for c in concepts_sorted},
    )
    _wjson(
        "concept_stats.json",
        {
            c: {
                "freq": int(len(concept_chunks[c])),
                "doc_count": int(len(concept_docs.get(c, set()))),
                "community": int(community_of.get(c, -1)),
            }
            for c in concepts_sorted
        },
    )
    np.save(graph_dir / "embeddings.npy", emb)
    _wjson("graph.json", json_graph.node_link_data(G, edges="links"))
    _wjson(
        "communities.json",
        {
            "resolution": resolved["leiden_resolution"],
            "seed": resolved["leiden_seed"],
            "concept_to_community": {c: int(community_of[c]) for c in concepts_sorted},
        },
    )
    np.save(graph_dir / "community_centroids.npy", centroids)

    n_communities = len(comm_ids)
    largest_comm = 0
    if community_of:
        counts: dict[int, int] = {}
        for c in concepts_sorted:
            comm = community_of.get(c)
            counts[comm] = counts.get(comm, 0) + 1
        largest_comm = max(counts.values()) if counts else 0

    build_s = round(time.time() - t0, 2)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "outputs_dir": str(outputs_dir),
        "embedder_model": resolved["embedder_model"],
        "dim": DIM,
        "n_chunks": len(text_chunks),
        "n_concepts_raw": n_concepts_raw,
        "n_concepts_kept": len(concepts_sorted),
        "n_edges": G.number_of_edges(),
        "n_communities": n_communities,
        "largest_community": largest_comm,
        "build_s": build_s,
        "code_version": CODE_VERSION,
        "params": {
            "leiden_seed": resolved["leiden_seed"],
            "leiden_resolution": resolved["leiden_resolution"],
            "min_concept_freq": resolved["min_concept_freq"],
            "min_edge_weight": resolved["min_edge_weight"],
            "max_concept_tokens": resolved["max_concept_tokens"],
            "max_concepts_per_chunk": resolved["max_concepts_per_chunk"],
            "drop_generic": resolved["drop_generic"],
        },
    }
    _wjson("meta.json", meta)

    print(
        _ascii(
            f"[graph] chunks={len(text_chunks)} "
            f"concepts(raw->kept)={n_concepts_raw}->{len(concepts_sorted)} "
            f"edges={G.number_of_edges()} communities={n_communities} "
            f"largest_comm={largest_comm} build_s={build_s}; "
            f"persisted to {graph_dir}"
        )
    )
    return meta


# ---------------------------------------------------------------------------
# Artifact loader (cached by graph_dir + meta.json mtime)
# ---------------------------------------------------------------------------
def _meta_mtime(graph_dir: Path) -> float:
    mp = graph_dir / "meta.json"
    return mp.stat().st_mtime if mp.is_file() else -1.0


@lru_cache(maxsize=4)
def _load_artifacts(graph_dir_str: str, _mtime: float) -> dict[str, Any] | None:
    import numpy as np
    from networkx.readwrite import json_graph

    graph_dir = Path(graph_dir_str)
    meta_p = graph_dir / "meta.json"
    if not meta_p.is_file():
        return None
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    concepts = json.loads((graph_dir / "concepts.json").read_text(encoding="utf-8"))
    concept_chunks = json.loads((graph_dir / "concept_chunks.json").read_text(encoding="utf-8"))
    communities = json.loads((graph_dir / "communities.json").read_text(encoding="utf-8"))
    community_of = {k: int(v) for k, v in communities["concept_to_community"].items()}
    emb = np.load(graph_dir / "embeddings.npy")
    centroids = np.load(graph_dir / "community_centroids.npy")
    G = json_graph.node_link_graph(
        json.loads((graph_dir / "graph.json").read_text(encoding="utf-8")),
        edges="links",
    )
    row_of = {c: i for i, c in enumerate(concepts)}
    return {
        "meta": meta,
        "concepts": concepts,
        "row_of": row_of,
        "emb": emb,
        "concept_chunks": concept_chunks,
        "community_of": community_of,
        "centroids": centroids,
        "G": G,
    }


# ---------------------------------------------------------------------------
# SearchHit materialization
# ---------------------------------------------------------------------------
def _build_chunk_map(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """chunk_id -> chunk dict, rebuilt from chunks.jsonl (single source of truth)."""
    out: dict[str, dict[str, Any]] = {}
    for c in _load_chunks(outputs_dir):
        cid = c.get("chunk_id")
        if cid is not None:
            out[cid] = c
    return out


def _chunk_to_hit(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    """Build a SearchHit in the EXACT search._row_to_hit shape.

    KEY DIFFERENCE: section_path is ALREADY a list in chunks.jsonl -> pass it
    through directly; do NOT json.loads it.
    """
    section_path = chunk.get("section_path")
    if not isinstance(section_path, list):
        section_path = []

    page = chunk.get("page")
    try:
        page = int(page) if page is not None and int(page) >= 0 else None
    except (TypeError, ValueError):
        page = None
    artifact_id = chunk.get("artifact_id") or None

    out_chunk: dict[str, Any] = {
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "text": chunk.get("text"),
        "kind": chunk.get("kind"),
        "section_path": section_path,
    }
    if page is not None:
        out_chunk["page"] = page
    if artifact_id:
        out_chunk["artifact_id"] = artifact_id

    citation: dict[str, Any] = {
        "doc_id": chunk.get("doc_id"),
        "chunk_id": chunk.get("chunk_id"),
        "section_path": section_path,
    }
    if page is not None:
        citation["page"] = page
    if artifact_id:
        citation["artifact_id"] = artifact_id

    return {"chunk": out_chunk, "score": float(score), "citation": citation}


# ---------------------------------------------------------------------------
# graph_search (RUNTIME)
# ---------------------------------------------------------------------------
def graph_search(
    query: str,
    top_k: int | None = None,
    config_path: str = "rag.config.yaml",
) -> list[dict[str, Any]]:
    """LazyGraphRAG retrieval -> top_k SearchHit dicts. LLM-free. Graceful on no-match."""
    import numpy as np

    from corpus_rag.index.embed import get_embedder

    cfg = _config.load(config_path)
    resolved = _resolve(cfg)
    k = int(top_k) if top_k else int((cfg.get("retrieve", {}) or {}).get("top_k", 8))

    graph_dir = _resolve_path(resolved["path"], config_path)
    outputs_dir = _resolve_outputs_dir(resolved["outputs_dir"], config_path)

    def _fallback_or_empty() -> list[dict[str, Any]]:
        if resolved["fallback_hybrid"]:
            from corpus_rag.retrieve.search import search_corpus
            return search_corpus(query, top_k=k, config_path=config_path)
        return []

    art = _load_artifacts(str(graph_dir), _meta_mtime(graph_dir))
    if art is None:
        print(_ascii(f"[graph] no artifacts at {graph_dir}; run build_graph(config_path) first."))
        return _fallback_or_empty()

    meta = art["meta"]
    concepts: list[str] = art["concepts"]
    E = art["emb"]
    concept_chunks: dict[str, list] = art["concept_chunks"]
    community_of: dict[str, int] = art["community_of"]
    centroids = art["centroids"]
    G = art["G"]

    if meta.get("embedder_model") != resolved["embedder_model"]:
        print(_ascii(
            f"[graph] WARNING embedder mismatch: graph={meta.get('embedder_model')} "
            f"config={resolved['embedder_model']} (cosines may be meaningless)."
        ))

    if not concepts or E.shape[0] == 0:
        return _fallback_or_empty()

    # embed + normalize query
    qraw = get_embedder(resolved["embedder_model"]).encode([query])[0]
    q = np.asarray(qraw, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn > 0:
        q = q / qn

    # BEST-FIRST: cosine(query, concept). bge-small has a HIGH baseline cosine to
    # almost everything, so an absolute floor barely gates -> use a RELATIVE threshold
    # (mean + z*std) so only genuinely above-baseline concepts seed (this also makes
    # the no-match / fallback path reachable for off-topic queries).
    sims = E @ q  # shape (n_concepts,)
    thresh = max(resolved["seed_sim_min"], float(sims.mean()) + resolved["seed_z"] * float(sims.std()))

    seed_sim: dict[str, float] = {}
    order = sorted(range(len(concepts)), key=lambda i: (-float(sims[i]), concepts[i]))
    for i in order:
        s = float(sims[i])
        if s < thresh and seed_sim:  # always keep the single best seed; prune the tail
            break
        seed_sim[concepts[i]] = s
        if len(seed_sim) >= resolved["seed_top_n"]:
            break

    # operator override: force-add configured seed concepts present in the graph
    row_of = art["row_of"]
    for sc in resolved["seed_concepts"]:
        if sc in row_of and sc not in seed_sim:
            seed_sim[sc] = float(sims[row_of[sc]])

    if not seed_sim:
        return _fallback_or_empty()

    seeds = sorted(seed_sim, key=lambda c: (-seed_sim[c], c))

    # COMMUNITY RANKING: seed-sum + centroid-prior
    comm_score: dict[int, float] = {}
    for c in seeds:
        comm = community_of.get(c)
        if comm is not None:
            comm_score[comm] = comm_score.get(comm, 0.0) + seed_sim[c]
    prior_w = resolved["community_prior_w"]
    if centroids.shape[0] > 0 and prior_w > 0:
        cent_sims = centroids @ q  # (n_comm,)
        for comm in range(centroids.shape[0]):
            comm_score[comm] = comm_score.get(comm, 0.0) + prior_w * float(cent_sims[comm])
    ranked_comms = sorted(comm_score, key=lambda cm: (-comm_score[cm], cm))
    relevant = set(ranked_comms[: resolved["relevant_communities"]])
    # always allow communities the seeds live in (anchor)
    relevant |= {community_of.get(c) for c in seeds if community_of.get(c) is not None}

    # precompute max incident weight per node for activation propagation
    def _max_incident(u: str) -> float:
        ws = [G[u][v]["weight"] for v in G.neighbors(u)] if u in G else []
        return max(ws) if ws else 1.0

    # BREADTH-FIRST budgeted expansion
    rel: dict[str, float] = {c: seed_sim[c] for c in seeds}
    visited: set = set(seeds)
    frontier: list[str] = list(seeds)
    decay = resolved["hop_decay"]
    budget_c = resolved["budget_concepts"]

    def _candidate_chunk_count(active: set) -> int:
        seen: set = set()
        for c in active:
            seen.update(concept_chunks.get(c, []))
        return len(seen)

    for _hop in range(1, resolved["max_hops"] + 1):
        if len(visited) >= budget_c:
            break
        next_frontier: list[str] = []
        for u in sorted(frontier, key=lambda x: (-rel.get(x, 0.0), x)):
            if u not in G:
                continue
            mi = _max_incident(u)
            for v in sorted(G.neighbors(u)):
                if v in visited:
                    continue
                if community_of.get(v) not in relevant:
                    continue
                if len(visited) >= budget_c:
                    break
                w = G[u][v]["weight"]
                prop = rel[u] * (w / mi) * decay
                rel[v] = max(rel.get(v, 0.0), prop)
                visited.add(v)
                next_frontier.append(v)
            if len(visited) >= budget_c:
                break
        # chunk budget check
        if _candidate_chunk_count(visited) >= resolved["budget_chunks"]:
            break
        frontier = next_frontier
        if not frontier:
            break

    active = visited  # each c with weight rel[c]

    # GATHER + SCORE
    accum: dict[str, float] = {}
    hits_count: dict[str, int] = {}
    per_concept_chunks: dict[str, list] = {}  # for rrf
    for c in sorted(active):
        w = rel.get(c, 0.0)
        cids = concept_chunks.get(c, [])
        per_concept_chunks[c] = cids
        for cid in cids:
            accum[cid] = accum.get(cid, 0.0) + w
            hits_count[cid] = hits_count.get(cid, 0) + 1

    if not accum:
        return _fallback_or_empty()

    fusion = resolved["fusion"]
    multi_w = resolved["multi_hit_w"]

    # (A) weighted accumulation + bridging bonus
    score_a: dict[str, float] = {}
    for cid, base in accum.items():
        score_a[cid] = base * (1.0 + multi_w * math.log1p(hits_count[cid]))

    final: dict[str, float]
    if fusion == "accum":
        final = score_a
    else:
        # (B) RRF over per-SEED ranked chunk lists
        from corpus_rag.retrieve.multi import rrf

        freq_of = {c: len(concept_chunks.get(c, [])) for c in concepts}
        lists: list[list[str]] = []
        for c in seeds:
            cids = list(concept_chunks.get(c, []))
            # rank by per-seed contribution (seed_sim[c] is constant per list) then
            # chunk freq desc, tie-break chunk_id
            cids.sort(key=lambda x: (-freq_of.get(x, 0), x))
            lists.append(cids)
        ordered_ids = rrf(lists, rrf_k=resolved["rrf_k"])
        rank_score = {cid: float(len(ordered_ids) - i) for i, cid in enumerate(ordered_ids)}
        if fusion == "rrf":
            # use RRF, tie-break by (A)
            final = {cid: rank_score.get(cid, 0.0) for cid in accum}
            ordered = sorted(
                accum, key=lambda cid: (-final[cid], -score_a[cid], cid)
            )
            return _materialize(
                ordered, score_a, k, outputs_dir, meta,
                seeds, active, relevant, accum,
            )
        else:  # hybrid: min-max normalize A and B, blend
            def _minmax(d: dict[str, float], keys: list[str]) -> dict[str, float]:
                vals = [d.get(x, 0.0) for x in keys]
                lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
                rng = hi - lo
                if rng == 0:
                    return {x: 0.0 for x in keys}
                return {x: (d.get(x, 0.0) - lo) / rng for x in keys}

            keys = list(accum)
            na = _minmax(score_a, keys)
            nb = _minmax(rank_score, keys)
            alpha = resolved["alpha"]
            final = {cid: alpha * na[cid] + (1 - alpha) * nb[cid] for cid in keys}

    ordered = sorted(final, key=lambda cid: (-final[cid], cid))
    return _materialize(
        ordered, final, k, outputs_dir, meta, seeds, active, relevant, accum
    )


def _materialize(
    ordered_ids: list[str],
    score_map: dict[str, float],
    k: int,
    outputs_dir: Path,
    meta: dict[str, Any],
    seeds: list[str],
    active: set,
    relevant: set,
    accum: dict[str, float],
) -> list[dict[str, Any]]:
    """Load chunk text/citation from chunks.jsonl and build top_k SearchHits."""
    chunk_map = _build_chunk_map(outputs_dir)
    if meta.get("n_chunks") is not None:
        cur_text = sum(1 for c in chunk_map.values() if c.get("kind") == "text")
        if cur_text != meta["n_chunks"]:
            print(_ascii(
                f"[graph] WARNING stale graph: built on n_chunks={meta['n_chunks']} "
                f"but corpus now has {cur_text} text chunks."
            ))

    hits: list[dict[str, Any]] = []
    for cid in ordered_ids:
        if len(hits) >= k:
            break
        chunk = chunk_map.get(cid)
        if chunk is None:  # corpus drift -> skip, do not crash
            continue
        hits.append(_chunk_to_hit(chunk, round(float(score_map.get(cid, 0.0)), 6)))

    if hits:
        hits[0]["_graph"] = {
            "n_seeds": len(seeds),
            "n_active_concepts": len(active),
            "n_communities_searched": len(relevant),
            "n_candidates": len(accum),
        }
    return hits
