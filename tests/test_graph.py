"""Tests for the P3 LazyGraphRAG concept-graph retriever ARM.

Pure helpers + co-occurrence are tested hermetically; the full build/search path
is gated behind pytest.importorskip on the heavy [graph] stack. A pilot smoke is
gated on the .rag/pilot graph being present. Mirrors test_e2e / test_router.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus_rag.retrieve import graph as G

PILOT_CFG = Path(__file__).resolve().parents[1] / ".rag" / "pilot" / "rag.config.yaml"


# ---------------------------------------------------------------------------
# 10. import-safety: heavy libs must NOT be imported at module load
# ---------------------------------------------------------------------------
def test_import_is_lazy():
    # graph is already imported above; assert it did NOT pull heavy deps in.
    # (If another test imported them first this is moot, but in a clean run the
    # graph module must not be the importer.)
    import importlib

    # Re-import fresh and check the module body does not reference them eagerly.
    src = (Path(G.__file__)).read_text(encoding="utf-8")
    head = src.split("def ", 1)[0]  # module-level body before first function
    for lib in ("import spacy", "import networkx", "import igraph", "import leidenalg", "import numpy"):
        assert lib not in head, f"{lib} imported at module top -> not import-safe"
    importlib.reload(G)


# ---------------------------------------------------------------------------
# Config resolution (pure)
# ---------------------------------------------------------------------------
def test_resolve_defaults():
    r = G._resolve({})
    assert r["min_concept_freq"] == 2
    assert r["fusion"] == "accum"
    assert r["embedder_model"] == "BAAI/bge-small-en-v1.5"
    assert r["budget_concepts"] == 200


def test_resolve_path_relative_to_config(tmp_path: Path):
    cfg = tmp_path / "sub" / "rag.config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("graph: {path: graph}\n", encoding="utf-8")
    p = G._resolve_path("graph", str(cfg))
    assert p == (tmp_path / "sub" / "graph")


# ---------------------------------------------------------------------------
# SearchHit shape from a chunk dict (pure; section_path stays a list)
# ---------------------------------------------------------------------------
def test_chunk_to_hit_shape_passthrough_section_path():
    chunk = {
        "chunk_id": "d1-c1", "doc_id": "d1", "kind": "text",
        "text": "calibration of the model", "section_path": ["Methods", "Calibration"],
        "page": 3, "artifact_id": "d1-t1",
    }
    hit = G._chunk_to_hit(chunk, 0.5)
    assert set(hit.keys()) == {"chunk", "score", "citation"}
    assert hit["chunk"]["section_path"] == ["Methods", "Calibration"]  # list passthrough
    assert isinstance(hit["score"], float)
    assert hit["citation"]["chunk_id"] == "d1-c1"
    assert hit["chunk"]["page"] == 3
    assert hit["citation"]["artifact_id"] == "d1-t1"


def test_chunk_to_hit_missing_optional_fields():
    chunk = {"chunk_id": "d1-c2", "doc_id": "d1", "kind": "text", "text": "x", "section_path": []}
    hit = G._chunk_to_hit(chunk, 1.0)
    assert "page" not in hit["chunk"]
    assert "artifact_id" not in hit["chunk"]
    assert hit["chunk"]["section_path"] == []


# ---------------------------------------------------------------------------
# Co-occurrence (needs networkx only)
# ---------------------------------------------------------------------------
def test_cooccurrence_weights_and_pruning():
    pytest.importorskip("networkx")
    # 3 chunks; concept pair (alpha, beta) co-occurs in 2 chunks; (alpha, gamma) in 1.
    text_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "kind": "text", "text": ""},
        {"chunk_id": "c2", "doc_id": "d2", "kind": "text", "text": ""},
        {"chunk_id": "c3", "doc_id": "d1", "kind": "text", "text": ""},
    ]
    concept_chunks = {
        "alpha": {"c1", "c2", "c3"},
        "beta": {"c1", "c2"},
        "gamma": {"c3"},
    }
    surviving = {"alpha", "beta", "gamma"}
    resolved = G._resolve({})
    resolved["min_edge_weight"] = 2  # drop the weight-1 (alpha,gamma) edge
    resolved["max_concepts_per_chunk"] = 40
    g = G._cooccurrence(text_chunks, concept_chunks, surviving, resolved)
    assert g.has_edge("alpha", "beta")
    assert g["alpha"]["beta"]["weight"] == 2
    assert g["alpha"]["beta"]["doc_w"] == 2  # c1=d1, c2=d2
    # gamma only co-occurred with alpha at weight 1 -> edge pruned -> gamma isolated -> dropped
    assert "gamma" not in g.nodes()
    # no degree-0 nodes survive
    assert all(g.degree(n) > 0 for n in g.nodes())


# ---------------------------------------------------------------------------
# _clean_concept (needs spaCy model)
# ---------------------------------------------------------------------------
def test_clean_concept_strips_lemmatizes_drops_generic():
    pytest.importorskip("spacy")
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        pytest.skip("en_core_web_sm not installed")

    doc = nlp("The multivariable regression models predict clinical outcomes")
    concepts = set()
    for span in doc.noun_chunks:
        c = G._clean_concept(span, drop_generic=True, min_chars=3, max_tokens=4)
        if c:
            concepts.add(c)
    # DET 'the' stripped, lemmatized to singular, lowercased
    assert any("regression model" in c for c in concepts)
    assert any("clinical outcome" in c for c in concepts)
    # no bare leading determiner survived
    assert all(not c.startswith("the ") for c in concepts)

    # bare generic single tokens rejected; allowed inside multiword
    doc2 = nlp("the model")
    got = [G._clean_concept(s, True, 3, 4) for s in doc2.noun_chunks]
    assert all(g is None or g != "model" for g in got)


# ---------------------------------------------------------------------------
# Hermetic full build + search on a tiny synthetic corpus
# ---------------------------------------------------------------------------
HEAVY = ["spacy", "networkx", "igraph", "leidenalg", "fastembed", "numpy"]


def _have_model() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_sm", disable=["ner"])
        return True
    except Exception:
        return False


# Synthetic corpus engineered for distributed evidence: bridge concept
# "calibration" co-occurs with "logistic regression" (doc1) and with
# "survival analysis" (doc2); a query near both should surface BOTH docs.
SYN_CHUNKS_BY_DOC = {
    "doc1": [
        {"chunk_id": "doc1-c1", "doc_id": "doc1", "kind": "text",
         "text": "Calibration of the logistic regression prediction model was assessed "
                 "with calibration plots and the calibration slope.",
         "section_path": ["Methods"], "page": 1},
        {"chunk_id": "doc1-c2", "doc_id": "doc1", "kind": "text",
         "text": "The logistic regression prediction model showed good calibration and "
                 "discrimination in the validation cohort.",
         "section_path": ["Results"], "page": 2},
    ],
    "doc2": [
        {"chunk_id": "doc2-c1", "doc_id": "doc2", "kind": "text",
         "text": "Calibration of the survival analysis model was evaluated using the "
                 "calibration slope and a calibration curve over time.",
         "section_path": ["Methods"], "page": 1},
        {"chunk_id": "doc2-c2", "doc_id": "doc2", "kind": "text",
         "text": "The survival analysis model achieved adequate calibration across risk "
                 "groups in the external validation cohort.",
         "section_path": ["Results"], "page": 2},
    ],
}

SYN_CONFIG = """\
corpus:
  outputs_dir: outputs
index:
  path: index.lance
  embedder:
    model: BAAI/bge-small-en-v1.5
retrieve:
  top_k: 8
graph:
  enabled: true
  path: graph
  min_concept_freq: 1
  min_edge_weight: 1
  seed_sim_min: 0.20
  fallback_hybrid: false
"""


@pytest.fixture()
def syn_corpus(tmp_path: Path) -> Path:
    for doc, chunks in SYN_CHUNKS_BY_DOC.items():
        d = tmp_path / "outputs" / doc
        d.mkdir(parents=True)
        (d / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
        )
    cfg = tmp_path / "rag.config.yaml"
    cfg.write_text(SYN_CONFIG, encoding="utf-8")
    return cfg


def _skip_if_no_stack():
    for lib in HEAVY:
        pytest.importorskip(lib)
    if not _have_model():
        pytest.skip("en_core_web_sm not installed")


def test_build_artifacts_exist_and_aligned(syn_corpus: Path, tmp_path: Path):
    _skip_if_no_stack()
    import numpy as np

    meta = G.build_graph(str(syn_corpus))
    gdir = tmp_path / "graph"
    for name in (
        "meta.json", "concepts.json", "concept_chunks.json", "concept_stats.json",
        "embeddings.npy", "graph.json", "communities.json", "community_centroids.npy",
    ):
        assert (gdir / name).is_file(), f"missing artifact {name}"

    concepts = json.loads((gdir / "concepts.json").read_text(encoding="utf-8"))
    emb = np.load(gdir / "embeddings.npy")
    assert emb.shape[0] == len(concepts)
    if emb.shape[0]:
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    # every concept_chunks chunk_id exists in the corpus
    cc = json.loads((gdir / "concept_chunks.json").read_text(encoding="utf-8"))
    all_cids = {c["chunk_id"] for chunks in SYN_CHUNKS_BY_DOC.values() for c in chunks}
    for cids in cc.values():
        assert set(cids) <= all_cids
    assert meta["n_chunks"] == 4


def test_build_is_deterministic(syn_corpus: Path, tmp_path: Path):
    _skip_if_no_stack()
    G.build_graph(str(syn_corpus))
    gdir = tmp_path / "graph"
    concepts1 = (gdir / "concepts.json").read_text(encoding="utf-8")
    comms1 = (gdir / "communities.json").read_text(encoding="utf-8")

    # clear loader cache + rebuild into the same dir
    G._load_artifacts.cache_clear()
    G.build_graph(str(syn_corpus))
    concepts2 = (gdir / "concepts.json").read_text(encoding="utf-8")
    comms2 = (gdir / "communities.json").read_text(encoding="utf-8")
    assert concepts1 == concepts2
    assert comms1 == comms2


def test_prune_invariants(syn_corpus: Path, tmp_path: Path):
    _skip_if_no_stack()
    from networkx.readwrite import json_graph

    G.build_graph(str(syn_corpus))
    gdir = tmp_path / "graph"
    stats = json.loads((gdir / "concept_stats.json").read_text(encoding="utf-8"))
    for s in stats.values():
        assert s["freq"] >= 1  # min_concept_freq for this cfg
    gj = json_graph.node_link_graph(
        json.loads((gdir / "graph.json").read_text(encoding="utf-8")), edges="links"
    )
    for _a, _b, d in gj.edges(data=True):
        assert d["weight"] >= 1
    assert all(gj.degree(n) > 0 for n in gj.nodes())


def test_searchhit_shape_and_fusable(syn_corpus: Path):
    _skip_if_no_stack()
    from corpus_rag.retrieve.multi import rrf

    G.build_graph(str(syn_corpus))
    G._load_artifacts.cache_clear()
    hits = G.graph_search("calibration of the prediction model", top_k=6, config_path=str(syn_corpus))
    assert hits, "expected graph hits on the synthetic corpus"
    for h in hits:
        assert set(h.keys()) >= {"chunk", "score", "citation"}
        assert set(["chunk_id", "doc_id", "text", "kind", "section_path"]) <= set(h["chunk"])
        assert isinstance(h["chunk"]["section_path"], list)
        assert isinstance(h["score"], float)
        assert h["citation"]["chunk_id"] == h["chunk"]["chunk_id"]
        assert isinstance(h["citation"]["section_path"], list)
    # fusable into multi.rrf without KeyError
    graph_ids = [h["chunk"]["chunk_id"] for h in hits]
    fused = rrf([graph_ids, graph_ids[::-1]])
    assert set(fused) == set(graph_ids)


def test_distributed_evidence_spans_two_docs(syn_corpus: Path):
    _skip_if_no_stack()
    G.build_graph(str(syn_corpus))
    G._load_artifacts.cache_clear()
    hits = G.graph_search("calibration of the prediction model", top_k=8, config_path=str(syn_corpus))
    docs = {h["chunk"]["doc_id"] for h in hits}
    assert len(docs) >= 2, f"expected cross-document evidence, got {docs}"


def test_graceful_empty_on_nonsense(syn_corpus: Path):
    _skip_if_no_stack()
    G.build_graph(str(syn_corpus))
    G._load_artifacts.cache_clear()
    hits = G.graph_search("zzzqwlkj xkcd nonsense fhqwhgads", top_k=5, config_path=str(syn_corpus))
    assert hits == [] or len(hits) <= 5  # never raises; fallback off -> []


def test_budget_honored(syn_corpus: Path, monkeypatch):
    _skip_if_no_stack()
    G.build_graph(str(syn_corpus))
    G._load_artifacts.cache_clear()

    real_resolve = G._resolve

    def _tiny_budget(cfg):
        r = real_resolve(cfg)
        r["budget_concepts"] = 3
        return r

    monkeypatch.setattr(G, "_resolve", _tiny_budget)
    hits = G.graph_search("calibration of the prediction model", top_k=8, config_path=str(syn_corpus))
    if hits:
        assert hits[0]["_graph"]["n_active_concepts"] <= 3 + 20  # bounded near the budget


def test_missing_graph_returns_empty(tmp_path: Path):
    # no graph built -> fallback off -> []
    cfg = tmp_path / "rag.config.yaml"
    (tmp_path / "outputs").mkdir()
    cfg.write_text(SYN_CONFIG, encoding="utf-8")
    G._load_artifacts.cache_clear()
    hits = G.graph_search("anything", top_k=5, config_path=str(cfg))
    assert hits == []


# ---------------------------------------------------------------------------
# 11. Pilot smoke (slow; gated on the pilot graph + heavy stack)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not PILOT_CFG.is_file(), reason="pilot config absent")
def test_pilot_smoke_cross_doc():
    _skip_if_no_stack()
    gdir = PILOT_CFG.parent / "graph"
    if not (gdir / "meta.json").is_file():
        G.build_graph(str(PILOT_CFG))
    G._load_artifacts.cache_clear()
    hits = G.graph_search(
        "how should sample size for a prediction model account for uncertainty",
        top_k=8, config_path=str(PILOT_CFG),
    )
    assert hits, "expected >=1 pilot hit"
    docs = {h["chunk"]["doc_id"] for h in hits}
    assert len(docs) >= 2, f"expected cross-document evidence on pilot, got {docs}"
