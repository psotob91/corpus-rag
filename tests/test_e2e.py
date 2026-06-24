"""Hermetic end-to-end test for the retrieval pipeline.

Builds a tiny index from a temp OUTPUT_LAYOUT corpus and searches it — exercising
the real A-pipeline (build_index -> LanceDB hybrid -> search_corpus) with NO
network and NO real PDF. Skips cleanly if the heavy stack is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("fastembed")

from corpus_rag.index.build import build_index  # noqa: E402
from corpus_rag.retrieve.search import search_corpus  # noqa: E402

CHUNKS = [
    {
        "chunk_id": "d1-c1", "doc_id": "d1", "kind": "text",
        "text": "We fit a Cox proportional hazards model to estimate the hazard ratio for mortality.",
        "section_path": ["Methods", "Cox model"], "page": 2,
    },
    {
        "chunk_id": "d1-c2", "doc_id": "d1", "kind": "text",
        "text": "Survival was summarized with the Kaplan-Meier estimator and compared by log-rank test.",
        "section_path": ["Methods", "Kaplan-Meier"], "page": 2,
    },
    {
        "chunk_id": "d1-t1", "doc_id": "d1", "kind": "table", "artifact_id": "d1-t1",
        "text": "Table 1. Adjusted hazard ratios by treatment arm.",
        "section_path": ["Results"], "page": 3,
    },
]

CONFIG = """\
corpus:
  outputs_dir: outputs
index:
  path: .rag/test.lance
  embedder:
    provider: fastembed
    model: BAAI/bge-small-en-v1.5
retrieve:
  mode: hybrid
  rrf_k: 60
  top_k: 8
  rerank:
    enabled: false
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    doc = tmp_path / "outputs" / "d1"
    doc.mkdir(parents=True)
    (doc / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8"
    )
    cfg = tmp_path / "rag.config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return cfg


def test_build_and_search(corpus: Path):
    summary = build_index(str(corpus))
    assert summary["n_chunks"] == 3
    assert summary["n_docs"] == 1

    hits = search_corpus("hazard ratio Cox model", top_k=3, config_path=str(corpus))
    assert hits, "expected at least one hit"
    assert all(h["chunk"]["doc_id"] == "d1" for h in hits)
    top = hits[0]
    assert top["citation"]["chunk_id"] == top["chunk"]["chunk_id"]
    assert isinstance(top["score"], float)
    assert isinstance(top["chunk"]["section_path"], list)  # JSON round-trips back to a list


def test_table_chunk_retrievable_with_artifact_id(corpus: Path):
    build_index(str(corpus))
    hits = search_corpus(
        "adjusted hazard ratios table treatment arm", top_k=3, config_path=str(corpus)
    )
    by_id = {h["chunk"]["chunk_id"]: h for h in hits}
    assert "d1-t1" in by_id, f"table chunk not retrieved; got {sorted(by_id)}"
    assert by_id["d1-t1"]["citation"].get("artifact_id") == "d1-t1"
