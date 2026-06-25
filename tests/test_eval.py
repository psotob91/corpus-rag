"""Hermetic test for the BenchmarkQED-lite eval (no network, no real PDF)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("fastembed")

from corpus_rag.eval.benchmark import DECISIONS, decide, run_eval  # noqa: E402
from corpus_rag.index.build import build_index  # noqa: E402

# 'warfarin' is cross-cutting (2 docs / 2 sections) -> a global probe; each chunk
# has distinct salient terms -> local probes.
CHUNKS = [
    {"chunk_id": "d1-c1", "doc_id": "d1", "kind": "text",
     "text": "Warfarin anticoagulation increased major bleeding risk in elderly patients.",
     "section_path": ["Results", "Bleeding"], "page": 1},
    {"chunk_id": "d1-c2", "doc_id": "d1", "kind": "text",
     "text": "Kaplan Meier survival curves diverged early between the two treatment arms.",
     "section_path": ["Results", "Survival"], "page": 2},
    {"chunk_id": "d2-c1", "doc_id": "d2", "kind": "text",
     "text": "Warfarin dosing required frequent INR monitoring and careful dose titration.",
     "section_path": ["Methods", "Dosing"], "page": 1},
    {"chunk_id": "d2-c2", "doc_id": "d2", "kind": "text",
     "text": "Cox proportional hazards regression adjusted for age and comorbidity burden.",
     "section_path": ["Methods", "Model"], "page": 2},
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
eval:
  top_k: 8
  n_local: 30
  thresholds:
    local_recall_ok: 0.6
    global_recall_ok: 0.7
    gap: 0.25
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    sources = {"d1": "pdf-native", "d2": "pdf-scan-ocr"}  # one native, one scanned
    for doc in ("d1", "d2"):
        d = tmp_path / "outputs" / doc
        d.mkdir(parents=True)
        rows = [c for c in CHUNKS if c["doc_id"] == doc]
        (d / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in rows), encoding="utf-8"
        )
        (d / "document.meta.json").write_text(
            json.dumps({"id": doc, "source": sources[doc]}), encoding="utf-8"
        )
    cfg = tmp_path / "rag.config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return cfg


def test_eval_runs_and_decides(corpus: Path):
    build_index(str(corpus))
    res = run_eval(str(corpus))

    assert res["n_probes"] > 0
    assert res["local"]["n"] >= 1
    assert res["global"]["n"] >= 1, "expected a cross-cutting ('warfarin') global probe"
    # metrics are well-formed fractions
    for scope in ("local", "global"):
        assert 0.0 <= res[scope]["recall_at_k"] <= 1.0

    d = decide(res)
    assert d["decision"] in DECISIONS
    assert d["rationale"]


def test_local_retrieval_is_healthy(corpus: Path):
    # On a tiny clean corpus each chunk's own keywords should retrieve it.
    build_index(str(corpus))
    res = run_eval(str(corpus))
    assert res["local"]["recall_at_k"] >= 0.5


def test_eval_stratifies_by_doc_class(corpus: Path):
    build_index(str(corpus))
    res = run_eval(str(corpus))
    assert res["n_docs_native"] == 1 and res["n_docs_scan"] == 1
    strata = res["strata"]
    assert strata["local"]["native"]["n"] >= 1
    assert strata["local"]["scan"]["n"] >= 1
    # the cross-cutting 'warfarin' probe spans d1(native)+d2(scan) -> any-scan => scan
    assert strata["global"]["scan"]["n"] >= 1
    assert strata["global"]["native"]["n"] == 0
    assert all("doc_class" in p and "n_scan_gold" in p for p in res["per_probe"])
