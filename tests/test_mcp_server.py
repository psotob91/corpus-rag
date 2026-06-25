"""Thorough tests for the MCP corpus tools (make_tools + build_app)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("fastembed")

from corpus_rag.index.build import build_index  # noqa: E402
from corpus_rag.server.mcp_server import build_app, make_tools  # noqa: E402

CHUNKS = [
    {"chunk_id": "d1-c1", "doc_id": "d1", "kind": "text",
     "text": "We fit a Cox proportional hazards model for survival.",
     "section_path": ["Methods"], "page": 1},
    {"chunk_id": "d1-c2", "doc_id": "d1", "kind": "text",
     "text": "Calibration was assessed with the calibration slope and the c-statistic.",
     "section_path": ["Results"], "page": 2},
    {"chunk_id": "d1-t1", "doc_id": "d1", "kind": "table", "artifact_id": "d1-t1",
     "text": "Table 1. Adjusted hazard ratios by arm.", "section_path": ["Results"], "page": 3},
]

CONFIG = """\
corpus: {outputs_dir: outputs}
index: {path: .rag/test.lance, embedder: {model: BAAI/bge-small-en-v1.5}}
retrieve: {rrf_k: 60, top_k: 8, routing: {enabled: true, global_top_k: 6, global_candidate_k: 12}}
"""


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    doc = tmp_path / "outputs" / "d1"
    (doc / "tables").mkdir(parents=True)
    (doc / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
    (doc / "document.md").write_text("# Methods\n\nCox model.\n", encoding="utf-8")
    (doc / "document.meta.json").write_text(
        json.dumps({"id": "d1", "source": "pdf-native", "n_chunks": 3,
                    "n_tables": 1, "n_figures": 0, "n_formulas": 0}), encoding="utf-8")
    (doc / "tables" / "d1-t1.table.json").write_text(
        json.dumps({"id": "d1-t1", "caption": "Adjusted hazard ratios",
                    "cells": [["arm", "HR"], ["A", "0.62"]]}), encoding="utf-8")
    cfg = tmp_path / "rag.config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    build_index(str(cfg))
    return cfg


def test_search_single_multi_and_subqueries(corpus: Path):
    search = make_tools(str(corpus))["search_corpus"]
    single = search("hazard ratio Cox model")
    assert single and all("chunk" in h and "citation" in h for h in single)

    multi = search("compare and summarize calibration across the studies", multi_query=True)
    assert multi and all("chunk" in h for h in multi)

    agent = search("calibration", sub_queries=["calibration slope", "c-statistic"])
    assert agent and all("chunk" in h for h in agent)


def test_list_documents(corpus: Path):
    docs = make_tools(str(corpus))["list_documents"]()
    assert docs and docs[0]["id"] == "d1"
    assert docs[0]["n_chunks"] == 3 and docs[0]["source"] == "pdf-native"


def test_get_document_success_and_error(corpus: Path):
    tools = make_tools(str(corpus))
    assert "Methods" in tools["get_document"]("d1")
    assert "error" in tools["get_document"]("missing").lower()


def test_get_table_success_and_errors(corpus: Path):
    tools = make_tools(str(corpus))
    assert tools["get_table"]("d1-t1").get("caption") == "Adjusted hazard ratios"
    assert "error" in tools["get_table"]("d1-f1")   # kind mismatch (figure id to get_table)
    assert "error" in tools["get_table"]("d1-t9")   # file not found
    assert "error" in tools["get_figure"]("not-an-id")  # unparseable id


def test_build_app_registers_all_tools(corpus: Path):
    app = build_app(str(corpus))
    names = {t.name for t in asyncio.run(app.list_tools())}
    assert {"search_corpus", "list_documents", "get_document",
            "get_table", "get_figure", "get_formula"} <= names
