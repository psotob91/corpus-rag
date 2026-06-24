"""Contract tests — stdlib only, so they pass in the scaffold phase (no deps)."""

from __future__ import annotations

import pytest

from corpus_rag.contracts import (
    ContractError,
    validate_chunk,
    validate_intermediate,
)


def test_intermediate_ok():
    d = {"document": "snell2023", "markdown": "# x", "tables": [], "figures": [], "formulas": []}
    assert validate_intermediate(d)["document"] == "snell2023"


def test_intermediate_missing_key():
    with pytest.raises(ContractError):
        validate_intermediate({"document": "x", "markdown": "y"})  # missing tables/figures/formulas


def test_chunk_text_ok():
    c = {"chunk_id": "doc-c1", "doc_id": "doc", "kind": "text", "text": "hello"}
    assert validate_chunk(c)["chunk_id"] == "doc-c1"


def test_chunk_text_without_text_fails():
    with pytest.raises(ContractError):
        validate_chunk({"chunk_id": "doc-c1", "doc_id": "doc", "kind": "text", "text": ""})


def test_chunk_table_requires_artifact_id():
    with pytest.raises(ContractError):
        validate_chunk({"chunk_id": "doc-c2", "doc_id": "doc", "kind": "table"})
    ok = {"chunk_id": "doc-c2", "doc_id": "doc", "kind": "table", "artifact_id": "doc-t1"}
    assert validate_chunk(ok)["artifact_id"] == "doc-t1"


def test_chunk_bad_kind():
    with pytest.raises(ContractError):
        validate_chunk({"chunk_id": "doc-c3", "doc_id": "doc", "kind": "weird"})
