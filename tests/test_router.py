"""Tests for the P2 query router (lexical helper is pure; exemplar path needs fastembed)."""

from __future__ import annotations

import pytest

from corpus_rag.retrieve.router import _lexical_scope, classify_query


def test_lexical_detects_global_markers():
    assert _lexical_scope("compare bootstrap and cross-validation across all studies")[0] == "global"
    assert _lexical_scope("summarize the overall consensus")[0] == "global"


def test_lexical_defaults_to_local():
    assert _lexical_scope("the value of the c-statistic")[0] == "local"   # 'value of' marker
    assert _lexical_scope("hello there")[0] == "local"                    # tie -> fast path


def test_classify_query_embedding_paths():
    pytest.importorskip("fastembed")
    g = classify_query("compare and synthesize the methods across these papers")
    assert g["scope"] == "global"
    loc = classify_query("what is the value of the .632 estimator in Efron 1983")
    assert loc["scope"] == "local"
    assert g["method"] in ("exemplar", "lexical") and "score" in g
