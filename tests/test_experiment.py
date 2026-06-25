"""Pure-function tests for the retrieval experiment helpers (no index needed)."""

from __future__ import annotations

from corpus_rag.eval.experiment import _rrf, _subqueries


def test_rrf_ranks_consistently_top_id_first():
    out = _rrf([["a", "b", "c"], ["a", "c", "b"]])  # 'a' is top in both lists
    assert out[0] == "a"
    assert set(out) == {"a", "b", "c"}


def test_subqueries_keep_original_and_extract_core_concept():
    subs = _subqueries("compare and summarize across the studies: calibration", n_sub=4)
    assert subs[0].startswith("compare")     # the original query is kept first
    assert "calibration" in subs             # the core concept becomes a sub-query
    assert "compare" not in subs[1:]         # synthesis scaffolding is stripped
    assert "studies" not in subs[1:]
