"""Hermetic tests for the docling -> Intermediate mapping.

These exercise the mapping logic (heading grouping, formula extraction) with a
FAKE duck-typed docling document, so they run WITHOUT docling/torch installed and
without the heavy models. The full conversion is validated manually / by the
heavy integration run, not in the unit suite.
"""

from __future__ import annotations

from corpus_rag.ingest.docling_pipeline import _formulas_from_doc, _sections_from_doc


class _Prov:
    def __init__(self, page_no: int):
        self.page_no = page_no


class _Item:
    def __init__(self, label: str, text: str = "", level: int | None = None, page: int = 1):
        self.label = label
        self.text = text
        if level is not None:
            self.level = level
        self.prov = [_Prov(page)]


class _Doc:
    def __init__(self, items: list[_Item]):
        self._items = items

    def iterate_items(self):
        for it in self._items:
            yield it, 0


def test_sections_group_body_under_headers():
    doc = _Doc([
        _Item("section_header", "Methods", level=1, page=2),
        _Item("text", "We enrolled 200 patients.", page=2),
        _Item("list_item", "Inclusion: adults.", page=2),
        _Item("section_header", "Results", level=1, page=3),
        _Item("text", "Survival improved in the treatment arm.", page=3),
    ])
    secs = _sections_from_doc(doc)
    by_title = {s["title"]: s for s in secs if s["title"]}
    assert "Methods" in by_title and "Results" in by_title
    assert "enrolled 200 patients" in by_title["Methods"]["body"]
    assert "Inclusion" in by_title["Methods"]["body"]
    assert by_title["Methods"]["page"] == 2
    assert by_title["Results"]["section_path"][-1] == "Results"


def test_section_nesting_by_level():
    doc = _Doc([
        _Item("section_header", "Methods", level=1, page=1),
        _Item("section_header", "Patients", level=2, page=1),
        _Item("text", "Cohort description.", page=1),
    ])
    secs = _sections_from_doc(doc)
    patients = next(s for s in secs if s["title"] == "Patients")
    assert patients["section_path"] == ["Methods", "Patients"]


def test_formula_extraction_fills_latex():
    doc = _Doc([
        _Item("formula", "E = mc^2", page=1),
        _Item("text", "not a formula", page=1),
        _Item("formula", "\\alpha + \\beta", page=2),
    ])
    formulas = _formulas_from_doc(doc, "doc")
    assert [f["id"] for f in formulas] == ["doc-e1", "doc-e2"]
    assert formulas[0]["latex"] == "E = mc^2"
    assert formulas[0]["source"] == "docling"
    assert formulas[1]["page"] == 2
