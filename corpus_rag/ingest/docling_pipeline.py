"""docling structural ingestion (P1.5): better headings, tables, and formula->LaTeX.

Optional HEAVY route (`uv sync --extra docling`; pulls torch + layout/table/formula
models). Produces the SAME Intermediate shape as the lean PyMuPDF route, so
`write_product` is reused unchanged. The win over the lean route: real layout
analysis (reliable heading nesting), TableFormer structure, and formula
enrichment that fills `latex` (the lean route leaves it empty).

ASCII-only output. Defensive about docling's object API so minor version
differences degrade gracefully instead of crashing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from corpus_rag.contracts import Intermediate, validate_intermediate
from corpus_rag.ingest.pipeline import slug_from_path


def _page_no(item: Any) -> int:
    try:
        prov = getattr(item, "prov", None) or []
        if prov:
            return int(getattr(prov[0], "page_no", 1))
    except Exception:
        pass
    return 1


def _label(item: Any) -> str:
    return str(getattr(item, "label", "")).lower()


def _text(item: Any) -> str:
    return (getattr(item, "text", "") or "").strip()


def _converter(do_formula: bool):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    if do_formula:
        try:
            opts.do_formula_enrichment = True
        except Exception:
            pass
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _sections_from_doc(doc: Any) -> list[dict]:
    """Walk the document tree, grouping body text under section headers.

    docling's layout model gives reliable heading levels, so section_path nests
    correctly (the lean route only guesses from font size).
    """
    sections: list[dict] = []
    path_stack: list[tuple[int, str]] = []
    current = {"level": 0, "title": "", "page": 1, "section_path": [], "_body": []}
    first_body_page_set = False

    def _close(sec: dict) -> None:
        sec["body"] = "\n\n".join(sec.pop("_body")).strip()
        sections.append(sec)

    for item, _lvl in doc.iterate_items():
        lab = _label(item)
        text = _text(item)
        is_head = ("section_header" in lab) or (lab == "title")
        if is_head and text:
            _close(current)
            level = 1
            if "section_header" in lab:
                try:
                    level = max(1, int(getattr(item, "level", 1) or 1))
                except Exception:
                    level = 1
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, text))
            current = {
                "level": level, "title": text, "page": _page_no(item),
                "section_path": [t for _, t in path_stack], "_body": [],
            }
            first_body_page_set = True
        elif lab in ("text", "paragraph", "list_item", "caption", "code", "footnote") and text:
            current["_body"].append(text)
            if not first_body_page_set:
                current["page"] = _page_no(item)
                first_body_page_set = True

    current["body"] = "\n\n".join(current.pop("_body")).strip()
    sections.append(current)
    return [s for s in sections if s.get("title") or s.get("body")]


def _tables_from_doc(doc: Any, slug: str) -> list[dict]:
    out: list[dict] = []
    for i, tbl in enumerate(getattr(doc, "tables", []) or [], 1):
        cells: list[list[str]] = []
        try:
            try:
                df = tbl.export_to_dataframe(doc)  # newer docling wants the doc
            except TypeError:
                df = tbl.export_to_dataframe()
            header = [str(c) for c in df.columns.tolist()]
            body = [[("" if v is None else str(v)) for v in row] for row in df.values.tolist()]
            cells = [header] + body
        except Exception:
            data = getattr(tbl, "data", None)
            grid = getattr(data, "grid", None)
            if grid:
                cells = [[str(getattr(c, "text", "") or "") for c in row] for row in grid]
        caption = ""
        try:
            caption = (tbl.caption_text(doc) or "").strip()
        except Exception:
            pass
        out.append({
            "id": f"{slug}-t{i}", "page": _page_no(tbl),
            "n_rows": len(cells), "n_cols": len(cells[0]) if cells else 0,
            "cells": cells, "caption": caption,
        })
    return out


def _figures_from_doc(doc: Any, slug: str) -> list[dict]:
    out: list[dict] = []
    for i, pic in enumerate(getattr(doc, "pictures", []) or [], 1):
        caption = ""
        try:
            caption = (pic.caption_text(doc) or "").strip()
        except Exception:
            pass
        out.append({
            "id": f"{slug}-f{i}", "page": _page_no(pic),
            "has_caption": bool(caption), "caption": caption,
        })
    return out


def _formulas_from_doc(doc: Any, slug: str) -> list[dict]:
    out: list[dict] = []
    i = 0
    for item, _lvl in doc.iterate_items():
        if "formula" in _label(item):
            latex = _text(item)
            i += 1
            out.append({
                "id": f"{slug}-e{i}", "page": _page_no(item),
                "latex": latex, "source": "docling", "text": latex,
            })
    return out


def ingest_docling(pdf_path: str, do_formula: bool = True) -> Intermediate:
    """Ingest one PDF into a validated Intermediate via docling (structural route)."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"pdf not found: {path}")

    slug = slug_from_path(pdf_path)
    conv = _converter(do_formula)
    doc = conv.convert(str(path)).document

    sections = _sections_from_doc(doc)
    tables = _tables_from_doc(doc, slug)
    figures = _figures_from_doc(doc, slug)
    formulas = _formulas_from_doc(doc, slug)
    markdown = doc.export_to_markdown()

    try:  # cheap deterministic traits (reuse the lean classifier)
        from corpus_rag.ingest.classify import classify
        traits = classify(str(path))
    except Exception:
        traits = {"source_class": "pdf-native", "route": "docling"}

    intermediate: dict = {
        "document": slug,
        "markdown": markdown,
        "sections": sections,
        "tables": tables,
        "figures": figures,
        "formulas": formulas,
        "n_tables": len(tables),
        "n_figures": len(figures),
        "n_formulas": len(formulas),
        "engine": "docling",
        "device": "cpu",
        "source_class": traits.get("source_class", "pdf-native"),
        "route": "docling",
        "traits": traits,
    }
    return validate_intermediate(intermediate)
