"""Ingest one PDF -> OUTPUT_LAYOUT product. CLI: python -m corpus_rag.ingest.pipeline <pdf>.

LEAN route (PyMuPDF only, no torch / no docling). RESCUE: ms-rie
orchestrator/step2_pipeline_a.py (structure) + step4_vlm.py (route selection).

ingest(pdf_path) -> Intermediate:
  - open with fitz, infer a heading hierarchy from font sizes,
  - build RAG-ready hierarchical markdown + a structured ``sections`` list,
  - run classify / tables / figures / formulas,
  - assemble and validate the Intermediate.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

from corpus_rag.contracts import Intermediate, validate_intermediate
from corpus_rag.ingest.classify import classify
from corpus_rag.ingest.figures import extract_figures
from corpus_rag.ingest.formulas import extract_formulas
from corpus_rag.ingest.tables import extract_tables

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug_from_path(pdf_path: str) -> str:
    """Filesystem-safe slug derived from the PDF filename stem."""
    stem = Path(pdf_path).stem.lower()
    slug = _SLUG_RE.sub("-", stem).strip("-")
    return slug or "document"


def _block_text(block) -> str:
    return " ".join(
        span.get("text", "")
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ).strip()


def _block_size(block) -> float:
    """Representative (max) font size across the block's spans."""
    sizes = [
        span.get("size", 0.0)
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    return max(sizes) if sizes else 0.0


def _block_bold(block) -> bool:
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            flags = span.get("flags", 0)
            font = span.get("font", "")
            if flags & 16 or "bold" in font.lower() or "black" in font.lower():
                return True
    return False


def _collect_blocks(doc):
    """All text blocks across the document with (page, size, bold, text)."""
    blocks = []
    for page in doc:
        info = page.get_text("dict")
        for block in info.get("blocks", []):
            if block.get("type") != 0:
                continue
            text = _block_text(block)
            if not text:
                continue
            blocks.append(
                {
                    "page": page.number + 1,
                    "size": round(_block_size(block), 2),
                    "bold": _block_bold(block),
                    "text": text,
                    "y": block.get("bbox", (0, 0, 0, 0))[1],
                }
            )
    return blocks


def _body_size(blocks) -> float:
    """The most common font size == body text size."""
    counter = Counter(b["size"] for b in blocks)
    if not counter:
        return 0.0
    return counter.most_common(1)[0][0]


def _heading_levels(blocks, body_size):
    """Map the distinct larger-than-body sizes to heading levels 1..N."""
    bigger = sorted({b["size"] for b in blocks if b["size"] > body_size + 0.5}, reverse=True)
    # Cap at 3 heading levels; rarer/larger sizes share level 1.
    levels = {}
    for i, size in enumerate(bigger[:3]):
        levels[size] = i + 1
    for size in bigger[3:]:
        levels[size] = 3
    return levels


def _is_heading(block, body_size, levels) -> int:
    """Return heading level (1..3) for a block, or 0 if it's body text."""
    if block["size"] in levels:
        # Long blocks at heading size are usually emphasized paragraphs, not headings.
        if len(block["text"]) <= 120:
            return levels[block["size"]]
    # Bold, short, at body size and not ending in a sentence period -> minor heading.
    if (
        block["bold"]
        and len(block["text"]) <= 80
        and not block["text"].rstrip().endswith(".")
        and block["size"] >= body_size
    ):
        return 3
    return 0


def build_sections(doc) -> tuple[list[dict], str]:
    """Infer a heading hierarchy and return (sections, markdown).

    Each section: {"level": int, "title": str, "page": int, "section_path":
    list[str], "body": str}. The first section may be a synthetic preamble with
    level 0 and an empty title if the document opens with body text.
    """
    blocks = _collect_blocks(doc)
    body_size = _body_size(blocks)
    levels = _heading_levels(blocks, body_size)

    sections: list[dict] = []
    path_stack: list[tuple[int, str]] = []  # (level, title)
    current = {
        "level": 0,
        "title": "",
        "page": blocks[0]["page"] if blocks else 1,
        "section_path": [],
        "_body": [],
    }

    def _finalize(sec):
        sec["body"] = "\n\n".join(sec.pop("_body")).strip()
        return sec

    for block in blocks:
        lvl = _is_heading(block, body_size, levels)
        if lvl:
            # Close the running section.
            sections.append(_finalize(current))
            # Pop stack entries at or below this level.
            while path_stack and path_stack[-1][0] >= lvl:
                path_stack.pop()
            path_stack.append((lvl, block["text"]))
            current = {
                "level": lvl,
                "title": block["text"],
                "page": block["page"],
                "section_path": [t for _, t in path_stack],
                "_body": [],
            }
        else:
            current["_body"].append(block["text"])
    sections.append(_finalize(current))

    # Drop the synthetic empty preamble if it has no body.
    sections = [s for s in sections if s["title"] or s["body"]]

    # Render hierarchical markdown.
    md_parts: list[str] = []
    for sec in sections:
        if sec["title"]:
            hashes = "#" * max(1, min(6, sec["level"]))
            md_parts.append(f"{hashes} {sec['title']}")
        if sec["body"]:
            md_parts.append(sec["body"])
    markdown = "\n\n".join(md_parts).strip() + "\n"
    return sections, markdown


def ingest_pymupdf(pdf_path: str) -> Intermediate:
    """Ingest one PDF into a validated Intermediate (lean PyMuPDF route)."""
    import fitz  # lazy import

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"pdf not found: {path}")

    slug = slug_from_path(pdf_path)
    traits = classify(str(path))

    doc = fitz.open(str(path))
    try:
        sections, markdown = build_sections(doc)
        tables = extract_tables(doc, slug)
        figures = extract_figures(doc, slug)
        formulas = extract_formulas(doc, slug)
    finally:
        doc.close()

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
        "engine": "pymupdf",
        "device": "cpu",
        "source_class": traits["source_class"],
        "route": traits["route"],
        "traits": traits,
    }
    return validate_intermediate(intermediate)


def _docling_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("docling") is not None


def ingest(pdf_path: str, engine: str = "auto") -> Intermediate:
    """Ingest one PDF, selecting the extraction engine.

    engine='auto' (default) uses docling if installed (better structure +
    formula->LaTeX), else the lean PyMuPDF route. Force with 'docling' /
    'pymupdf' or the CORPUS_RAG_INGEST_ENGINE env var.
    """
    eng = (engine or "auto").lower()
    if eng == "auto":
        eng = (
            os.environ.get("CORPUS_RAG_INGEST_ENGINE", "").strip().lower()
            or ("docling" if _docling_available() else "pymupdf")
        )
    if eng == "docling":
        from corpus_rag.ingest.docling_pipeline import ingest_docling

        return ingest_docling(pdf_path)
    return ingest_pymupdf(pdf_path)


def main(argv: list[str] | None = None) -> int:
    """CLI: ingest the given PDF, write the OUTPUT_LAYOUT product, print the dir."""
    from corpus_rag.ingest.write_product import write_product

    args = list(sys.argv[1:] if argv is None else argv[1:])
    if not args:
        print("usage: python -m corpus_rag.ingest.pipeline <pdf> [outputs_dir]")
        return 2

    pdf_path = args[0]
    outputs_dir = args[1] if len(args) > 1 else "outputs"

    inter = ingest(pdf_path)
    out_dir = write_product(inter, outputs_dir=outputs_dir)
    print(
        "ingested %s [%s] -> %s (sections=%d tables=%d figures=%d formulas=%d)"
        % (
            inter["document"],
            inter.get("engine", "?"),
            out_dir,
            len(inter.get("sections", [])),
            inter["n_tables"],
            inter["n_figures"],
            inter["n_formulas"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
