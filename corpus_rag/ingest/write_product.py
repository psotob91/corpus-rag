"""Write the OUTPUT_LAYOUT product: document.md, chunks.jsonl, meta.json, artifacts.

RESCUE: ms-rie orchestrator/step5_formatter.write_product

Serializes a validated Intermediate into ``outputs/<slug>/`` per OUTPUT_LAYOUT:
  document.md, document.meta.json, chunks.jsonl,
  tables/<id>.table.json, figures/<id>.json, formulas/<id>.json.

Chunking: one text chunk per heading-section (from the Intermediate ``sections``)
plus one chunk per table/figure/formula (kind set, artifact_id back-reference,
text = caption). Every chunk passes validate_chunk before it is written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from corpus_rag.contracts import validate_chunk


def _sha256_16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _table_text(table: dict) -> str:
    """Searchable text for a table chunk: caption + flattened cells."""
    parts = []
    if table.get("caption"):
        parts.append(table["caption"])
    for row in table.get("cells", []):
        cells = [c for c in row if c]
        if cells:
            parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def write_product(intermediate: dict, outputs_dir: str = "outputs") -> str:
    """Write the OUTPUT_LAYOUT product for one Intermediate; return the dir path."""
    slug = intermediate["document"]
    out_dir = Path(outputs_dir) / slug
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "formulas").mkdir(parents=True, exist_ok=True)

    # ---- document.md -------------------------------------------------------
    markdown = intermediate.get("markdown", "") or ""
    doc_md_path = out_dir / "document.md"
    doc_md_path.write_text(markdown, encoding="utf-8")

    tables = intermediate.get("tables", [])
    figures = intermediate.get("figures", [])
    formulas = intermediate.get("formulas", [])
    sections = intermediate.get("sections", [])

    # ---- chunks.jsonl ------------------------------------------------------
    chunks: list[dict] = []
    c = 0

    # Text chunks: one per heading-section (with a non-empty body).
    for sec in sections:
        body = (sec.get("body") or "").strip()
        if not body:
            continue
        c += 1
        section_path = sec.get("section_path") or ([sec["title"]] if sec.get("title") else [])
        title = sec.get("title") or ""
        # Prefix the heading into the chunk text so retrieval sees the context.
        text = (f"{title}\n\n{body}" if title else body).strip()
        chunks.append(
            validate_chunk(
                {
                    "chunk_id": f"{slug}-c{c}",
                    "doc_id": slug,
                    "text": text,
                    "kind": "text",
                    "section_path": section_path,
                    "parents": [],
                    "page": int(sec.get("page", 1)),
                }
            )
        )

    # If no section produced a text chunk, fall back to one chunk of the markdown.
    if not any(ch["kind"] == "text" for ch in chunks) and markdown.strip():
        c += 1
        chunks.append(
            validate_chunk(
                {
                    "chunk_id": f"{slug}-c{c}",
                    "doc_id": slug,
                    "text": markdown.strip(),
                    "kind": "text",
                    "section_path": [],
                    "parents": [],
                    "page": 1,
                }
            )
        )

    # Artifact chunks: tables, figures, formulas.
    for table in tables:
        c += 1
        text = _table_text(table) or table.get("caption") or f"Table {table['id']}"
        chunks.append(
            validate_chunk(
                {
                    "chunk_id": f"{slug}-c{c}",
                    "doc_id": slug,
                    "text": text,
                    "kind": "table",
                    "section_path": [],
                    "parents": [],
                    "page": int(table.get("page", 1)),
                    "artifact_id": table["id"],
                }
            )
        )

    for fig in figures:
        c += 1
        text = fig.get("caption") or f"Figure {fig['id']}"
        chunks.append(
            validate_chunk(
                {
                    "chunk_id": f"{slug}-c{c}",
                    "doc_id": slug,
                    "text": text,
                    "kind": "figure",
                    "section_path": [],
                    "parents": [],
                    "page": int(fig.get("page", 1)),
                    "artifact_id": fig["id"],
                }
            )
        )

    for formula in formulas:
        c += 1
        text = formula.get("text") or formula.get("latex") or f"Formula {formula['id']}"
        chunks.append(
            validate_chunk(
                {
                    "chunk_id": f"{slug}-c{c}",
                    "doc_id": slug,
                    "text": text,
                    "kind": "formula",
                    "section_path": [],
                    "parents": [],
                    "page": int(formula.get("page", 1)),
                    "artifact_id": formula["id"],
                }
            )
        )

    chunks_path = out_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for ch in chunks:
            fh.write(json.dumps(ch, ensure_ascii=False) + "\n")

    # ---- artifact files ----------------------------------------------------
    for table in tables:
        _write_json(out_dir / "tables" / f"{table['id']}.table.json", {**table, "doc_id": slug})
    for fig in figures:
        _write_json(out_dir / "figures" / f"{fig['id']}.json", {**fig, "doc_id": slug})
    for formula in formulas:
        _write_json(out_dir / "formulas" / f"{formula['id']}.json", {**formula, "doc_id": slug})

    # ---- document.meta.json (fixed schema) ---------------------------------
    status = "ok"
    errors: list[str] = []
    if not chunks:
        status = "needs_human_review"
        errors.append("no chunks produced")

    meta = {
        "id": slug,
        "source": intermediate.get("source_class", "pdf-native"),
        "source_confidence": intermediate.get("source_confidence"),
        "route": intermediate.get("route", intermediate.get("source_class", "pdf-native")),
        "doi": None,
        "status": status,
        "sha256_16": _sha256_16(markdown),
        "n_chunks": len(chunks),
        "n_tables": len(tables),
        "n_figures": len(figures),
        "n_formulas": len(formulas),
        "group_id": None,
        "role": None,
        "errors": errors,
    }
    signals = intermediate.get("source_signals")
    if signals:
        meta["source_signals"] = signals
    _write_json(out_dir / "document.meta.json", meta)

    return str(out_dir)
