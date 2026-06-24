"""Extract tables -> structured dicts using PyMuPDF page.find_tables().

RESCUE: ms-rie orchestrator/step2_pipeline_a (find_tables) + assets/table.schema.json

Lean (no docling / no torch). Each table becomes a dict:
  {"id": "<slug>-t<N>", "page": p, "n_rows", "n_cols", "cells": rows, "caption"}
where ``cells`` is a list of rows (each a list of cell strings) and ``caption``
is nearby text above the table that looks like a table caption.
"""
from __future__ import annotations

import re

# A caption line typically starts with "Table 1", "TABLE 2.", "Tab. 3" etc.
_CAPTION_RE = re.compile(r"^\s*(table|tab\.?)\s*\d+", re.IGNORECASE)

# How far above the table top edge to look for a caption (PDF points).
_CAPTION_LOOKUP_PTS = 60.0


def _clean(value) -> str:
    """Normalize a cell value to a single-line string."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _find_caption(page, table_bbox) -> str:
    """Find a 'Table N ...' line just above (or below) the table, if any."""
    tx0, ty0, tx1, ty1 = table_bbox
    candidates: list[tuple[float, str]] = []
    info = page.get_text("dict")
    for block in info.get("blocks", []):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
        # Horizontal overlap with the table and vertically near it.
        horiz_overlap = min(bx1, tx1) - max(bx0, tx0)
        near_above = 0 <= (ty0 - by1) <= _CAPTION_LOOKUP_PTS
        near_below = 0 <= (by0 - ty1) <= _CAPTION_LOOKUP_PTS
        if horiz_overlap <= 0 or not (near_above or near_below):
            continue
        text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        text = _clean(text)
        if _CAPTION_RE.match(text):
            # Distance from the table edge: prefer the closest caption line.
            dist = (ty0 - by1) if near_above else (by0 - ty1)
            candidates.append((dist, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def extract_tables(doc, slug: str) -> list[dict]:
    """Extract all tables from an open PyMuPDF document.

    ``doc`` is an open ``fitz.Document``. Returns a list of table dicts in
    reading order, numbered ``<slug>-t1``, ``<slug>-t2``, ...
    """
    out: list[dict] = []
    n = 0
    for page in doc:
        try:
            finder = page.find_tables()
        except Exception:
            # find_tables can fail on degenerate pages; skip rather than crash.
            continue
        for table in finder.tables:
            try:
                raw = table.extract()
            except Exception:
                continue
            rows = [[_clean(c) for c in row] for row in raw]
            if not rows or all(not any(r) for r in rows):
                continue  # empty / blank table
            n += 1
            n_rows = table.row_count if getattr(table, "row_count", None) else len(rows)
            n_cols = table.col_count if getattr(table, "col_count", None) else (
                max((len(r) for r in rows), default=0)
            )
            out.append(
                {
                    "id": f"{slug}-t{n}",
                    "page": page.number + 1,
                    "n_rows": int(n_rows),
                    "n_cols": int(n_cols),
                    "cells": rows,
                    "caption": _find_caption(page, table.bbox),
                }
            )
    return out
