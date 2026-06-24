"""Extract figures from image blocks; emit figure dicts with nearby captions.

RESCUE: ms-rie scripts/extract_figures.py + orchestrator/vlm_gemini.py

Lean (no VLM). Each figure becomes:
  {"id": "<slug>-f<N>", "page": p, "bbox": [...], "has_caption": bool,
   "caption": "Figure N ..."}
The caption is nearby text starting with "Fig"/"Figure" (VLM captioning is a
later upgrade; this is the deterministic-first baseline).
"""
from __future__ import annotations

import re

# A figure caption typically starts with "Figure 1", "Fig. 2", "FIG 3".
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)

# How far below/above the image bbox to look for a caption (PDF points).
_CAPTION_LOOKUP_PTS = 80.0

# Ignore tiny images (logos, rules, icons) that are not real figures.
_MIN_IMAGE_AREA = 2500.0  # ~50x50 pt


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _find_caption(page, bbox) -> str:
    """Find a 'Figure N ...' line just below (preferred) or above an image."""
    ix0, iy0, ix1, iy1 = bbox
    candidates: list[tuple[float, str]] = []
    info = page.get_text("dict")
    for block in info.get("blocks", []):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
        horiz_overlap = min(bx1, ix1) - max(bx0, ix0)
        near_below = 0 <= (by0 - iy1) <= _CAPTION_LOOKUP_PTS
        near_above = 0 <= (iy0 - by1) <= _CAPTION_LOOKUP_PTS
        if horiz_overlap <= 0 or not (near_below or near_above):
            continue
        text = _clean(
            " ".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
        )
        if _CAPTION_RE.match(text):
            # Captions sit below figures far more often than above.
            dist = (by0 - iy1) if near_below else (iy0 - by1) + 1000.0
            candidates.append((dist, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def extract_figures(doc, slug: str) -> list[dict]:
    """Extract figures (image blocks) from an open PyMuPDF document.

    ``doc`` is an open ``fitz.Document``. Returns a list of figure dicts in
    reading order, numbered ``<slug>-f1``, ``<slug>-f2``, ...
    """
    out: list[dict] = []
    n = 0
    for page in doc:
        info = page.get_text("dict")
        for block in info.get("blocks", []):
            if block.get("type") != 1:  # image block
                continue
            x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
            if abs((x1 - x0) * (y1 - y0)) < _MIN_IMAGE_AREA:
                continue
            bbox = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
            caption = _find_caption(page, (x0, y0, x1, y1))
            n += 1
            out.append(
                {
                    "id": f"{slug}-f{n}",
                    "page": page.number + 1,
                    "bbox": bbox,
                    "has_caption": bool(caption),
                    "caption": caption,
                }
            )
    return out
