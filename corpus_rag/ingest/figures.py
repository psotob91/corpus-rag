"""Extract figures + VLM captions; emit figures/<id>.json with backref.

SCAFFOLD — not implemented. RESCUE: ms-rie scripts/extract_figures.py + orchestrator/vlm_gemini.py
"""
from __future__ import annotations


def extract_figures(pdf_path: str, slug: str) -> list[dict]:
    raise NotImplementedError("scaffold — implement per ARCHITECTURE.md")
