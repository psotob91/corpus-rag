"""Acquire a source: DOI -> JATS -> open-access PDF cascade (robots/ToS-respecting).

SCAFFOLD — not implemented. RESCUE: ms-rie orchestrator/fetch_source.py (NCBI efetch > Europe PMC > Unpaywall > Crossref)
"""
from __future__ import annotations


def fetch(doi_or_path: str, cache_dir: str = 'corpus/_fetched') -> str:
    raise NotImplementedError("scaffold — implement per ARCHITECTURE.md")
