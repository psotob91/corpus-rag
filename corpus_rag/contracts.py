"""Boundary contracts for corpus-rag.

Two layers:

1. `Intermediate` — the ingest product, RESCUED from ms-rie
   (`orchestrator/contracts.py`). Both ingest routes (docling / OCR-VLM / JATS)
   converge on this, and it is what `write_product` serializes into the
   `OUTPUT_LAYOUT` (`outputs/<slug>/...`).
2. `Chunk` / `Citation` / `SearchHit` — what the index consumes and what the
   retriever / MCP server return.

Stdlib only (typing) — imports without the heavy ingest/index deps, so the
contract and its tests run in the scaffold phase.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# ─────────────────────────── ingest product (ms-rie) ───────────────────────────


class Intermediate(TypedDict, total=False):
    """The single structure every ingest route produces (rescued from ms-rie)."""

    document: str               # slug / stable id of the document
    markdown: str               # hierarchical, RAG-ready markdown
    tables: list[dict]          # validated .table.json twins
    figures: list[dict]         # {figure_id, page, has_caption, caption}
    formulas: list[dict]        # {formula_id, latex, source, coherence?}
    n_tables: int
    n_figures: int
    n_formulas: int
    engine: str                 # 'docling' | 'pymupdf' | ...
    device: str                 # 'cpu' | 'cuda'
    ocr_status: str             # 'degraded' | 'vlm' (scanned route only)
    pmcid: str                  # JATS route only
    sections: list[dict]        # heading-sections for chunking (ingest)
    route: str                  # extraction path
    source_class: str           # 'pdf-native' | 'pdf-scan-ocr' | 'pdf-scan-image'
    source_confidence: float    # classifier confidence [0, 1]
    source_signals: dict        # per-document classification evidence


INTERMEDIATE_REQUIRED = ("document", "markdown", "tables", "figures", "formulas")


# ─────────────────────────── index / retrieval types ───────────────────────────

ChunkKind = Literal["text", "table", "figure", "formula"]


class Chunk(TypedDict, total=False):
    """One line of `chunks.jsonl` — the unit the index embeds and retrieves."""

    chunk_id: str               # stable: <doc_slug>-c<N>
    doc_id: str                 # document slug
    text: str                   # chunk text (for text/caption chunks)
    kind: ChunkKind             # text | table | figure | formula
    section_path: list[str]     # hierarchical heading path
    parents: list[str]          # parent chunk_ids (hierarchy)
    page: int
    artifact_id: str            # <doc>-t/-f/-e<N> when kind != text
    group_id: str               # main<->annex group, or absent
    role: str                   # 'main' | 'annex'


class Citation(TypedDict, total=False):
    doc_id: str
    chunk_id: str
    section_path: list[str]
    page: int
    artifact_id: str            # present for table/figure/formula hits


class SearchHit(TypedDict):
    chunk: Chunk
    score: float
    citation: Citation


class ContractError(ValueError):
    """An artifact did not satisfy its contract."""


def validate_intermediate(d: dict) -> Intermediate:
    """Runtime guard: every ingest route must pass here before write_product."""
    missing = [k for k in INTERMEDIATE_REQUIRED if k not in d]
    if missing:
        raise ContractError(f"intermediate incomplete: missing {missing} (have: {sorted(d)})")
    return d  # type: ignore[return-value]


def validate_chunk(d: dict) -> Chunk:
    """Runtime guard for a `chunks.jsonl` line before indexing."""
    for k in ("chunk_id", "doc_id", "kind"):
        if k not in d:
            raise ContractError(f"chunk missing required key '{k}' (have: {sorted(d)})")
    if d["kind"] not in ("text", "table", "figure", "formula"):
        raise ContractError(f"invalid chunk kind: {d['kind']!r}")
    if d["kind"] == "text" and not d.get("text"):
        raise ContractError(f"text chunk {d['chunk_id']!r} has no text")
    if d["kind"] != "text" and not d.get("artifact_id"):
        raise ContractError(f"{d['kind']} chunk {d['chunk_id']!r} missing artifact_id")
    return d  # type: ignore[return-value]
