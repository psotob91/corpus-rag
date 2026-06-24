# corpus-rag

A domain module that turns a corpus of **scientific articles** (with equations,
figures, and tables) into a queryable knowledge source, exposed to Claude Code /
Cowork as an **MCP server**. It is the retrieval companion to the
[`datavidence-template-project`](https://github.com/psotob91/datavidence-template-project)
template and the `psotobverse-utils` plugin — a **separate, per-project module**,
not part of either.

> **Status: P1 (MVP) — working and verified.** Lean ingest (PyMuPDF) + hybrid
> index (LanceDB + fastembed) + retrieval + MCP server are implemented and tested
> **end-to-end on a real open-access paper** (fetch → ingest → index → search →
> MCP). Heavy structural extraction (docling/VLM), query routing, and a concept
> graph are later phases (P1.5 / P2 / P3). See `ARCHITECTURE.md` and
> `docs/adr/0001-architecture.md`.

## What it does (all-in-one)

```
fetch (DOI→JATS→PDF)  →  ingest (multimodal)  →  index (hybrid)  →  retrieve  →  MCP server
                          chunks + tables/             LanceDB +        BM25+dense   search_corpus(...)
                          figures/formulas             fastembed        (+rerank?)   → chunks + citations
```

1. **fetch** — acquire sources ethically (DOI → JATS → open-access PDF cascade).
2. **ingest** — extract text, **tables, figures, equations** into the
   `OUTPUT_LAYOUT` contract (`chunks.jsonl` + `tables/figures/formulas` + `meta`).
   P1 uses a **lean PyMuPDF** extractor (no GPU); docling/VLM is an optional
   upgrade (`--extra docling`). *CMCM classification + the OUTPUT_LAYOUT contract are rescued from `ms-rie-orchestrator`.*
3. **index** — build a **hybrid** index (dense embeddings + BM25 full-text) with
   **LanceDB + fastembed** (local, no GPU, reproducible). The **recipe** is
   committed; the built index is gitignored.
4. **retrieve** — hybrid search (RRF fusion), optional cross-encoder rerank
   (**off by default** — it can hurt on table-heavy content).
5. **server** — an **MCP server** (`corpus`) exposing `search_corpus`,
   `get_document`, `get_table/figure/formula` to the agent in any project.

## Why these choices (evidence)

See `docs/adr/0001-architecture.md`. Short version, from a 2024–2026 review:
hybrid (BM25+dense) is the minimum-viable baseline; a reranker is **not** a
guaranteed win on tables; VLM/MinerU lead on scientific-PDF extraction; a
concept **graph (LazyGraphRAG)** is added **only if** a BenchmarkQED-style
local-vs-global evaluation shows the hybrid baseline failing on multi-hop
queries. **Commit the recipe, not the index.**

## Phased plan

- **P1 (MVP) — DONE:** fetch + lean ingest + hybrid index + retrieve + MCP server.
- **P1.5:** docling/VLM structural extraction (better headings, tables, formula→LaTeX).
- **P2:** query routing (lexical / dense / table-QA) when there is more than one query type.
- **P3:** concept graph (LazyGraphRAG) — gated by `eval/` metrics, not by default.

## Usage

```bash
make setup                                   # uv sync (install core deps)

# (optional) fetch an open-access PDF by DOI, else just drop PDFs into corpus/
uv run python -c "from corpus_rag.fetch.sources import fetch; print(fetch('10.1371/journal.pone.0173955', email='you@example.org'))"

make ingest PDF=corpus/_fetched/<file>.pdf   # -> outputs/<slug>/ (OUTPUT_LAYOUT)
make index                                   # build the hybrid index from outputs/
make serve                                   # run the MCP server (stdio)
```

Attach the server to any project by copying `.mcp.json.example` into that
project's `.mcp.json` (set the absolute path to this repo). The Claude agent can
then call `search_corpus`, `get_document`, and `get_table/figure/formula`.

### P1 verification (evidence)

Tested end-to-end on a real open-access paper (PLOS ONE `10.1371/journal.pone.0173955`):
fetch → ingest (49 sections, 5 tables, 14 figures) → index (54 chunks / 2 docs) →
`search_corpus` returns the paper with citations → MCP server lists 5 tools and
`get_document`/`get_table` resolve. Hermetic regression test: `tests/test_e2e.py`.

**P1 lean-ingest limitations** (improved by `--extra docling` in P1.5): heading
nesting is font-size heuristic (can mis-nest); inline formulas are detected
heuristically with empty LaTeX; figures need a raster image block (vector-drawn
figures are missed).
