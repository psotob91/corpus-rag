# corpus-rag

A domain module that turns a corpus of **scientific articles** (with equations,
figures, and tables) into a queryable knowledge source, exposed to Claude Code /
Cowork as an **MCP server**. It is the retrieval companion to the
[`datavidence-template-project`](https://github.com/psotob91/datavidence-template-project)
template and the `psotobverse-utils` plugin — a **separate, per-project module**,
not part of either.

> **Status: P1 + P1.5 — working and verified.** fetch + ingest + hybrid index +
> retrieval + MCP server, all tested **end-to-end on a real open-access paper**.
> P1.5 adds a **docling** structural route (real heading nesting, TableFormer,
> formula→LaTeX; default when installed) over the lean PyMuPDF fallback, and an
> **eval gate** (`corpus_rag/eval`) that decides hybrid-vs-routing-vs-graph from
> data. Query routing (P2) and a concept graph (P3) are built **only if the eval
> says so**. See `ARCHITECTURE.md` and `docs/adr/0001-…`, `0002-…`.

## What it does (all-in-one)

```
fetch (DOI→JATS→PDF)  →  ingest (multimodal)  →  index (hybrid)  →  retrieve  →  MCP server
                          chunks + tables/             LanceDB +        BM25+dense   search_corpus(...)
                          figures/formulas             fastembed        (+rerank?)   → chunks + citations
```

1. **fetch** — acquire sources ethically (DOI → JATS → open-access PDF cascade).
2. **ingest** — extract text, **tables, figures, equations** into the
   `OUTPUT_LAYOUT` contract (`chunks.jsonl` + `tables/figures/formulas` + `meta`).
   Two engines, same output: **docling** (`--extra docling`, default when
   installed — real layout, TableFormer, formula→LaTeX) or a **lean PyMuPDF**
   no-GPU fallback. Pick with `CORPUS_RAG_INGEST_ENGINE=docling|pymupdf`.
   *CMCM classification + the OUTPUT_LAYOUT contract are rescued from `ms-rie-orchestrator`.*
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
- **P1.5 — DONE:** docling structural extraction (real headings, TableFormer,
  formula→LaTeX) + **eval gate** (`make eval`) to decide P2/P3 from data.
- **P2 (gated):** query routing (lexical / dense / table-QA) — only if `make eval` flags a gap.
- **P3 (gated):** concept graph (LazyGraphRAG) — only if `make eval` shows hybrid failing on multi-source queries.

## Usage

```bash
make setup                                   # uv sync (install core deps)

# (optional) fetch an open-access PDF by DOI, else just drop PDFs into corpus/
uv run python -c "from corpus_rag.fetch.sources import fetch; print(fetch('10.1371/journal.pone.0173955', email='you@example.org'))"

make ingest PDF=corpus/_fetched/<file>.pdf   # -> outputs/<slug>/ (docling if installed, else PyMuPDF)
make index                                   # build the hybrid index from outputs/
make eval                                    # decide hybrid vs routing vs graph from YOUR corpus
make serve                                   # run the MCP server (stdio)
```

Attach the server to any project by copying `.mcp.json.example` into that
project's `.mcp.json` (set the absolute path to this repo). The Claude agent can
then call `search_corpus`, `get_document`, and `get_table/figure/formula`.

### Verification (evidence)

Tested end-to-end on a real open-access paper (PLOS ONE `10.1371/journal.pone.0173955`):
- **fetch → ingest → index → search → MCP**: docling ingest yields 23 sections
  with real heading titles, 5 tables, 16 figures; `search_corpus` returns the
  paper with citations; the MCP `corpus` server lists 5 tools.
- **eval gate**: local recall@10 = 0.97, global recall@10 = 0.88 →
  `HYBRID_SUFFICIENT` (no graph justified on this corpus; re-run on yours).

Hermetic tests (no network/models): `tests/test_e2e.py` (build+search),
`tests/test_eval.py` (the gate), `tests/test_docling_mapping.py` (docling mapping).

**Limitations:** the lean PyMuPDF fallback uses font-size heading heuristics and
empty formula LaTeX; docling fixes headings/tables and fills formula LaTeX **when
the PDF has display equations** (the clinical test paper had none). Deep heading
nesting needs docling to emit distinct header levels.
