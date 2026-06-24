# corpus-rag

A domain module that turns a corpus of **scientific articles** (with equations,
figures, and tables) into a queryable knowledge source, exposed to Claude Code /
Cowork as an **MCP server**. It is the retrieval companion to the
[`datavidence-template-project`](https://github.com/psotob91/datavidence-template-project)
template and the `psotobverse-utils` plugin — a **separate, per-project module**,
not part of either.

> **Status: SCAFFOLD.** This repo currently contains the architecture, contracts,
> recipe, and stubs. The engine is not implemented yet and dependencies are not
> installed. See `ARCHITECTURE.md` and `docs/adr/0001-architecture.md`, then the
> phased plan below.

## What it does (all-in-one)

```
fetch (DOI→JATS→PDF)  →  ingest (multimodal)  →  index (hybrid)  →  retrieve  →  MCP server
                          chunks + tables/             LanceDB +        BM25+dense   search_corpus(...)
                          figures/formulas             fastembed        (+rerank?)   → chunks + citations
```

1. **fetch** — acquire sources ethically (DOI → JATS → open-access PDF cascade).
2. **ingest** — extract text, **tables, figures, equations** into the
   `OUTPUT_LAYOUT` contract (`chunks.jsonl` + `tables/figures/formulas` + `meta`).
   *Rescued from `ms-rie-orchestrator` (docling pipeline, formula/figure extraction, CMCM classification).*
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

- **P1 (MVP):** fetch + ingest + hybrid index + retrieve + MCP server. *(this scaffold targets P1)*
- **P2:** query routing (lexical / dense / table-QA) when there is more than one query type.
- **P3:** concept graph (LazyGraphRAG) — gated by `eval/` metrics, not by default.

## Usage (once implemented)

```bash
make setup        # uv sync (installs deps — NOT done yet)
make ingest PDF=path/to/article.pdf     # → outputs/<slug>/...
make index        # build the hybrid index from outputs/ (the recipe)
make serve        # run the MCP server
```
Then register the MCP server in a project's `.mcp.json` so the Claude agent can
call `search_corpus`.
