# 1. corpus-rag architecture: all-in-one, hybrid-first, MCP-served

Date: 2026-06

## Status

Accepted (scaffold). Implementation pending (P1).

## Context

Some projects need retrieval over scientific articles containing equations,
figures, and tables. A 2024–2026 review (and a partial adversarial deep-research
pass) produced these load-bearing findings:

- VERIFIED: for text-and-table documents, **hybrid** retrieval (BM25 + dense) is
  the strongest base; sparse BM25 alone often beats dense (T2-RAGBench, EACL 2026).
- VERIFIED (correction): a cross-encoder **reranker can HURT** on table-heavy
  content — do not assume it helps; A/B-test it.
- PRIMARY-SOURCE: VLM/MinerU lead on scientific-PDF parsing (OmniDocBench, CVPR
  2025); evaluate formula extraction with LLM-as-judge semantic equivalence, not
  character matching. LazyGraphRAG (Microsoft) claims ~0.1% of full-GraphRAG
  indexing cost — treat as a vendor claim. BenchmarkQED measures local-vs-global
  query mix to decide if a graph is needed.

## Decision

1. **All-in-one module, separate repo, MCP-served.** `corpus-rag` fetches +
   ingests + indexes + retrieves + serves, in its own repository, attached to a
   project via an MCP server. NOT in the template, NOT in `psotobverse-utils`.
2. **Reuse `ms-rie` for ingestion** (docling pipeline, OCR/VLM fallback, CMCM
   classification, figure/formula extraction, DOI→JATS→PDF fetch) via the
   `OUTPUT_LAYOUT` contract, which stays the explicit ingest→index boundary.
3. **Hybrid-first retrieval** with **LanceDB + fastembed** (local, no GPU,
   reproducible). Rerank optional and **off by default**.
4. **Commit the recipe, not the index.** `rag.config.yaml` + `build.py` are
   versioned; the `.rag/` index is gitignored and regenerated.
5. **Escalation ladder, metrics-gated:** P1 hybrid → P2 query routing → P3
   graph (LazyGraphRAG) only if `eval/` shows the hybrid baseline failing on
   global/multi-hop queries. MinerU is a future extraction upgrade over docling.

## Consequences

- The module is autonomous (can ingest PDFs itself) but stays decoupled at the
  contract, so ms-rie or a MinerU ingester can also feed it.
- Heavier dependency surface than a retrieve-only module (docling/VLM may need a
  GPU for some PDFs — kept in an optional extra, isolated like ms-rie's MinerU env).
- A new ADR supersedes this once the engine is built and measured.
