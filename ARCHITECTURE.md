# Architecture

`corpus-rag` is an **all-in-one** scientific-corpus retrieval module: it fetches,
ingests, indexes, retrieves, and serves. It is a **separate per-project module**
attached to a project via MCP — it is NOT in the template repo and NOT in the
`psotobverse-utils` plugin (those stay domain-free).

## The pipeline and its boundary contract

```
 corpus/ (PDFs, or DOIs)
        │
   [fetch]  sources.py — DOI → JATS → open-access PDF cascade (NCBI efetch >
        │               Europe PMC > Unpaywall > Crossref). Robots/ToS-respecting.
        ▼
   [ingest] pipeline.py — docling (native) | OCR/VLM (scanned); classify.py (CMCM,
        │               deterministic-first traits before any VLM); figures.py /
        │               formulas.py (→ LaTeX) / tables.py (→ structured) ; write_product.py
        ▼
   ===== OUTPUT_LAYOUT contract (the boundary) =====
   outputs/<slug>/ document.md · chunks.jsonl · meta.json · tables/ figures/ formulas/
        │           (one hierarchical chunk per line, with section_path + parents)
        ▼
   [index]  build.py — read chunks.jsonl + artifact metadata → embed (fastembed) →
        │             LanceDB table (vector + full-text/BM25). THE RECIPE is committed;
        │             the .lance index is gitignored and regenerated.
        ▼
   [retrieve] search.py — hybrid (dense + BM25, RRF fusion) → optional rerank (off
        │                by default) → chunks + citations + linked tables/figures/formulas
        ▼
   [server] mcp_server.py — MCP tools: search_corpus, get_document, get_table/figure/formula
        ▼
   Claude agent (Code or Cowork) in ANY project queries the corpus via MCP.
```

The **contract** (`corpus_rag/contracts.py`, rescued from `ms-rie`) is what
decouples ingestion from retrieval: anything that emits a valid `OUTPUT_LAYOUT`
product can be indexed. (Even though this module is all-in-one, keeping the
contract explicit lets ms-rie — or a future MinerU-based ingester — feed it.)

## Rescued from `ms-rie-orchestrator`

| Piece | From ms-rie | Role here |
|---|---|---|
| `OUTPUT_LAYOUT` + `contracts.Intermediate` | `docs/OUTPUT_LAYOUT.md`, `orchestrator/contracts.py` | the ingest→index boundary schema |
| docling pipeline + OCR/VLM fallback | `orchestrator/step2_pipeline_a.py`, `step4_vlm.py` | `ingest/pipeline.py` |
| CMCM deterministic-first classifier | `scripts/classify_corpus.py`, `orchestrator/pdf_utils.py` | `ingest/classify.py` |
| figure / formula extraction | `scripts/extract_figures.py`, `orchestrator/formula_extractor.py` | `ingest/figures.py`, `ingest/formulas.py` |
| DOI→JATS→PDF fetch cascade | `orchestrator/fetch_source.py` | `fetch/sources.py` |
| main↔annex relationships | `scripts/build_relationships.py` | ingest metadata (`group_id`/`role`) |

## Retrieval stack (local, reproducible, no GPU)

- **LanceDB** — embedded, file-based; native vector + full-text (BM25) + hybrid
  fusion + rerankers. One dependency covers the "hybrid is the baseline" finding.
- **fastembed** — ONNX embeddings (no torch / no GPU); default `bge-small-en-v1.5`.
- **MCP SDK** — serve over stdio; registered in a project's `.mcp.json`.
- **rerank** — optional cross-encoder, **off by default** (evidence: it can
  *degrade* retrieval on text+table corpora; A/B-test before enabling).

## Reproducibility

Commit the **recipe** (`rag.config.yaml` + `corpus_rag/index/build.py`), never the
index. The `.rag/` index and any model cache are gitignored. Re-running `make
index` rebuilds deterministically from `outputs/`.

## When to add a graph (P3)

Do NOT start with GraphRAG. `corpus_rag/eval/benchmark.py` generates a
local-vs-global query split (BenchmarkQED-style) and measures whether the hybrid
baseline fails on global/multi-hop questions. Only then add **LazyGraphRAG**
(cheap indexing, comparable global-query quality per Microsoft) as an escalation.
