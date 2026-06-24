# OUTPUT_LAYOUT — the ingest→index boundary contract

How every ingested document is stored. Project-agnostic. WRITTEN by
`corpus_rag.ingest.write_product`, CONSUMED by `corpus_rag.index.build`.
(Rescued from ms-rie; this is the seam that lets ms-rie — or a future
MinerU-based ingester — feed the index.)

## One self-contained product per document

```
outputs/<doc_slug>/
  document.md              # hierarchical, RAG-ready markdown
  document.meta.json       # provenance + status + counts (fixed schema, below)
  chunks.jsonl             # ONE hierarchical chunk per line (section_path + parents)
  tables/<id>.table.json   # validated table twin;   <id> = <doc_slug>-t<N>
  figures/<id>.json        # figure metadata + backref; <id> = <doc_slug>-f<N>
  formulas/<id>.json       # {latex, valid, coherence, source} + backref; <id> = <doc_slug>-e<N>
outputs/INDEX.md           # human README of processed documents
outputs/llms.txt           # LLM-navigable index (<=10KB)
```

## `chunks.jsonl` — the unit the index embeds (see `contracts.Chunk`)

Each line: `chunk_id`, `doc_id`, `kind` (`text`|`table`|`figure`|`formula`),
`text` (for text/caption), `section_path`, `parents`, `page`, and `artifact_id`
(for non-text chunks linking to the `tables/figures/formulas` artifact).

## `document.meta.json` — fixed schema

`id`, `source` (`pdf-native`|`pdf-scan`|`jats`|`html`), `route`, `doi`,
`status` (`ok`|`needs_human_review`), `sha256_16`, `n_chunks`/`n_tables`/
`n_figures`/`n_formulas`, `group_id`/`role` (main↔annex), `errors`.

## Rules (what a validator checks)

1. Every `outputs/<slug>/` has `document.md` + `document.meta.json` (no half products).
2. Stable `id`; tables/figures/formulas carry slug-derived ids (`-t/-f/-e<N>`).
3. No `.table.json` left unvalidated against the table schema.
4. `llms.txt` ≤ 10KB; `INDEX.md` lists each document with status and errors.
5. Annexes link to their main via `group_id`.
