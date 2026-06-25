# Example corpus — Clinical Prediction Models (CPM)

A real ~20-document corpus on clinical prediction models (Cox, Efron, Harrell,
Riley, Snell/TRIPOD, Altman, ...) used to exercise corpus-rag end to end and to
run the **eval gate** on a realistic corpus instead of a toy.

## We ship the DOIs, not the PDFs (and why)

`prediction-models.dois.txt` lists the articles by DOI. **The PDFs are NOT in
this repo** (`corpus/` and `outputs/` are gitignored). Rationale — this is the
best-practice answer to "should the corpus folder live in the repo?":

- **Copyright.** Most of these are publisher-copyrighted (JRSS-B, JASA, Stat Med,
  JNCI, JAMIA); redistributing the PDFs from a public repo would infringe. A few
  are open access (BMJ/BMC are CC-BY; preprints vary) but mixing licenses is unsafe.
- **The project's own principle:** *commit the recipe, not the data.* The PDFs are
  data; the reproducible recipe is the DOI list + the `fetch` module.
- **Repo hygiene.** ~30 MB of binaries would bloat git history permanently.

## Reproduce the corpus + eval

```bash
# 1. Get the PDFs into corpus/_raw. Open-access ones via the fetch module, e.g.:
uv run python -c "from corpus_rag.fetch.sources import fetch; print(fetch('10.1136/bmj.m441', email='you@example.org'))"
#    Paywalled ones: supply your own copy (e.g. exported from Zotero / institutional access).
# 2. Ingest, index, evaluate:
make ingest PDF=corpus/_raw/<file>.pdf   # repeat per PDF (docling if installed, else PyMuPDF)
make index
make eval
```

## What the eval found (2026-06)

Run on this 20-doc corpus (1098 chunks, lean PyMuPDF ingest):

| scope | recall@10 | recall@20 | recall@30 | recall@50 |
|---|---|---|---|---|
| local (single-source) | 0.97 | 1.00 | 1.00 | 1.00 |
| global (multi-source) | 0.44 | 0.50 | 0.55 | 0.69 |

**Decision: `CONSIDER_GRAPH_OR_ROUTING`** (gap 0.52 at k=10). Hybrid retrieval
finds single chunks excellently but misses *distributed* evidence for
cross-cutting concepts. A 2-document toy corpus gave the opposite
(`HYBRID_SUFFICIENT`) — the gap only appears at realistic scale.

**Read before building anything:**

- Global recall climbs with depth (0.44 → 0.69 from k=10 → 50) but never matches
  local. Much of the easy win is simply *retrieve deeper for global queries* →
  **query routing (P2) is the cheaper first move**; a concept graph (P3) targets
  the residual and should be **gated on a re-measure after routing**.
- **Confounder:** lean PyMuPDF over-fragments older PDFs (e.g. 325 "sections"),
  spreading a concept across more chunks and depressing global recall. Re-run with
  `--extra docling` (fewer, denser chunks) before treating the gap as structural.
- Probes are templated single terms, not real questions. Validate with real user
  queries before committing budget to P3.
