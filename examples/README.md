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

Run on this 19-doc corpus (14 native, 5 scanned-OCR; ~1057 chunks, lean PyMuPDF).
Full reasoning in `docs/adr/0003-...md`.

**1. The gap is real at scale.** A 2-doc toy gave `HYBRID_SUFFICIENT`; the real
corpus gives `CONSIDER_GRAPH_OR_ROUTING` (local ~0.90 vs global ~0.43 recall@10).

**2. It is NOT a scanned-doc problem (stratified eval):**

| scope | native | scanned |
|---|---|---|
| local recall@10  | 0.90 | 0.89 |
| global recall@10 | 0.42 | 0.43 |

Native and scanned perform identically — the OCR text layer is good enough to
embed, and the global gap is equally present in clean modern PDFs. So better OCR
is *not* the fix.

**3. Query routing (P2) is the fix — it ~triples global recall.** Routing global/
synthesis queries to deeper retrieval (`global_top_k=50`):

| scope | routing OFF | routing ON | Δ |
|---|---|---|---|
| global (overall) | 0.12 | 0.38 | +0.27 |
| global (native)  | 0.21 | 0.57 | +0.36 |
| local            | 0.90 | 0.93 | +0.03 |

Deeper *hybrid* retrieval recovers the distributed evidence (k-sweep: global
0.43@10 → 0.69@50). Routing is **enabled by default**.

**4. A concept graph (P3) is NOT built.** Routing recovered most of the gap with a
cheap, existing mechanism; the residual is within probe-artifact noise (templated
probes, not real questions). Per best practice, build a graph only when metrics on
**real/LLM-authored** global queries prove hybrid+routing insufficient. The gate
stays closed until then.
