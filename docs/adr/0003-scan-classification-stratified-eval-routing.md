# 3. Scan classification, stratified eval, P2 routing — and why P3 (graph) stays gated off

Date: 2026-06. Supplements ADR-0001/0002.

## Status

Accepted. P3 (concept graph) deliberately NOT built.

## Context

On the real 19-doc clinical-prediction-models corpus the eval flagged a
local-vs-global gap (local ~0.90, global ~0.43 recall@10). Two hypotheses had to
be tested before paying for a graph: (a) old scanned papers (Efron, Harrell, …)
drag retrieval down; (b) the gap is a genuine distributed-evidence problem.

## Decisions & evidence

1. **Robust trichotomy classifier** (ADR detail in code): real full-page raster
   coverage (`get_images`+`get_image_bbox`) + OCR producer metadata + text-noise.
   Correctly flags the 5 pre-2000 papers as `pdf-scan-ocr`, 14 modern as
   `pdf-native`. (The old classifier flagged none — it required *no* text layer,
   but these scans carry an OCR text layer.)

2. **Stratified eval** (native vs scanned). Result on the real corpus:
   local 0.90 native / 0.89 scan; global 0.42 native / 0.43 scan. **The global gap
   is NOT a scanned-doc artifact** — it is equally present in clean text-native
   docs. Hypothesis (a) is REJECTED; OCR quality is not the bottleneck. (Local
   recall on scanned docs ≈ native, so the OCR text layer is good enough to embed.)

3. **P2 query routing** (`retrieve/router.py`): embedding-exemplar local-vs-global
   classifier (fastembed, no new deps) + lexical fallback; global queries retrieve
   deeper (`global_top_k=50`, per the k-sweep where evidence recovers ~k=50).
   Enabled by default. **A/B (real corpus): global recall 0.12 → 0.38 (native
   0.21 → 0.57, scan 0.10 → 0.34), local 0.90 → 0.93.** Routing ~triples global
   recall with no cost to local.

4. **P3 (LazyGraphRAG-lite) — NOT built (gate stays closed).** Rationale:
   - Routing (cheap, done) recovered most of the global gap; deeper *hybrid*
     retrieval gathers the distributed evidence (k-sweep: global 0.43@10 →
     0.69@50). A graph's marginal value over "route + deeper-k" is unproven.
   - Absolute global-recall numbers are **probe-artifact-sensitive** (single-term
     probes → 0.69@k50; synthesis-phrased → 0.57 native) — not reliable enough to
     justify a graph's build/maintenance cost.
   - The templated, LLM-free probes cannot represent real synthesis queries; the
     honest gate (and the 2024–26 literature) is "build a graph only when metrics
     on REAL queries prove hybrid+routing insufficient." We are not there.

## Consequences

- Default ingestion records a per-doc source class + confidence; `make eval`
  reports stratified (native vs scan) metrics + a stratum note.
- Routing is on by default; the MCP `search_corpus` tool uses adaptive width.
- **Re-open the P3 gate only with real/LLM-authored global queries** showing
  text-native global recall still below target after routing. Until then, no graph.
- docling OCR re-ingest was de-prioritized: the stratified result shows scanned
  docs are not the bottleneck, so it would not change this verdict (it remains a
  chunk-quality nicety, runnable in batches if desired).
