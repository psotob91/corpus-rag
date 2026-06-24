"""CMCM deterministic-first classifier: cheap PyMuPDF traits before any VLM; pick route.

RESCUE: ms-rie scripts/classify_corpus.py + orchestrator/pdf_utils.py

Samples a few pages with PyMuPDF (no torch / no VLM) to derive cheap traits and
route the document to a 'pdf-native' or 'pdf-scan' extraction path.
"""
from __future__ import annotations

from pathlib import Path


# Pages we sample for the cheap traits. Sampling (rather than scanning every
# page) keeps classify() fast on large PDFs; the traits are coarse by design.
_MAX_SAMPLE_PAGES = 5

# A page with very little extractable text but high image coverage is the
# fingerprint of a scanned page.
_TEXT_LAYER_MIN_CHARS = 40        # per sampled page, to count as "has text"
_SCAN_IMAGE_COVERAGE = 0.50       # image area / page area above this => image-heavy


def _sample_indices(n_pages: int) -> list[int]:
    """Evenly spaced page indices to sample (always includes first page)."""
    if n_pages <= _MAX_SAMPLE_PAGES:
        return list(range(n_pages))
    step = n_pages / _MAX_SAMPLE_PAGES
    return sorted({int(i * step) for i in range(_MAX_SAMPLE_PAGES)})


def _page_image_coverage(page) -> float:
    """Fraction of the page area covered by image blocks (clamped to [0, 1])."""
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    img_area = 0.0
    info = page.get_text("dict")
    for block in info.get("blocks", []):
        # block type 1 == image block in PyMuPDF's dict output.
        if block.get("type") == 1:
            x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
            img_area += abs((x1 - x0) * (y1 - y0))
    return max(0.0, min(1.0, img_area / page_area))


def _page_col_count(page) -> int:
    """Cheap 1-vs-2 column guess from the x-spread of text block left edges."""
    info = page.get_text("dict")
    width = abs(page.rect.width) or 1.0
    lefts = []
    for block in info.get("blocks", []):
        if block.get("type") == 0:  # text block
            x0 = block.get("bbox", (0,))[0]
            lefts.append(x0 / width)
    if len(lefts) < 4:
        return 1
    # If a meaningful share of blocks start past the horizontal midpoint, the
    # page is probably two columns.
    right_half = sum(1 for fx in lefts if fx > 0.5)
    return 2 if right_half >= max(2, len(lefts) // 4) else 1


def classify(pdf_path: str) -> dict:
    """Derive cheap traits from a few sampled pages and route the document.

    Returns a dict with: n_pages, image_coverage, has_text_layer,
    col_count_guess, looks_scanned, source_class ('pdf-native'|'pdf-scan'),
    and route (same value as source_class — the extraction path to take).
    """
    import fitz  # lazy: keep import cheap so the module imports without the dep

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    try:
        n_pages = doc.page_count
        if n_pages == 0:
            return {
                "n_pages": 0,
                "image_coverage": 0.0,
                "has_text_layer": False,
                "col_count_guess": 1,
                "looks_scanned": False,
                "source_class": "pdf-native",
                "route": "pdf-native",
            }

        indices = _sample_indices(n_pages)
        coverages: list[float] = []
        text_pages = 0
        col_guesses: list[int] = []
        for i in indices:
            page = doc.load_page(i)
            coverages.append(_page_image_coverage(page))
            if len(page.get_text("text").strip()) >= _TEXT_LAYER_MIN_CHARS:
                text_pages += 1
            col_guesses.append(_page_col_count(page))

        image_coverage = sum(coverages) / len(coverages)
        has_text_layer = text_pages > 0
        # Mode of the per-page column guesses.
        col_count_guess = max(set(col_guesses), key=col_guesses.count)
        # Scanned == almost no extractable text AND image-heavy pages.
        looks_scanned = (not has_text_layer) and image_coverage >= _SCAN_IMAGE_COVERAGE
    finally:
        doc.close()

    source_class = "pdf-scan" if looks_scanned else "pdf-native"
    return {
        "n_pages": n_pages,
        "image_coverage": round(image_coverage, 4),
        "has_text_layer": has_text_layer,
        "col_count_guess": col_count_guess,
        "looks_scanned": looks_scanned,
        "source_class": source_class,
        "route": source_class,
    }
