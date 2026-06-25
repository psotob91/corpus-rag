"""Deterministic-first PDF source classifier (trichotomy): native / scan-ocr / scan-image.

RESCUE: ms-rie scripts/classify_corpus.py + orchestrator/pdf_utils.py

Cheap PyMuPDF-only traits (no torch / no VLM) decide whether a PDF is born-digital
text or a SCAN, and — for scans — whether it carries a usable OCR text layer
(`pdf-scan-ocr`, e.g. Efron 1983 / Harrell 1996) or none (`pdf-scan-image`). The
load-bearing signal is full-page raster coverage (`get_images` + `get_image_bbox`);
producer/creator OCR metadata is a confirmer; text-noise distinguishes scan-ocr
from scan-image. The decision/confidence logic is factored into pure helpers
(`_classify_page`, `_aggregate`) for hermetic truth-table tests.
"""
from __future__ import annotations

from pathlib import Path

# Sampling keeps classify() fast on large PDFs; traits are coarse by design.
_MAX_SAMPLE_PAGES = 8

# Per-page raster coverage thresholds (image area / page area).
_FULLPAGE_COV = 0.80          # >= this => the page is a scan
_AMBIG_LO = 0.55              # [_AMBIG_LO, _FULLPAGE_COV) + <=1 font => also a scan

# Text-layer thresholds.
_TEXT_PRESENT_CHARS = 40      # per page, to count toward has_text_layer (legacy)
_TEXT_USABLE_CHARS = 200      # a scanned page with >= this much clean text => scan-ocr
_OCR_NOISE_RATIO = 0.06       # fraction of garbage chars above which text is "noisy"

# Substrings in producer/creator that fingerprint scanner/OCR software.
_SCANNER_PRODUCERS = (
    "abbyy", "finereader", "paper capture", "acrobat capture", "scansnap",
    "tesseract", "omnipage", "nuance", "kofax", "readiris", "scanned by",
    "capture plug-in",
)


def _sample_indices(n_pages: int) -> list[int]:
    """Evenly spaced page indices to sample (always includes the first page)."""
    if n_pages <= _MAX_SAMPLE_PAGES:
        return list(range(n_pages))
    step = n_pages / _MAX_SAMPLE_PAGES
    return sorted({int(i * step) for i in range(_MAX_SAMPLE_PAGES)})


def _page_image_coverage(page) -> float:
    """Fraction of the page covered by raster images (clamped to [0, 1]).

    Uses the real image bboxes (get_images + get_image_bbox), not text-dict image
    blocks — full-page scans show up here near 1.0 while born-digital pages are 0.
    """
    page_area = abs(page.rect.width * page.rect.height)
    if page_area <= 0:
        return 0.0
    img_area = 0.0
    try:
        images = page.get_images(full=True)
    except Exception:
        return 0.0
    for img in images:
        try:
            bbox = page.get_image_bbox(img)
            img_area += abs(bbox.width * bbox.height)
        except Exception:
            continue
    return max(0.0, min(1.0, img_area / page_area))


def _text_noise_ratio(text: str) -> float:
    """Fraction of non-space chars that are control / replacement / OCR-junk."""
    nonspace = [c for c in text if not c.isspace()]
    if not nonspace:
        return 0.0
    bad = 0
    for c in nonspace:
        o = ord(c)
        if (
            o < 0x20                      # C0 control (whitespace already excluded)
            or 0x7F <= o <= 0x9F          # DEL + C1 control
            or c == "�"              # replacement char
            or 0xE000 <= o <= 0xF8FF      # private use area (bad font mappings)
            or c in "ﬀﬁﬂﬃﬄ"  # isolated ligatures ff fi fl ffi ffl
        ):
            bad += 1
    return bad / len(nonspace)


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
    right_half = sum(1 for fx in lefts if fx > 0.5)
    return 2 if right_half >= max(2, len(lefts) // 4) else 1


# ───────────────────────── pure decision helpers (testable) ─────────────────────────


def _classify_page(cov: float, n_chars: int, noise_ratio: float, n_fonts: int) -> str:
    """Per-page label: 'native' | 'scan-ocr' | 'scan-image' (pure, no fitz)."""
    scanned = cov >= _FULLPAGE_COV or (_AMBIG_LO <= cov < _FULLPAGE_COV and n_fonts <= 1)
    if not scanned:
        return "native"
    if n_chars >= _TEXT_USABLE_CHARS and noise_ratio < _OCR_NOISE_RATIO:
        return "scan-ocr"
    return "scan-image"


def _confidence(scanned: bool, scan_fraction: float, meta_ocr_signature: bool) -> float:
    """Deterministic, monotonic confidence in [0, 1]."""
    frac = scan_fraction if scanned else (1.0 - scan_fraction)
    excess = max(0.0, (frac - 0.5) / 0.5)          # 0 at 0.5, 1 at 1.0
    if scanned:
        meta_term = 0.20 if meta_ocr_signature else 0.0
    else:
        meta_term = 0.0 if meta_ocr_signature else 0.20  # metadata contradicting native -> lower
    return round(max(0.0, min(1.0, 0.55 + 0.25 * excess + meta_term)), 4)


def _aggregate(voting_labels: list[str], meta_ocr_signature: bool) -> tuple[str, float, float]:
    """Per-page labels -> (source_class, confidence, scan_fraction).

    Coverage-driven decision (scan_fraction >= 0.5); OCR metadata only boosts
    confidence (it is a confirmer, never required — avoids stale-metadata false
    positives). source_class in {pdf-native, pdf-scan-ocr, pdf-scan-image}.
    """
    n = len(voting_labels)
    if n == 0:
        return "pdf-native", 0.0, 0.0
    n_ocr = voting_labels.count("scan-ocr")
    n_img = voting_labels.count("scan-image")
    scan_fraction = (n_ocr + n_img) / n
    scanned = scan_fraction >= 0.5
    if scanned:
        source_class = "pdf-scan-ocr" if n_ocr >= n_img else "pdf-scan-image"
    else:
        source_class = "pdf-native"
    return source_class, _confidence(scanned, scan_fraction, meta_ocr_signature), round(scan_fraction, 4)


def _has_scanner_signature(producer: str, creator: str) -> bool:
    blob = f"{producer} {creator}".lower()
    return any(sig in blob for sig in _SCANNER_PRODUCERS)


# ───────────────────────────────── public API ─────────────────────────────────


def classify(pdf_path: str) -> dict:
    """Classify a PDF's source. Returns n_pages, image_coverage, has_text_layer,
    col_count_guess, looks_scanned, source_class (pdf-native|pdf-scan-ocr|
    pdf-scan-image), source_confidence, route (legacy pdf-native|pdf-scan), and a
    `signals` evidence dict.
    """
    import fitz  # lazy: keep the module importable without the dep

    path = Path(pdf_path)
    doc = fitz.open(str(path))
    try:
        n_pages = doc.page_count
        meta = doc.metadata or {}
        producer = meta.get("producer") or ""
        creator = meta.get("creator") or ""
        meta_ocr = _has_scanner_signature(producer, creator)

        if n_pages == 0:
            return _result(
                n_pages=0, image_coverage=0.0, has_text_layer=False, col_count_guess=1,
                source_class="pdf-native", source_confidence=0.0, scan_fraction=0.0,
                meta_ocr=meta_ocr, producer=producer, creator=creator,
                page_labels=[], text_chars=[], noise_ratios=[], font_counts=[],
            )

        indices = _sample_indices(n_pages)
        page_labels: list[str] = []
        coverages: list[float] = []
        text_chars: list[int] = []
        noise_ratios: list[float] = []
        font_counts: list[int] = []
        col_guesses: list[int] = []
        text_pages = 0
        for i in indices:
            page = doc.load_page(i)
            cov = _page_image_coverage(page)
            text = page.get_text("text")
            n_chars = len(text.strip())
            noise = _text_noise_ratio(text)
            try:
                n_fonts = len(page.get_fonts(full=True))
            except Exception:
                n_fonts = 0
            coverages.append(cov)
            text_chars.append(n_chars)
            noise_ratios.append(noise)
            font_counts.append(n_fonts)
            page_labels.append(_classify_page(cov, n_chars, noise, n_fonts))
            if n_chars >= _TEXT_PRESENT_CHARS:
                text_pages += 1
            col_guesses.append(_page_col_count(page))

        # Cover-sheet guard: the first page of a digitized paper is often a
        # re-typeset cover (JSTOR), so drop it from the scan vote when we have
        # enough pages — but keep its text for has_text_layer.
        voting = page_labels[1:] if (n_pages >= 3 and len(page_labels) > 1) else page_labels
        source_class, source_confidence, scan_fraction = _aggregate(voting, meta_ocr)

        return _result(
            n_pages=n_pages,
            image_coverage=sum(coverages) / len(coverages),
            has_text_layer=text_pages > 0,
            col_count_guess=max(set(col_guesses), key=col_guesses.count),
            source_class=source_class,
            source_confidence=source_confidence,
            scan_fraction=scan_fraction,
            meta_ocr=meta_ocr, producer=producer, creator=creator,
            page_labels=page_labels, text_chars=text_chars,
            noise_ratios=noise_ratios, font_counts=font_counts,
        )
    finally:
        doc.close()


def _result(*, n_pages, image_coverage, has_text_layer, col_count_guess, source_class,
            source_confidence, scan_fraction, meta_ocr, producer, creator,
            page_labels, text_chars, noise_ratios, font_counts) -> dict:
    def _mean(xs, default=0.0):
        return round(sum(xs) / len(xs), 4) if xs else default

    route = "pdf-native" if source_class == "pdf-native" else "pdf-scan"
    return {
        "n_pages": n_pages,
        "image_coverage": round(image_coverage, 4),
        "has_text_layer": has_text_layer,
        "col_count_guess": col_count_guess,
        "looks_scanned": source_class != "pdf-native",
        "source_class": source_class,
        "source_confidence": source_confidence,
        "route": route,
        "signals": {
            "scan_fraction": scan_fraction,
            "meta_ocr_signature": meta_ocr,
            "producer": producer,
            "creator": creator,
            "page_labels": page_labels,
            "mean_text_chars": _mean([float(x) for x in text_chars]),
            "mean_noise_ratio": _mean(noise_ratios),
            "mean_fonts": _mean([float(x) for x in font_counts]),
            "thresholds": {
                "fullpage_cov": _FULLPAGE_COV,
                "text_usable": _TEXT_USABLE_CHARS,
                "noise": _OCR_NOISE_RATIO,
            },
        },
    }
