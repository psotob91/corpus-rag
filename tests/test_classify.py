"""Hermetic tests for the trichotomy classifier helpers + meta plumbing (no fitz, no PDF)."""

from __future__ import annotations

import json
from pathlib import Path

from corpus_rag.ingest.classify import (
    _aggregate,
    _classify_page,
    _has_scanner_signature,
)
from corpus_rag.ingest.write_product import write_product


# ── _classify_page truth table ───────────────────────────────────────────────

def test_page_native():
    assert _classify_page(cov=0.0, n_chars=2000, noise_ratio=0.0, n_fonts=6) == "native"


def test_page_scan_ocr():
    # full-page image + plenty of clean text => scanned WITH a usable OCR layer
    assert _classify_page(cov=0.98, n_chars=3000, noise_ratio=0.01, n_fonts=1) == "scan-ocr"


def test_page_scan_image_no_text():
    assert _classify_page(cov=0.98, n_chars=5, noise_ratio=0.0, n_fonts=0) == "scan-image"


def test_page_scan_image_noisy_text():
    # full-page image + lots of text but very noisy => not a usable layer
    assert _classify_page(cov=0.98, n_chars=3000, noise_ratio=0.30, n_fonts=1) == "scan-image"


def test_page_ambiguous_band_one_font_is_scan():
    assert _classify_page(cov=0.60, n_chars=3000, noise_ratio=0.0, n_fonts=1) == "scan-ocr"


def test_page_ambiguous_band_many_fonts_is_native():
    assert _classify_page(cov=0.60, n_chars=3000, noise_ratio=0.0, n_fonts=6) == "native"


# ── _aggregate (doc label + confidence) ──────────────────────────────────────

def test_aggregate_all_native():
    cls, conf, frac = _aggregate(["native", "native", "native"], meta_ocr_signature=False)
    assert cls == "pdf-native" and frac == 0.0 and conf >= 0.7


def test_aggregate_scan_ocr_majority():
    cls, conf, frac = _aggregate(["scan-ocr", "scan-ocr", "native"], meta_ocr_signature=False)
    assert cls == "pdf-scan-ocr" and frac > 0.5


def test_aggregate_scan_image_when_no_usable_text():
    cls, _conf, _frac = _aggregate(["scan-image", "scan-image", "scan-ocr"], meta_ocr_signature=False)
    assert cls == "pdf-scan-image"


def test_aggregate_metadata_boosts_confidence():
    _c1, conf_no_meta, _f1 = _aggregate(["scan-ocr", "scan-ocr"], meta_ocr_signature=False)
    _c2, conf_meta, _f2 = _aggregate(["scan-ocr", "scan-ocr"], meta_ocr_signature=True)
    assert conf_meta > conf_no_meta  # OCR metadata is a confirmer


def test_aggregate_confidence_monotonic_in_scan_fraction():
    _c1, c_half, _ = _aggregate(["scan-ocr", "native"], meta_ocr_signature=False)
    _c2, c_full, _ = _aggregate(["scan-ocr", "scan-ocr"], meta_ocr_signature=False)
    assert c_full >= c_half


def test_scanner_signature_match():
    assert _has_scanner_signature("Acrobat 5.0 Paper Capture Plug-in", "")
    assert _has_scanner_signature("", "ABBYY FineReader")
    assert not _has_scanner_signature("Adobe InDesign", "Adobe PDF Library")


# ── meta plumbing: source_class / confidence / signals reach document.meta.json ─

def test_write_product_records_source_fields(tmp_path: Path):
    inter = {
        "document": "scan_doc",
        "markdown": "# Methods\n\nWe fit a Cox model.\n",
        "sections": [{"level": 1, "title": "Methods", "page": 1,
                      "section_path": ["Methods"], "body": "We fit a Cox model."}],
        "tables": [], "figures": [], "formulas": [],
        "source_class": "pdf-scan-ocr",
        "source_confidence": 0.91,
        "source_signals": {"scan_fraction": 1.0, "meta_ocr_signature": True},
    }
    out_dir = write_product(inter, outputs_dir=str(tmp_path / "outputs"))
    meta = json.loads((Path(out_dir) / "document.meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "pdf-scan-ocr"
    assert meta["source_confidence"] == 0.91
    assert meta["source_signals"]["scan_fraction"] == 1.0
