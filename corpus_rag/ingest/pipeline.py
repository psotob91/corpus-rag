"""Ingest one PDF -> OUTPUT_LAYOUT product. CLI: python -m corpus_rag.ingest.pipeline <pdf>.

SCAFFOLD — not implemented. RESCUE: ms-rie orchestrator/step2_pipeline_a.py (docling) + step4_vlm.py (OCR/VLM)
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("scaffold — implement per ARCHITECTURE.md")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
