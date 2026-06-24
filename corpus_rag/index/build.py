"""THE RECIPE: build the hybrid LanceDB index from outputs/ per rag.config.yaml.

SCAFFOLD — not implemented. RESCUE: new (LanceDB hybrid: vector + full-text/BM25); reads chunks.jsonl
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("scaffold — implement per ARCHITECTURE.md")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
