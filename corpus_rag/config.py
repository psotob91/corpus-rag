"""Load the corpus-rag recipe (`rag.config.yaml`).

The recipe is the committed source of truth for how the corpus is ingested,
indexed, and retrieved. The built index is derived from it and gitignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG = "rag.config.yaml"


def load(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read and parse the recipe. PyYAML is a core dependency (installed by `make setup`)."""
    import yaml  # imported lazily so the contract/tests import without deps

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"recipe not found: {p} (expected {DEFAULT_CONFIG} at the repo root)")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
