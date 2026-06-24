"""MCP server 'corpus': tools search_corpus, get_document, get_table/figure/formula.

A FastMCP (stdio) server exposing the corpus over MCP:

  - search_corpus(query, top_k) -> hybrid retrieval hits (chunk + citation)
  - get_document(doc_id)        -> outputs/<doc_id>/document.md
  - get_table/figure/formula(artifact_id) -> the artifact JSON twin

Artifact ids are slug-derived: <doc>-t<N> / <doc>-f<N> / <doc>-e<N>. The owning
document slug is the id with its trailing -t/-f/-e<N> stripped, so a table id
`my-doc-t3` resolves to outputs/my-doc/tables/my-doc-t3.table.json.

Keep this import-safe: build the FastMCP app and register tools at import time
(no blocking), but only call app.run() from main().

ASCII-only output.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from corpus_rag import config as _config
from corpus_rag.retrieve import search as _search

# id suffix: ...-t<N> (table), -f<N> (figure), -e<N> (formula/equation)
_ARTIFACT_RE = re.compile(r"^(?P<doc>.+)-(?P<kind>[tfe])(?P<num>\d+)$")

_KIND_DIR = {"t": "tables", "f": "figures", "e": "formulas"}
_KIND_SUFFIX = {"t": ".table.json", "f": ".json", "e": ".json"}


def _outputs_dir(config_path: str = "rag.config.yaml") -> Path:
    cfg = _config.load(config_path)
    corpus = cfg.get("corpus", {}) or {}
    out = Path(corpus.get("outputs_dir", "outputs"))
    if not out.is_absolute():
        out = Path(config_path).resolve().parent / out
    return out


def _read_artifact(artifact_id: str, expected_kind: str, config_path: str) -> dict[str, Any]:
    """Resolve an artifact id to its JSON twin and return the parsed dict."""
    m = _ARTIFACT_RE.match(artifact_id)
    if not m:
        return {"error": f"unrecognized artifact id: {artifact_id!r}"}
    kind = m.group("kind")
    if kind != expected_kind:
        return {
            "error": (
                f"artifact id {artifact_id!r} is a "
                f"{_KIND_DIR.get(kind, '?')[:-1]} id, not a "
                f"{_KIND_DIR[expected_kind][:-1]} id"
            )
        }
    doc = m.group("doc")
    path = (
        _outputs_dir(config_path)
        / doc
        / _KIND_DIR[kind]
        / f"{artifact_id}{_KIND_SUFFIX[kind]}"
    )
    if not path.is_file():
        return {"error": f"artifact not found: {path}"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON in {path}: {e}"}


def build_app(config_path: str = "rag.config.yaml"):
    """Construct and return the FastMCP app with all corpus tools registered."""
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("corpus")

    @app.tool()
    def search_corpus(query: str, top_k: int = 8) -> list[dict]:
        """Hybrid (dense + BM25) search over the corpus. Returns ranked hits."""
        return _search.search_corpus(query, top_k=top_k, config_path=config_path)

    @app.tool()
    def get_document(doc_id: str) -> str:
        """Return the full RAG-ready markdown for a document slug."""
        path = _outputs_dir(config_path) / doc_id / "document.md"
        if not path.is_file():
            return f"error: document not found: {path}"
        return path.read_text(encoding="utf-8")

    @app.tool()
    def get_table(artifact_id: str) -> dict:
        """Return the structured table twin for a table artifact id (<doc>-t<N>)."""
        return _read_artifact(artifact_id, "t", config_path)

    @app.tool()
    def get_figure(artifact_id: str) -> dict:
        """Return the figure metadata for a figure artifact id (<doc>-f<N>)."""
        return _read_artifact(artifact_id, "f", config_path)

    @app.tool()
    def get_formula(artifact_id: str) -> dict:
        """Return the formula record for a formula artifact id (<doc>-e<N>)."""
        return _read_artifact(artifact_id, "e", config_path)

    return app


# Import-safe module-level app (registers tools; does NOT run the server).
app = build_app()


def main(argv: list[str] | None = None) -> int:
    """Run the MCP server over stdio (blocking)."""
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
