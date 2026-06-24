"""THE RECIPE: build the hybrid LanceDB index from outputs/ per rag.config.yaml.

Reads every outputs/<slug>/chunks.jsonl (validating each line against the Chunk
contract), embeds the chunk text (or the caption for non-text chunks) with
fastembed, and writes a LanceDB table named 'chunks' with a vector column plus
the scalar columns the retriever/citation layer needs. A full-text (BM25) index
is created on 'text' so the retriever can do hybrid (dense + BM25) search.

The index is a DERIVED artifact: the table is recreated (mode='overwrite') on
every run, so `make index` rebuilds deterministically from outputs/.

ASCII-only output (Windows cp1252 console).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from corpus_rag import config as _config
from corpus_rag.contracts import validate_chunk
from corpus_rag.index.embed import get_embedder

TABLE_NAME = "chunks"


def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields build needs out of the recipe, with safe defaults."""
    index = cfg.get("index", {}) or {}
    embedder_cfg = index.get("embedder", {}) or {}
    corpus = cfg.get("corpus", {}) or {}
    return {
        "index_path": index.get("path", ".rag/corpus.lance"),
        "model": embedder_cfg.get("model", "BAAI/bge-small-en-v1.5"),
        "outputs_dir": corpus.get("outputs_dir", "outputs"),
    }


def _chunk_text(chunk: dict[str, Any]) -> str:
    """Indexable text for a chunk: its text, falling back to a caption/label."""
    text = chunk.get("text")
    if text:
        return str(text)
    # Non-text chunks may carry the searchable string under other keys.
    for key in ("caption", "label", "title"):
        val = chunk.get(key)
        if val:
            return str(val)
    # Last resort: a stable label so the row is never empty (BM25 needs text).
    return f"{chunk.get('kind', 'artifact')} {chunk.get('artifact_id') or chunk.get('chunk_id', '')}".strip()


def _iter_chunk_files(outputs_dir: Path):
    """Yield every outputs/<slug>/chunks.jsonl, sorted for deterministic builds."""
    if not outputs_dir.is_dir():
        return
    for slug_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        cj = slug_dir / "chunks.jsonl"
        if cj.is_file():
            yield cj


def _load_rows(outputs_dir: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Read + validate all chunks. Returns (rows-without-vectors, doc_ids)."""
    rows: list[dict[str, Any]] = []
    doc_ids: set[str] = set()
    for cj in _iter_chunk_files(outputs_dir):
        for lineno, line in enumerate(cj.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{cj}:{lineno}: invalid JSON ({e})") from e
            chunk = validate_chunk(raw)
            section_path = chunk.get("section_path") or []
            row = {
                "chunk_id": str(chunk["chunk_id"]),
                "doc_id": str(chunk["doc_id"]),
                "text": _chunk_text(chunk),
                "kind": str(chunk["kind"]),
                "section_path": json.dumps(list(section_path)),
                "page": int(chunk["page"]) if chunk.get("page") is not None else -1,
                "artifact_id": str(chunk.get("artifact_id") or ""),
            }
            rows.append(row)
            doc_ids.add(row["doc_id"])
    return rows, doc_ids


def build_index(config_path: str = "rag.config.yaml") -> dict[str, Any]:
    """Build the LanceDB 'chunks' table from outputs/. Returns a summary dict."""
    import lancedb

    cfg = _config.load(config_path)
    resolved = _resolve(cfg)

    config_dir = Path(config_path).resolve().parent
    outputs_dir = Path(resolved["outputs_dir"])
    if not outputs_dir.is_absolute():
        outputs_dir = config_dir / outputs_dir
    index_path = Path(resolved["index_path"])
    if not index_path.is_absolute():
        index_path = config_dir / index_path

    rows, doc_ids = _load_rows(outputs_dir)
    if not rows:
        raise ValueError(
            f"no chunks found under {outputs_dir} "
            f"(expected outputs/<slug>/chunks.jsonl); nothing to index"
        )

    emb = get_embedder(resolved["model"])
    vectors = emb.encode([r["text"] for r in rows])
    for row, vec in zip(rows, vectors):
        row["vector"] = vec

    index_path.parent.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(index_path))
    table = db.create_table(TABLE_NAME, data=rows, mode="overwrite")
    table.create_fts_index("text", replace=True)

    return {
        "n_chunks": len(rows),
        "n_docs": len(doc_ids),
        "index_path": str(index_path),
        "table": TABLE_NAME,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: build the index and print a one-line ASCII summary."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else "rag.config.yaml"
    summary = build_index(config_path)
    print(f"indexed {summary['n_chunks']} chunks from {summary['n_docs']} docs")
    print(f"  table '{summary['table']}' at {summary['index_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
