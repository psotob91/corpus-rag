"""Local ONNX embedder (fastembed) per rag.config.yaml.

Thin wrapper over fastembed's TextEmbedding (BAAI/bge-small-en-v1.5, dim 384,
no torch / no GPU). The model instance is cached per-model so repeated
get_embedder() calls (build + every search) reuse one loaded ONNX session.
"""
from __future__ import annotations

from functools import lru_cache

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Adapter exposing .encode(texts) -> list[list[float]] over fastembed."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # lazy: keep module import-safe

        self.model = model
        self._emb = TextEmbedding(model)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings into a list of float vectors (dim 384)."""
        if not texts:
            return []
        return [list(v) for v in self._emb.embed(list(texts))]


@lru_cache(maxsize=4)
def _cached_embedder(model: str) -> Embedder:
    return Embedder(model)


def get_embedder(model: str = DEFAULT_MODEL) -> Embedder:
    """Return a cached Embedder for `model` (loads the ONNX model once)."""
    return _cached_embedder(model)


# Back-compat alias for the original scaffold name.
def embedder(model: str = DEFAULT_MODEL) -> Embedder:
    return get_embedder(model)
