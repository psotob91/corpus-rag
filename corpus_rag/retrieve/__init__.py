"""corpus_rag.retrieve — hybrid search over the built index. See ARCHITECTURE.md.

Re-exports `search_corpus` so callers can use `corpus_rag.retrieve.search_corpus`.
Import-safe: the search module imports only stdlib + config at module load;
lancedb / fastembed are imported lazily inside the functions.
"""
from corpus_rag.retrieve.search import search_corpus

__all__ = ["search_corpus"]
