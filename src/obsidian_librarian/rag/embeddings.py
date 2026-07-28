"""Embedding models — dense (FastEmbed nomic) + sparse (FastEmbed BM25) + rerank.

Three singletons, all in-process — no model server, no network:

* :func:`get_dense_embed_model` — ``nomic-embed-text-v1.5`` run in-process by **FastEmbed**
  (ONNX), wrapped as a LlamaIndex :class:`BaseEmbedding` so it drops into ``IngestionPipeline`` /
  response synthesis. Prepends nomic **task prefixes** (``search_document:`` on chunks,
  ``search_query:`` on queries) — the documented way to use nomic-embed-text for retrieval.
* :func:`get_sparse_embed_model` — FastEmbed **BM25** sparse encoder (``Qdrant/bm25``); produces
  Qdrant ``SparseVector``s for the named sparse field.
* :func:`get_reranker` — FastEmbed **cross-encoder** reranker (``ms-marco-MiniLM-L-6-v2``).

FastEmbed already ships nomic-embed-text-v1.5, so the dense model no longer needs a local Ollama
server: we dropped Ollama from the container entirely (it served embeddings and nothing else —
generation and the judge both go to Ollama *Cloud*). That removes the boot-blocking ``ollama pull``
and the server process. All three models now share one FastEmbed cache dir on the persistent
volume. The cloud models (deepseek / glm) are never used for embeddings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from ..config import get_settings

__all__ = [
    "NomicEmbedding",
    "get_dense_embed_model",
    "get_sparse_embed_model",
    "get_reranker",
    "sparse_to_qdrant",
]


class NomicEmbedding(BaseEmbedding):
    """LlamaIndex ``BaseEmbedding`` backed by in-process FastEmbed ``nomic-embed-text-v1.5``.

    Prepends nomic task prefixes so the same model works for both documents and queries
    (the asymmetric retrieval setup nomic-embed-text is trained for).

    **The prefixes are applied here, by us.** FastEmbed does *not* add them: its ``embed()`` and
    ``query_embed()`` produce byte-identical vectors for the same text (verified — cosine 1.0),
    unlike some other FastEmbed models where ``query_embed`` applies a model-specific query
    template. So we keep prefixing explicitly rather than delegating to ``query_embed``; dropping
    the prefixes would silently degrade retrieval quality rather than fail loudly.
    """

    model: str = "nomic-ai/nomic-embed-text-v1.5-Q"
    cache_dir: str = ""
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "

    _model: PrivateAttr = PrivateAttr(default=None)

    def _backend(self) -> TextEmbedding:
        """Lazily construct the ONNX model (first use pays the download/load)."""
        if self._model is None:
            kwargs: dict = {"model_name": self.model}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            self._model = TextEmbedding(**kwargs)
        return self._model

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._backend().embed(texts)]

    def _embed_one(self, text: str) -> List[float]:
        return self._embed_many([text])[0]

    # -- BaseEmbedding interface ----------------------------------------
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_one(f"{self.query_prefix}{query}")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_one(f"{self.doc_prefix}{text}")

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed_many([f"{self.doc_prefix}{t}" for t in texts])

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

    async def _aget_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        # Queries use the query prefix (not the doc prefix); mirror the sync path.
        return self._embed_many([f"{self.query_prefix}{q}" for q in queries])

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        return self._embed_many([f"{self.query_prefix}{q}" for q in queries])


@lru_cache(maxsize=1)
def get_dense_embed_model() -> NomicEmbedding:
    s = get_settings()
    # NOTE: no ``embed_dim=`` here — ``NomicEmbedding`` declares no such field, so pydantic's
    # default ``extra="ignore"`` silently dropped it. The collection's vector width comes from
    # ``settings.embed_dim`` at ``ensure_collection`` time, which is the only place it matters.
    # (v1.5-Q is 768-dim, same as the Ollama model it replaced, so existing collections stay
    # dimensionally valid — but vectors from the two models are NOT interchangeable; see the
    # re-index note in ``Settings.embed_model``.)
    return NomicEmbedding(
        model=s.embed_model,
        cache_dir=str(s.fastembed_cache_path),
        doc_prefix=s.embed_doc_prefix,
        query_prefix=s.embed_query_prefix,
    )


@lru_cache(maxsize=1)
def get_sparse_embed_model() -> SparseTextEmbedding:
    """FastEmbed BM25 sparse encoder. ``model.embed(texts)`` yields ``SparseVector``s.

    ``cache_dir`` points at the persistent volume so the ONNX model downloads once, not on
    every container recreation.
    """
    s = get_settings()
    return SparseTextEmbedding(model_name=s.sparse_model, cache_dir=str(s.fastembed_cache_path))


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    """FastEmbed cross-encoder reranker.

    Returns a :class:`TextCrossEncoder`. Its ``.rerank(query, documents)`` yields plain
    float scores **in input order** (no ``.rank``/``top_k``/per-result ``.index``/``.score``
    objects) — callers must sort and take the top-k themselves.

    ``cache_dir`` points at the persistent volume (see :meth:`get_sparse_embed_model`).
    """
    s = get_settings()
    return TextCrossEncoder(model_name=s.rerank_model, cache_dir=str(s.fastembed_cache_path))


def sparse_to_qdrant(sparse_obj: Any) -> Any:
    """Convert a FastEmbed ``SparseVector`` to a qdrant-client ``SparseVector``."""
    from qdrant_client.http.models import SparseVector

    return SparseVector(
        indices=list(map(int, sparse_obj.indices)),
        values=list(map(float, sparse_obj.values)),
    )