"""Embedding models — dense (local Ollama nomic) + sparse (FastEmbed BM25) + rerank.

Three singletons, all in-process or local:

* :func:`get_dense_embed_model` — ``nomic-embed-text`` served by **local** Ollama, wrapped as a
  LlamaIndex :class:`BaseEmbedding` so it drops into ``IngestionPipeline`` / response synthesis.
  Prepends nomic **task prefixes** (``search_document:`` on chunks, ``search_query:`` on queries)
  — the documented way to use nomic-embed-text for retrieval.
* :func:`get_sparse_embed_model` — FastEmbed **BM25** sparse encoder (``Qdrant/bm25``); produces
  Qdrant ``SparseVector``s for the named sparse field.
* :func:`get_reranker` — FastEmbed **cross-encoder** reranker (``ms-marco-MiniLM-L-6-v2``).

The dense model talks to *local* Ollama (embeddings stay in-container). The cloud models
(deepseek / glm) are never used for embeddings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

import httpx
from fastembed import SparseTextEmbedding
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
    """LlamaIndex ``BaseEmbedding`` backed by local Ollama ``nomic-embed-text``.

    Prepends nomic task prefixes so the same model works for both documents and queries
    (the asymmetric retrieval setup nomic-embed-text is trained for).
    """

    base_url: str = "http://127.0.0.1:11434"
    model: str = "nomic-embed-text"
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    # Default mirrors ``Settings.embed_request_timeout``; see the note there on why a cold
    # model load needs far more headroom than a steady-state embed.
    timeout: float = 300.0

    _client: PrivateAttr = PrivateAttr(default=None)

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

    # -- raw Ollama call -------------------------------------------------
    def _embed_one(self, text: str) -> List[float]:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        # Legacy /api/embeddings endpoint (still supported) returns {"embedding": [...]}.
        resp = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        return list(resp.json()["embedding"])

    def _embed_many(self, texts: List[str]) -> List[List[float]]:
        # Ollama's newer /api/embed accepts a batch under "input".
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=max(self.timeout, 10.0 * len(texts)),
        )
        if resp.status_code == 404:  # very old Ollama — fall back to per-text
            return [self._embed_one(t) for t in texts]
        resp.raise_for_status()
        return [list(v) for v in resp.json()["embeddings"]]

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
        return [self._get_query_embedding(q) for q in queries]

    def _get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        return [self._get_query_embedding(q) for q in queries]


@lru_cache(maxsize=1)
def get_dense_embed_model() -> NomicEmbedding:
    s = get_settings()
    # NOTE: no ``embed_dim=`` here — ``NomicEmbedding`` declares no such field, so pydantic's
    # default ``extra="ignore"`` silently dropped it. The collection's vector width comes from
    # ``settings.embed_dim`` at ``ensure_collection`` time, which is the only place it matters.
    return NomicEmbedding(
        base_url=s.ollama_local_base_url,
        model=s.embed_model,
        doc_prefix=s.embed_doc_prefix,
        query_prefix=s.embed_query_prefix,
        timeout=s.embed_request_timeout,
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