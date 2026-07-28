"""Retrieval — Qdrant hybrid (dense + sparse, server-side RRF) + cross-encoder rerank.

The baseline retrieval path (see SPEC): every query runs **dense + sparse fusion → rerank**.

* Dense (nomic, ``search_query:`` prefix) and sparse (BM25) are both sent to Qdrant as
  ``prefetch``s on the same collection; Qdrant fuses them **server-side with RRF** — one
  query, no client-side merge.
* The top-N fused candidates are then re-scored by a FastEmbed **cross-encoder** reranker,
  which fixes ordering, and the top-k are returned as LlamaIndex :class:`NodeWithScore`
  objects carrying all metadata (path, heading_path, tags, wikilinks, backlinks,
  frontmatter) for the synthesizer.

Filters (optional): a path prefix and/or a tag list narrow the candidate set via Qdrant
payload filters before fusion.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.schema import NodeWithScore, TextNode
from qdrant_client.http import models as qm

from ..config import get_settings
from .embeddings import get_dense_embed_model, get_reranker, get_sparse_embed_model, sparse_to_qdrant
from .ingest import DENSE_NAME, SPARSE_NAME, QdrantStore

__all__ = ["retrieve", "hybrid_search"]


def _build_filter(path_prefix: Optional[str], tags: Optional[List[str]]) -> Optional[qm.Filter]:
    must: list[qm.FieldCondition] = []
    if path_prefix:
        # Qdrant has no native string-prefix match. ``MatchText`` does tokenized
        # full-text matching, which for slash-delimited paths approximates "the path
        # is under this folder" well enough for a pre-fusion candidate narrow. We
        # intentionally avoid ``MatchValue`` here — that is exact equality and would
        # only match a single note whose path equals ``path_prefix`` verbatim.
        must.append(qm.FieldCondition(key="path", match=qm.MatchText(text=path_prefix)))
    if tags:
        must.append(qm.FieldCondition(key="tags", match=qm.MatchAny(any=tags)))
    return qm.Filter(must=must) if must else None


def hybrid_search(
    query: str,
    top_n: Optional[int] = None,
    path_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[qm.ScoredPoint]:
    """Run the dense+sparse RRF fusion in Qdrant. Returns fused ScoredPoints (pre-rerank)."""
    s = get_settings()
    n = top_n or s.retrieval_top_n
    dense = get_dense_embed_model()._get_query_embedding(query)
    sparse = next(get_sparse_embed_model().embed([query]))
    qs = QdrantStore.from_settings()
    flt = _build_filter(path_prefix, tags)

    res = qs.client.query_points(
        collection_name=qs.collection,
        prefetch=[
            qm.Prefetch(query=dense, using=DENSE_NAME, limit=n, filter=flt),
            qm.Prefetch(query=sparse_to_qdrant(sparse), using=SPARSE_NAME, limit=n, filter=flt),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=n,
        with_payload=True,
    )
    return list(res.points)


def _to_node(point: qm.ScoredPoint) -> TextNode:
    p = point.payload or {}
    return TextNode(
        text=p.get("text", ""),
        metadata={
            "path": p.get("path", ""),
            "title": p.get("title", ""),
            "heading_path": p.get("heading_path", ""),
            "tags": p.get("tags", []),
            "wikilinks": p.get("wikilinks", []),
            "backlinks": p.get("backlinks", []),
            "frontmatter": p.get("frontmatter", {}),
            "chunk_index": p.get("chunk_index", 0),
            "fused_score": float(point.score or 0.0),
        },
    )


def retrieve(
    query: str,
    top_n: Optional[int] = None,
    top_k: Optional[int] = None,
    path_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[NodeWithScore]:
    """Hybrid retrieve → rerank. Returns the top-k reranked nodes with cross-encoder scores."""
    s = get_settings()
    n = top_n or s.retrieval_top_n
    k = top_k or s.rerank_top_k

    fused = hybrid_search(query, top_n=n, path_prefix=path_prefix, tags=tags)
    if not fused:
        return []
    nodes = [_to_node(p) for p in fused]

    # Cross-encoder rerank over the fused candidates. FastEmbed's TextCrossEncoder
    # returns plain float scores in *input order* (no .rank/.top_k/per-result objects),
    # so we sort by score descending and take the top-k ourselves.
    reranker = get_reranker()
    texts = [nd.text for nd in nodes]
    scores = list(reranker.rerank(query, texts))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [NodeWithScore(node=nodes[i], score=float(scores[i])) for i in order]