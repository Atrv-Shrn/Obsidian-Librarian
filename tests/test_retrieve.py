"""Tests for :mod:`obsidian_librarian.rag.retrieve` — rerank ordering + top-k cutoff.

Patches the Qdrant fusion (``hybrid_search``) and the FastEmbed reranker so the cross-encoder
re-scoring path can be exercised deterministically without a live Qdrant / ONNX model. The
fakes mirror the shapes the production code reads: ``ScoredPoint.payload`` / ``.score`` for
fused candidates, and a ``.rerank(query, texts)`` yielding plain float scores in input order.
"""

from __future__ import annotations

from types import SimpleNamespace

from obsidian_librarian.rag import retrieve as retrieve_mod


def _point(path, text, score=0.0):
    return SimpleNamespace(
        score=score,
        payload={
            "text": text,
            "path": path,
            "title": path.rsplit("/", 1)[-1][:-3] if path.endswith(".md") else path,
            "heading_path": "",
            "tags": [],
            "wikilinks": [],
            "chunk_index": 0,
        },
    )


class _FakeReranker:
    """Yields the scores it was handed, in input order (matches FastEmbed's contract)."""

    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, texts):
        return iter(list(self._scores))


def test_retrieve_orders_by_rerank_score_descending(monkeypatch):
    # Fused candidates in an arbitrary order; reranker scores say the 2nd is best, then 0th, then 1st.
    pts = [_point("a.md", "alpha"), _point("b.md", "beta"), _point("c.md", "gamma")]
    monkeypatch.setattr(retrieve_mod, "hybrid_search", lambda *a, **k: pts)
    monkeypatch.setattr(retrieve_mod, "get_reranker", lambda: _FakeReranker([0.2, 0.9, 0.5]))

    out = retrieve_mod.retrieve("q", top_k=3)
    assert [o.node.metadata["path"] for o in out] == ["b.md", "c.md", "a.md"]
    # Scores are the cross-encoder scores, attached verbatim.
    assert out[0].score == 0.9
    assert out[1].score == 0.5
    assert out[2].score == 0.2


def test_retrieve_top_k_truncates_after_sorting(monkeypatch):
    pts = [_point(f"n{i}.md", f"t{i}") for i in range(5)]
    # Scores decreasing in input order: best two are n0, n1.
    monkeypatch.setattr(retrieve_mod, "hybrid_search", lambda *a, **k: pts)
    monkeypatch.setattr(retrieve_mod, "get_reranker", lambda: _FakeReranker([0.9, 0.8, 0.1, 0.05, 0.0]))

    out = retrieve_mod.retrieve("q", top_k=2)
    assert len(out) == 2
    assert [o.node.metadata["path"] for o in out] == ["n0.md", "n1.md"]


def test_retrieve_empty_fusion_returns_empty(monkeypatch):
    monkeypatch.setattr(retrieve_mod, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieve_mod, "get_reranker", lambda: _FakeReranker([]))
    assert retrieve_mod.retrieve("q") == []


def test_retrieve_ties_keep_input_order(monkeypatch):
    # Equal scores → Python's sort is stable, so input order is preserved.
    pts = [_point("z.md", "z"), _point("y.md", "y"), _point("x.md", "x")]
    monkeypatch.setattr(retrieve_mod, "hybrid_search", lambda *a, **k: pts)
    monkeypatch.setattr(retrieve_mod, "get_reranker", lambda: _FakeReranker([0.5, 0.5, 0.5]))
    out = retrieve_mod.retrieve("q", top_k=3)
    assert [o.node.metadata["path"] for o in out] == ["z.md", "y.md", "x.md"]