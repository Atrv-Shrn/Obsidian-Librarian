"""Tests for :mod:`obsidian_librarian.rag.query_engine` — context block + source formatting.

Exercises the pure helpers (``_context_block``, ``format_sources``) with lightweight fakes
so no live LlamaIndex / Qdrant / cloud LLM is required. The fakes mirror the
``NodeWithScore`` shape the production code reads (``.node.metadata``, ``.node.text``, ``.score``).
"""

from __future__ import annotations

from types import SimpleNamespace

from obsidian_librarian.rag import query_engine
from obsidian_librarian.rag.query_engine import _context_block, format_sources


def _node(path, title, text, heading="", tags=None, score=0.0):
    return SimpleNamespace(
        node=SimpleNamespace(
            text=text,
            metadata={
                "path": path,
                "title": title,
                "heading_path": heading,
                "tags": tags or [],
            },
        ),
        score=score,
    )


# --------------------------------------------------------------------------- _context_block


def test_context_block_numbers_and_cites_titles():
    nodes = [
        _node("a/b.md", "Bayesian Reasoning", "text A", heading="Core", score=0.9),
        _node("a/c.md", "Priors", "text C", heading="", score=0.4),
    ]
    block = _context_block(nodes)
    # Numbered, title cited as a wikilink, heading carried inline in the meta parens.
    assert "[1] source: [[Bayesian Reasoning]] (heading: Core)" in block
    assert "[2] source: [[Priors]]" in block
    # No trailing meta parens when heading_path is empty (and no other metadata present).
    assert "Priors]] (" not in block
    assert "text A" in block and "text C" in block


def test_context_block_falls_back_to_path_when_no_title():
    n = _node("a/untitled.md", "", "body")
    block = _context_block([n])
    assert "[[a/untitled.md]]" in block


def test_context_block_empty_nodes_returns_empty_string():
    assert _context_block([]) == ""


# --------------------------------------------------------------------------- format_sources


def test_format_sources_dedups_by_path_and_keeps_order():
    nodes = [
        _node("a.md", "A", "x", score=0.8),
        _node("a.md", "A", "x again", score=0.5),  # dup path → dropped
        _node("b.md", "B", "y", score=0.3),
    ]
    srcs = format_sources(nodes)
    assert [s["path"] for s in srcs] == ["a.md", "b.md"]
    # The first occurrence's score is the one kept (dedup is first-wins).
    assert srcs[0]["score"] == 0.8
    assert srcs[0]["title"] == "A"
    assert srcs[1]["tags"] == []


def test_format_sources_skips_empty_path():
    nodes = [
        _node("", "NoPath", "x"),
        _node("b.md", "B", "y", score=0.2),
    ]
    srcs = format_sources(nodes)
    assert [s["path"] for s in srcs] == ["b.md"]


def test_format_sources_empty_nodes_returns_empty_list():
    assert format_sources([]) == []


# --------------------------------------------------------------------------- query_notes empty-vault branch


def test_query_notes_empty_retrieval_returns_groundless_answer(monkeypatch):
    # When retrieval yields nothing, query_notes short-circuits without calling the LLM.
    monkeypatch.setattr(query_engine, "retrieve", lambda *a, **k: [])
    out = query_engine.query_notes("anything")
    assert out["sources"] == []
    assert out["contexts"] == []
    assert "don't have enough" in out["answer"].lower()