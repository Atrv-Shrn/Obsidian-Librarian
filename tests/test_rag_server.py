"""Tests for the RAG-MCP server's vault-docs TTL cache + reindex cache-reset.

``fastmcp`` is a heavy dep that's stubbed when missing, which would shadow ``@mcp.tool()``-decorated
functions (the stub's ``tool()`` returns a stub, not the original fn). To exercise the *real*
``reindex`` tool + ``_vault_docs`` cache wiring, we install a tiny fake ``fastmcp`` module whose
``FastMCP.tool()`` is an identity decorator, then reload ``rag_server`` so the tool functions
survive decoration as real callables. ``test_smoke_imports`` only checks ``mod.__name__``, so the
reload leaves it unaffected.
"""

from __future__ import annotations

import importlib
import sys
import types


def _install_identity_fastmcp() -> None:
    """Put a fake ``fastmcp`` in ``sys.modules`` whose ``FastMCP.tool()`` returns the fn unchanged."""
    fake = types.ModuleType("fastmcp")

    class _FastMCP:
        def __init__(self, name=None, **kww):
            self.name = name

        def tool(self, *a, **k):
            def deco(fn):
                return fn

            return deco

        def run(self, *a, **k):
            pass

    fake.FastMCP = _FastMCP
    sys.modules["fastmcp"] = fake


# Install before importing rag_server so the @mcp.tool() decorators preserve the real functions.
_install_identity_fastmcp()
from obsidian_librarian.mcp import rag_server as rs  # noqa: E402

importlib.reload(rs)  # noqa: E402,F401  re-decorate with the identity-decorator fake fastmcp


# --------------------------------------------------------------------------- _vault_docs TTL


def test_vault_docs_caches_within_ttl_then_reloads_after_reset(monkeypatch):
    calls = {"n": 0}

    def fake_load_vault():
        calls["n"] += 1
        return [{"file_path": f"n{calls['n']}.md"}]

    monkeypatch.setattr(rs, "load_vault", fake_load_vault)
    rs._reset_vault_docs_cache()

    a = rs._vault_docs()
    b = rs._vault_docs()
    # Same cached snapshot returned within the TTL window; the vault is read exactly once.
    assert a is b
    assert calls["n"] == 1

    rs._reset_vault_docs_cache()
    c = rs._vault_docs()
    # After reset, the next call re-reads the vault → a fresh snapshot (and a fresh list object).
    assert c is not a
    assert calls["n"] == 2


# --------------------------------------------------------------------------- get_backlinks


def _doc(file_path, title, wikilinks):
    """A minimal stand-in for a LlamaIndex Document: only ``.metadata`` is read by the tools."""
    from types import SimpleNamespace

    return SimpleNamespace(metadata={"file_path": file_path, "title": title, "wikilinks": wikilinks})


def test_get_backlinks_matches_frontmatter_title(monkeypatch):
    """get_backlinks must match wikilink targets against the target note's frontmatter title,
    not just its stem — mirroring ``compute_backlinks``, which indexes by title AND stem. A note
    ``bayes.md`` titled ``Bayesian Reasoning`` is backlinked by ``[[Bayesian Reasoning]]``;
    stem-only matching would miss that linker and return a set that contradicts the
    ``backlinks`` field stored in the Qdrant payload."""
    docs = [
        _doc("bayes.md", "Bayesian Reasoning", []),
        _doc("priors.md", "Priors", ["Bayesian Reasoning"]),  # links by title
        _doc("notes.md", "Notes", ["bayes"]),  # links by stem
        _doc("unrelated.md", "Unrelated", ["something-else"]),
    ]
    monkeypatch.setattr(rs, "load_vault", lambda: docs)
    rs._reset_vault_docs_cache()

    res = rs.get_backlinks("bayes.md")
    assert res["exists"] is True and res["resolved_path"] == "bayes.md"
    back = res["backlinks"]
    # Both the title-linker and the stem-linker must surface; the unrelated note must not.
    assert "priors.md" in back
    assert "notes.md" in back
    assert "unrelated.md" not in back


def test_get_backlinks_still_matches_when_no_frontmatter_title(monkeypatch):
    """A note with no frontmatter title falls back to the stem (as ``compute_backlinks`` does),
    so stem-targeted wikilinks still resolve. The title branch must not break this default."""
    docs = [
        _doc("bayes.md", "", []),  # empty title → stem is the only identity
        _doc("notes.md", "Notes", ["bayes"]),
    ]
    monkeypatch.setattr(rs, "load_vault", lambda: docs)
    rs._reset_vault_docs_cache()

    res = rs.get_backlinks("bayes.md")
    assert res["exists"] is True
    assert res["backlinks"] == ["notes.md"]


def test_get_backlinks_resolves_bare_name_and_reports_existence(monkeypatch):
    """The exact bug from stress testing: asked for backlinks of 'Embeddings' (a bare name in a
    subfolder), the agent claimed the note didn't exist. get_backlinks must resolve the bare name
    to its real path and report exists=True, so the agent can't hallucinate a missing note."""
    docs = [
        _doc("RAG Pipeline Basics/Embeddings.md", "Embeddings", []),  # exists, zero inbound links
        _doc("RAG Pipeline Basics/What is RAG.md", "What is RAG", ["Embeddings"]),
    ]
    monkeypatch.setattr(rs, "load_vault", lambda: docs)
    rs._reset_vault_docs_cache()

    res = rs.get_backlinks("Embeddings")  # bare name, no path, no .md
    assert res["exists"] is True
    assert res["resolved_path"] == "RAG Pipeline Basics/Embeddings.md"
    assert res["backlinks"] == ["RAG Pipeline Basics/What is RAG.md"]


def test_get_backlinks_reports_nonexistent(monkeypatch):
    docs = [_doc("a.md", "A", ["Ghost"])]  # links to a note that isn't in the vault
    monkeypatch.setattr(rs, "load_vault", lambda: docs)
    rs._reset_vault_docs_cache()
    res = rs.get_backlinks("Ghost")
    assert res["exists"] is False and res["resolved_path"] is None
    # Even a non-existent target can have inbound links pointing at the broken wikilink.
    assert res["backlinks"] == ["a.md"]


# --------------------------------------------------------------------------- reindex resets cache


def test_reindex_tool_resets_vault_docs_cache(monkeypatch):
    """``reindex`` must invalidate the vault-docs cache so backlinks/recency reflect the post-sync
    snapshot on the next call. Asserts the reset fires exactly once per reindex call."""
    resets = {"n": 0}

    def fake_reindex(path=None):
        return {"path": path, "ok": True}

    monkeypatch.setattr(rs.ingest, "reindex", fake_reindex)

    orig = rs._reset_vault_docs_cache

    def counting_reset():
        resets["n"] += 1
        orig()

    monkeypatch.setattr(rs, "_reset_vault_docs_cache", counting_reset)

    out = rs.reindex(path="note.md")
    assert out == {"path": "note.md", "ok": True}
    assert resets["n"] == 1


def test_reindex_tool_calls_ingest_reindex_with_path(monkeypatch):
    # The tool must forward the path arg to ingest.reindex (single-note vs whole-vault).
    received = {}

    def fake_reindex(path=None):
        received["path"] = path
        return {"ok": True}

    monkeypatch.setattr(rs.ingest, "reindex", fake_reindex)
    monkeypatch.setattr(rs, "_reset_vault_docs_cache", lambda: None)

    rs.reindex(path="notes/x.md")
    assert received["path"] == "notes/x.md"

    rs.reindex()
    assert received["path"] is None