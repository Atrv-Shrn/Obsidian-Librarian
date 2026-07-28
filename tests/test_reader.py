"""Tests for pure helpers in :mod:`obsidian_librarian.rag.reader`.

The YAML/wikilink/tag parsers and the single-note loader are dependency-free; we exercise
them directly and stub the LlamaIndex ``Document`` only where ``load_note`` constructs one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from obsidian_librarian.rag import reader
from obsidian_librarian.rag.reader import (
    _extract_tags,
    _extract_wikilinks,
    _parse_frontmatter,
    _to_list,
    load_note,
)


# --------------------------------------------------------------------------- frontmatter


def test_frontmatter_none_when_no_fence():
    fm, body = _parse_frontmatter("Just prose, no frontmatter.")
    assert fm == {}
    assert body == "Just prose, no frontmatter."


def test_frontmatter_unclosed_fence_returns_whole_text():
    # No closing `\n---` → not frontmatter; whole text returned as body.
    text = "---\ntitle: X\nthis never closes"
    fm, body = _parse_frontmatter(text)
    assert fm == {}
    assert body == text


def test_frontmatter_inline_scalar_and_empty():
    fm, body = _parse_frontmatter("---\ntitle: Hello\nempty:\n---\nbody here")
    assert fm["title"] == "Hello"
    assert fm["empty"] == []
    assert body == "body here"


def test_frontmatter_inline_list():
    fm, _ = _parse_frontmatter("---\ntags: [a, b, 'c']\n---\n")
    assert fm["tags"] == ["a", "b", "c"]


def test_frontmatter_yaml_block_list():
    fm, _ = _parse_frontmatter("---\ntags:\n  - alpha\n  - beta\n---\n")
    assert fm["tags"] == ["alpha", "beta"]


def test_frontmatter_skips_comments_and_blank_lines():
    fm, _ = _parse_frontmatter("---\n# a comment\ntitle: Real\n\n---\n")
    assert fm == {"title": "Real"}
    assert "a comment" not in fm


# --------------------------------------------------------------------------- wikilinks


def test_wikilinks_basic():
    assert _extract_wikilinks("see [[Note]] and [[Other]]") == ["Note", "Other"]


def test_wikilinks_alias_and_block():
    assert _extract_wikilinks("[[Note|alias]] [[Note#^block]]") == ["Note|alias", "Note#^block"]


def test_wikilinks_none():
    assert _extract_wikilinks("plain text") == []


# --------------------------------------------------------------------------- tags


def test_tags_basic():
    assert _extract_tags("a #topic note #proj/sub") == ["topic", "proj/sub"]


def test_tags_negative_lookbehind_word():
    # `word#tag` is not a tag (no boundary before #).
    assert _extract_tags("foo#bar") == []


def test_tags_negative_lookbehind_html_entity():
    # `&#128;` emoji entities should not be treated as tags.
    assert _extract_tags("smile &#128; ok") == []


def test_tags_standalone_hash_not_a_tag():
    # `#` alone with no following word char → no match.
    assert _extract_tags("a # by itself") == []


# --------------------------------------------------------------------------- _to_list


def test_to_list_none():
    assert _to_list(None) == []


def test_to_list_scalar():
    assert _to_list("x") == ["x"]
    assert _to_list(7) == ["7"]


def test_to_list_list():
    assert _to_list(["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- load_note


class _FakeDocument:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


def test_load_note_nonexistent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(reader, "Document", _FakeDocument)
    assert load_note("missing.md", vault_path=tmp_path) is None


def test_load_note_non_md_returns_none(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    monkeypatch.setattr(reader, "Document", _FakeDocument)
    assert load_note("notes.txt", vault_path=tmp_path) is None


def test_load_note_parses_and_returns_document(monkeypatch, tmp_path):
    note = tmp_path / "n.md"
    note.write_text(
        "---\ntitle: My Note\ntags: [imported]\n---\n# Heading\nbody #inline [[Link]]",
        encoding="utf-8",
    )
    monkeypatch.setattr(reader, "Document", _FakeDocument)

    doc = load_note("n.md", vault_path=tmp_path)
    assert doc is not None
    assert isinstance(doc, _FakeDocument)
    assert doc.metadata["file_path"] == "n.md"
    assert doc.metadata["title"] == "My Note"
    # tags come from frontmatter + inline, deduped + sorted.
    assert "imported" in doc.metadata["tags"]
    assert "inline" in doc.metadata["tags"]
    assert doc.metadata["wikilinks"] == ["Link"]
    assert doc.metadata["frontmatter"]["title"] == "My Note"
    assert "mtime" in doc.metadata
    # Body has frontmatter stripped.
    assert "title: My Note" not in doc.text
    assert "Heading" in doc.text


def test_load_note_rejects_traversal_out_of_vault(monkeypatch, tmp_path):
    # A ``..`` path that escapes the vault must NOT read the host file, even though that file
    # exists and is a .md — containment is checked before the read. The autouse isolation
    # fixture already created ``tmp_path/vault``; we drop a secret *sibling* of it.
    vault = tmp_path / "vault"
    secret = tmp_path / "secret.md"  # sibling of the vault dir, i.e. outside it
    secret.write_text("HOST SECRET", encoding="utf-8")
    try:
        monkeypatch.setattr(reader, "Document", _FakeDocument)
        # vault/../secret.md resolves to tmp_path/secret.md — outside the vault.
        assert load_note("../secret.md", vault_path=vault) is None
    finally:
        secret.unlink(missing_ok=True)


def test_load_note_rejects_absolute_path(monkeypatch, tmp_path):
    # An absolute path is not relative to the vault; ``vault / abs`` collapses to ``abs``,
    # which resolves outside the vault → must be rejected even when the file exists.
    vault = tmp_path / "vault"
    outside = tmp_path / "abs_note.md"
    outside.write_text("x", encoding="utf-8")
    try:
        monkeypatch.setattr(reader, "Document", _FakeDocument)
        assert load_note(str(outside), vault_path=vault) is None
    finally:
        outside.unlink(missing_ok=True)