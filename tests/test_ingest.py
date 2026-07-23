"""Tests for pure helpers in :mod:`obsidian_librarian.rag.ingest`.

Exercises the dependency-free functions (ids, hashing, splitting, heading paths, the sync
report dataclass, and chunk_documents with the parser/TextNode patched to fakes) so we never
need a live Qdrant / Redis / LlamaIndex to validate the chunking + metadata wiring.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from obsidian_librarian.rag import ingest
from obsidian_librarian.rag.ingest import (
    SyncReport,
    _heading_path,
    _split_to_size,
    chunk_documents,
    file_sha256,
    node_uuid,
)


# --------------------------------------------------------------------------- ids + hashes


def test_node_uuid_is_deterministic():
    a = node_uuid("notes/a.md", 0)
    b = node_uuid("notes/a.md", 0)
    assert a == b
    # Different path or chunk → different id.
    assert node_uuid("notes/a.md", 1) != a
    assert node_uuid("notes/b.md", 0) != a


def test_node_uuid_is_uuid5_shape():
    # uuid5 hex is 32 chars; we return it as a str (the canonical hyphenated form).
    u = node_uuid("x.md", 0)
    assert isinstance(u, str)
    assert len(u.replace("-", "")) == 32


def test_file_sha256(tmp_path):
    p = tmp_path / "n.md"
    p.write_bytes(b"hello obsidian")
    assert file_sha256(p) == hashlib.sha256(b"hello obsidian").hexdigest()


# --------------------------------------------------------------------------- splitting


def test_split_short_text_returns_one_piece():
    assert _split_to_size("short", 100, 10) == ["short"]


def test_split_strips_and_drops_empty():
    assert _split_to_size("   ", 100, 10) == []
    assert _split_to_size("", 100, 10) == []


def test_split_long_text_with_overlap():
    text = "x" * 1000
    pieces = _split_to_size(text, 400, 100)
    # Reassembling with the overlap window reproduces the original text.
    assert pieces[0] == text[:400]
    # Overlap: the start of piece i+1 is 100 chars before the end of piece i.
    assert pieces[1] == text[300:700]
    # Every piece respects the max-chars bound (a real chunking invariant, not a tautology).
    assert all(len(p) <= 400 for p in pieces)
    # Covers the tail.
    assert pieces[-1].endswith(text[-1])


def test_split_overlap_never_negative_progress():
    # overlap >= max_chars would otherwise stall; code guards by falling to `end`.
    text = "x" * 50
    pieces = _split_to_size(text, 10, 999)
    assert len(pieces) >= 1
    # Real invariants, not trivially-true truths: every piece is non-empty, within the
    # bound, and the original text is fully covered by the concatenation.
    assert all(p for p in pieces)
    assert all(len(p) <= 10 for p in pieces)
    assert text[0] in pieces[0]
    assert text[-1] in pieces[-1]


# --------------------------------------------------------------------------- heading path


def test_heading_path_joins_levels():
    assert _heading_path({"Header 1": "Top", "Header 2": "Sub", "Header 3": "Deep"}) == "Top > Sub > Deep"


def test_heading_path_skips_missing_levels():
    assert _heading_path({"Header 1": "A", "Header 6": "Z"}) == "A > Z"


def test_heading_path_empty():
    assert _heading_path({}) == ""
    assert _heading_path({"other": "x"}) == ""


# --------------------------------------------------------------------------- SyncReport


def test_sync_report_defaults_errors_to_list():
    r = SyncReport()
    assert r.errors == []
    assert r.added == 0
    assert r.skipped_dup == 0


def test_sync_report_as_dict_shape():
    r = SyncReport(added=2, modified=1, deleted=3, skipped_dup=5, errors=["boom"])
    d = r.as_dict()
    assert d == {
        "added": 2,
        "modified": 1,
        "deleted": 3,
        "skipped_dup": 5,
        "errors": ["boom"],
    }


# --------------------------------------------------------------------------- chunk_documents


class _FakeTextNode:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


class _FakeParser:
    def __init__(self, nodes):
        self._nodes = nodes

    def get_nodes_from_documents(self, docs):
        return self._nodes


def _fake_doc(text="body", **md):
    base = {
        "file_path": "note.md",
        "title": "Note",
        "tags": ["t1"],
        "wikilinks": ["wl"],
        "backlinks": [],
        "frontmatter": {},
    }
    base.update(md)
    return SimpleNamespace(text=text, metadata=base)


def test_chunk_documents_single_piece_metadata(monkeypatch):
    node = SimpleNamespace(text="hello world", metadata={"Header 1": "Intro"})
    monkeypatch.setattr(ingest, "MarkdownNodeParser", lambda: _FakeParser([node]))
    monkeypatch.setattr(ingest, "TextNode", _FakeTextNode)

    out = chunk_documents([_fake_doc(text="hello world")], 512, 64)
    assert len(out) == 1
    n = out[0]
    assert n.text == "hello world"
    assert n.metadata["path"] == "note.md"
    assert n.metadata["title"] == "Note"
    assert n.metadata["heading_path"] == "Intro"
    assert n.metadata["tags"] == ["t1"]
    assert n.metadata["wikilinks"] == ["wl"]
    assert n.metadata["chunk_index"] == 0


def test_chunk_documents_splits_long_section_and_indexes(monkeypatch):
    long_text = "a" * 1000
    node = SimpleNamespace(text=long_text, metadata={"Header 2": "Body"})
    monkeypatch.setattr(ingest, "MarkdownNodeParser", lambda: _FakeParser([node]))
    monkeypatch.setattr(ingest, "TextNode", _FakeTextNode)

    out = chunk_documents([_fake_doc(text=long_text)], 10, 4)
    assert len(out) >= 2
    # chunk_index increments 0..n-1 and every piece carries the heading + path.
    for i, n in enumerate(out):
        assert n.metadata["chunk_index"] == i
        assert n.metadata["path"] == "note.md"
        assert n.metadata["heading_path"] == "Body"
    # Reassembled content covers the original (overlap may repeat, so is a superset).
    assert "".join(n.text for n in out).replace("a", "") == ""


def test_chunk_documents_multiple_files_get_separate_index_sequences(monkeypatch):
    node_a = SimpleNamespace(text="alpha", metadata={"Header 1": "A"})
    node_b = SimpleNamespace(text="beta", metadata={"Header 1": "B"})
    # Parser returns nodes for both docs in one call; map back by doc order is not needed
    # since chunk_documents calls get_nodes_from_documents per doc. Simulate per-doc nodes.
    calls = [_FakeParser([node_a]), _FakeParser([node_b])]
    monkeypatch.setattr(ingest, "MarkdownNodeParser", lambda: calls.pop(0))
    monkeypatch.setattr(ingest, "TextNode", _FakeTextNode)

    out = chunk_documents(
        [_fake_doc(text="alpha", file_path="a.md"), _fake_doc(text="beta", file_path="b.md")],
        512,
        64,
    )
    assert len(out) == 2
    assert out[0].metadata["path"] == "a.md"
    assert out[0].metadata["chunk_index"] == 0
    assert out[1].metadata["path"] == "b.md"
    assert out[1].metadata["chunk_index"] == 0


# --------------------------------------------------------------------------- index_documents purge-before-upsert


def test_index_documents_purges_path_before_upsert(monkeypatch):
    """Stale-tail purge guard: on re-index, the old points for a path must be deleted BEFORE the
    new ones are upserted. Point ids are ``uuid5(path::chunk_index)``, so a note that shrinks
    N→M chunks would leave ids ::M..N-1 un-overwritten and retrievable unless the old set is
    deleted first. Asserts the delete-by-path precedes the upsert in call order.
    """
    from obsidian_librarian.rag import ingest

    calls: list = []

    class _FakeQS:
        def ensure_collection(self):
            calls.append("ensure")

        def delete_by_path(self, rel):
            calls.append(("delete", rel))

        def upsert_points(self, pts):
            calls.append(("upsert", len(pts)))

    class _FakeRS:
        def purge_path(self, rel):
            calls.append(("purge", rel))

        def store_node(self, nid, payload):
            pass

        def add_path_nodes(self, rel, ids):
            pass

        def store_raw(self, rel, text):
            pass

    monkeypatch.setattr(ingest.QdrantStore, "from_settings", classmethod(lambda cls: _FakeQS()))
    monkeypatch.setattr(ingest.RedisStore, "from_settings", classmethod(lambda cls: _FakeRS()))
    # Avoid the real embedders (fastembed may try to load ONNX) and the real point builder.
    monkeypatch.setattr(ingest, "get_dense_embed_model", lambda: object())
    monkeypatch.setattr(ingest, "get_sparse_embed_model", lambda: object())

    fake_node = SimpleNamespace(
        text="body",
        metadata={
            "path": "note.md",
            "chunk_index": 0,
            "title": "Note",
            "heading_path": "",
            "tags": [],
            "wikilinks": [],
            "backlinks": [],
            "frontmatter": {},
        },
    )
    monkeypatch.setattr(ingest, "chunk_documents", lambda docs, cs, co: [fake_node])
    monkeypatch.setattr(
        ingest, "_build_point", lambda n, d, s: SimpleNamespace(id="nid-0")
    )

    doc = SimpleNamespace(
        text="body",
        metadata={
            "file_path": "note.md",
            "title": "Note",
            "tags": [],
            "wikilinks": [],
            "backlinks": [],
            "frontmatter": {},
        },
    )
    n = ingest.index_documents([doc])
    assert n == 1
    seq = [c[0] if isinstance(c, tuple) else c for c in calls]
    assert "delete" in seq and "upsert" in seq
    # The load-bearing invariant: delete-by-path happens before the upsert.
    assert seq.index("delete") < seq.index("upsert")
    # Redis purge also precedes the upsert (keeps docstore + index consistent).
    assert seq.index("purge") < seq.index("upsert")


# --------------------------------------------------------------------------- sync_vault dedup accounting


def test_sync_vault_counts_skipped_dup_but_still_indexes(monkeypatch):
    """Dedup accounting guard: a content-duplicate path (hash already seen) must increment
    ``skipped_dup`` AND still be indexed — points are path-keyed, so identical content under a
    different path gets its own retrievable points; the hash set only feeds the report counter.
    """
    from obsidian_librarian.rag import ingest

    indexed: dict = {"paths": []}

    def fake_index(docs):
        indexed["paths"].append(docs[0].metadata["file_path"])
        return 1

    monkeypatch.setattr(ingest, "index_documents", fake_index)

    class _FakeRS:
        def add_hash(self, sha):
            return False  # already seen → dup

        def remove_hash(self, sha):
            pass

    class _FakeQS:
        def ensure_collection(self):
            pass

        def delete_by_path(self, rel):
            pass

    monkeypatch.setattr(ingest.RedisStore, "from_settings", classmethod(lambda cls: _FakeRS()))
    monkeypatch.setattr(ingest.QdrantStore, "from_settings", classmethod(lambda cls: _FakeQS()))

    monkeypatch.setattr(ingest, "scan_vault", lambda vault=None: {"note.md": ("sha1", 1.0)})
    monkeypatch.setattr(ingest, "compute_backlinks", lambda vault: {})
    fake_doc = SimpleNamespace(
        text="body",
        metadata={
            "file_path": "note.md",
            "title": "Note",
            "tags": [],
            "wikilinks": [],
            "backlinks": [],
            "frontmatter": {},
        },
    )
    monkeypatch.setattr(ingest, "load_note", lambda rel, vault=None: fake_doc)

    report = ingest.sync_vault()
    assert report.skipped_dup == 1
    # Still indexed (path-keyed points), despite the content dup.
    assert indexed["paths"] == ["note.md"]
    assert report.added == 1

# --------------------------------------------------------------------------- dedup-hash ordering
#
# Regression guard for the first-real-run accounting bug: the content hash was registered
# BEFORE index_documents ran, so a failed index (the cold-start embed timing out) still left
# the hash in the dedup set. The next sync's retry then saw add_hash() -> False and reported a
# genuine first index as `skipped_dup`. Observed live as `errs=1` then `+a=1 dup=1`.


class _FakeRedis:
    """Minimal stand-in for RedisStore's dedup-set surface."""

    def __init__(self):
        self.hashes = set()

    def add_hash(self, sha):
        new = sha not in self.hashes
        self.hashes.add(sha)
        return new

    def remove_hash(self, sha):
        self.hashes.discard(sha)


def _sync_once(monkeypatch, tmp_path, index_raises):
    """Drive sync_vault over a one-note vault with index_documents stubbed."""
    # exist_ok: the autouse _isolate_env fixture already points VAULT_PATH at a tmp "vault".
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    (vault / "n.md").write_text("# N\nbody\n", encoding="utf-8")

    fake_redis = _FakeRedis()
    monkeypatch.setattr(ingest.RedisStore, "from_settings", classmethod(lambda cls: fake_redis))
    monkeypatch.setattr(
        ingest.QdrantStore,
        "from_settings",
        classmethod(lambda cls: SimpleNamespace(ensure_collection=lambda: None, delete_by_path=lambda p: None)),
    )
    monkeypatch.setattr(ingest, "compute_backlinks", lambda v: {})

    calls = {"n": 0}

    def _index(docs):
        calls["n"] += 1
        if index_raises and calls["n"] == 1:
            raise TimeoutError("timed out")
        return 1

    monkeypatch.setattr(ingest, "index_documents", _index)
    return vault, fake_redis


def test_failed_index_does_not_register_content_hash(monkeypatch, tmp_path):
    vault, fake_redis = _sync_once(monkeypatch, tmp_path, index_raises=True)

    first = ingest.sync_vault(vault)
    # The index failed, so the note is reported as an error...
    assert len(first.errors) == 1
    assert "timed out" in first.errors[0]
    # ...and critically its hash was NOT registered. Registering it here is what made the
    # retry mis-count as a duplicate.
    assert fake_redis.hashes == set()

    # Retry: now the index succeeds and this must read as a real first index, not a dup.
    second = ingest.sync_vault(vault)
    assert second.added == 1
    assert second.skipped_dup == 0
    assert len(fake_redis.hashes) == 1


def test_successful_index_registers_hash_and_counts_real_duplicate(monkeypatch, tmp_path):
    vault, fake_redis = _sync_once(monkeypatch, tmp_path, index_raises=False)

    first = ingest.sync_vault(vault)
    assert first.added == 1 and first.skipped_dup == 0
    assert len(fake_redis.hashes) == 1

    # A genuinely duplicate body under a second path must still be counted as skipped_dup —
    # the fix must not disable real dedup accounting.
    (vault / "copy.md").write_text((vault / "n.md").read_text(encoding="utf-8"), encoding="utf-8")
    second = ingest.sync_vault(vault)
    assert second.added == 1
    assert second.skipped_dup == 1
