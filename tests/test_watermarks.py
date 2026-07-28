"""Tests for :mod:`obsidian_librarian.rag.sync.watermarks` — diff + CRUD round-trip."""

from __future__ import annotations

import pytest

from obsidian_librarian.rag.sync import watermarks
from obsidian_librarian.rag.sync.watermarks import SyncDiff


# --------------------------------------------------------------------------- diff


def test_diff_added():
    scan = {"a.md": ("sha-a", 1.0)}
    d = watermarks.diff(scan, {})
    assert d.added == ["a.md"]
    assert d.modified == []
    assert d.deleted == []
    assert d.total == 1
    assert not d.is_empty()


def test_diff_modified_on_sha_change():
    scan = {"a.md": ("sha-new", 2.0)}
    wm = {"a.md": ("sha-old", 1.0)}
    d = watermarks.diff(scan, wm)
    assert d.added == []
    assert d.modified == ["a.md"]
    assert d.deleted == []


def test_diff_mtime_only_change_is_not_modified():
    # diff is sha-based: a touch (mtime change, same content) must not register as modified.
    scan = {"a.md": ("same-sha", 99.0)}
    wm = {"a.md": ("same-sha", 1.0)}
    d = watermarks.diff(scan, wm)
    assert d.added == []
    assert d.modified == []
    assert d.deleted == []
    assert d.is_empty()


def test_diff_deleted():
    scan = {}
    wm = {"gone.md": ("sha", 1.0)}
    d = watermarks.diff(scan, wm)
    assert d.deleted == ["gone.md"]
    assert d.added == []
    assert d.modified == []


def test_diff_mixed():
    scan = {"new.md": ("s1", 1.0), "changed.md": ("s2new", 2.0), "same.md": ("s3", 3.0)}
    wm = {"changed.md": ("s2old", 1.0), "same.md": ("s3", 1.0), "removed.md": ("s4", 1.0)}
    d = watermarks.diff(scan, wm)
    assert set(d.added) == {"new.md"}
    assert set(d.modified) == {"changed.md"}
    assert set(d.deleted) == {"removed.md"}
    assert d.total == 3


def test_syncdiff_empty_helpers():
    assert SyncDiff().is_empty()
    assert SyncDiff().total == 0
    assert not SyncDiff(added=["a"]).is_empty()


# --------------------------------------------------------------------------- CRUD round-trip


def test_connect_creates_schema_and_round_trip(tmp_path, monkeypatch):
    db = tmp_path / "wm.db"
    monkeypatch.setenv("SQLITE_PATH", str(db))
    from obsidian_librarian.config import reset_settings_cache

    reset_settings_cache()

    conn = watermarks.connect()
    # Fresh table → empty.
    assert watermarks.load_all(conn) == {}

    watermarks.upsert(conn, "notes/a.md", "sha-a", 1.0, "2026-07-21T00:00:00")
    watermarks.upsert(conn, "notes/b.md", "sha-b", 2.0, "2026-07-21T00:00:01")
    loaded = watermarks.load_all(conn)
    assert set(loaded) == {"notes/a.md", "notes/b.md"}
    assert loaded["notes/a.md"] == ("sha-a", 1.0)

    # Upsert on existing path updates in place (no duplicate row).
    watermarks.upsert(conn, "notes/a.md", "sha-a2", 1.5, "2026-07-21T00:00:02")
    loaded = watermarks.load_all(conn)
    assert loaded["notes/a.md"] == ("sha-a2", 1.5)
    assert len(loaded) == 2

    watermarks.delete(conn, "notes/b.md")
    loaded = watermarks.load_all(conn)
    assert "notes/b.md" not in loaded
    assert set(loaded) == {"notes/a.md"}

    conn.close()


def test_connect_is_idempotent_schema(tmp_path, monkeypatch):
    db = tmp_path / "wm.db"
    monkeypatch.setenv("SQLITE_PATH", str(db))
    from obsidian_librarian.config import reset_settings_cache

    reset_settings_cache()
    c1 = watermarks.connect()
    watermarks.upsert(c1, "x.md", "sx", 1.0, "t")
    c1.close()
    # Re-opening must not wipe the table.
    c2 = watermarks.connect()
    assert watermarks.load_all(c2) == {"x.md": ("sx", 1.0)}
    c2.close()


def test_connection_context_manager_closes_handle(tmp_path, monkeypatch):
    db = tmp_path / "wm.db"
    monkeypatch.setenv("SQLITE_PATH", str(db))
    from obsidian_librarian.config import reset_settings_cache

    reset_settings_cache()
    with watermarks.connection() as conn:
        watermarks.upsert(conn, "y.md", "sy", 2.0, "2026-07-22T00:00:00")
        assert watermarks.load_all(conn) == {"y.md": ("sy", 2.0)}
    # On exit the connection is closed — further use must raise.
    import sqlite3

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")