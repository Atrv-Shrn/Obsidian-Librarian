"""SQLite sync watermarks — the single source of truth for what is indexed.

Table ``watermarks(path PK, sha256, mtime, indexed_at)``. A sync run diffs the live
filesystem scan against this table to compute added / modified / deleted paths, then
rewrites it. Sync deletes are **index-only** — this module never touches vault files.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator

from ...config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watermarks (
    path        TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    mtime       REAL NOT NULL,
    indexed_at  TEXT NOT NULL
);
"""


@dataclass
class SyncDiff:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def is_empty(self) -> bool:
        return self.total == 0


def connect() -> sqlite3.Connection:
    s = get_settings()
    s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(s.sqlite_path.as_posix())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """Context-managed :func:`connect` — closes the handle on exit.

    Use this in production paths (``with watermarks.connection() as conn:``) so a sync run
    never leaks a SQLite handle. :func:`connect` stays available for tests that want a raw
    connection they close themselves.
    """
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def load_all(conn: sqlite3.Connection) -> Dict[str, tuple[str, float]]:
    rows = conn.execute("SELECT path, sha256, mtime FROM watermarks").fetchall()
    return {r["path"]: (r["sha256"], float(r["mtime"])) for r in rows}


def upsert(conn: sqlite3.Connection, path: str, sha256: str, mtime: float, indexed_at: str) -> None:
    conn.execute(
        "INSERT INTO watermarks(path, sha256, mtime, indexed_at) VALUES(?,?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime=excluded.mtime, "
        "indexed_at=excluded.indexed_at",
        (path, sha256, mtime, indexed_at),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("DELETE FROM watermarks WHERE path = ?", (path,))
    conn.commit()


def diff(scan: Dict[str, tuple[str, float]], watermarks: Dict[str, tuple[str, float]]) -> SyncDiff:
    """Compare a live scan (path → (sha256, mtime)) against the watermark table."""
    out = SyncDiff()
    for path, (sha, _mtime) in scan.items():
        if path not in watermarks:
            out.added.append(path)
        elif watermarks[path][0] != sha:
            out.modified.append(path)
    for path in watermarks:
        if path not in scan:
            out.deleted.append(path)
    return out