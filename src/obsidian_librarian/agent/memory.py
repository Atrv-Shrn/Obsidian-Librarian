"""Memory — the async ``AsyncSqliteSaver`` checkpointer.

Chat clients (Open WebUI, Copilot) resend the full conversation each turn, so within-run and
across-turn message history rides along in ``messages`` for free. The checkpointer's real job
here is to **carry a pending write across the propose→confirm turns** and to survive restarts.
It's keyed by a `thread_id` (derived from the conversation in the API layer). Lives in `/data`.

Async, not sync: the agent graph is driven with ``ainvoke`` / ``astream_events`` (async). The
sync ``SqliteSaver`` raises ``NotImplementedError`` on every ``aget_*`` / ``aput`` method, so an
async agent turn would crash on its first real run. ``AsyncSqliteSaver`` (from
``langgraph.checkpoint.sqlite.aio``) is the async-native sibling and owns its own ``aiosqlite``
connection — we must NOT hand it a pre-opened ``sqlite3`` connection (that leaks a sync handle
and breaks the event loop). We open the connection with ``aiosqlite`` and let the saver own it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..config import get_settings

# Module-level cache replaces ``lru_cache``: the saver holds an async ``aiosqlite`` connection
# opened with ``await``, which can't happen inside a sync lru_cached function.
_SAVER: Any = None
# Guards the check-then-set build. The agent eval runs golden tasks concurrently
# (``asyncio.gather``), so without a lock several coroutines could all see ``_SAVER is None``
# on the first turn and each open a redundant connection + ``setup()`` (leaking the losers).
# Python 3.11+ defers loop binding for ``asyncio.Lock``, so creating it at import is safe.
_SAVER_LOCK = asyncio.Lock()


async def get_checkpointer() -> AsyncSqliteSaver:
    """Return the async SqliteSaver singleton, opening + setting it up on first call."""
    global _SAVER
    if _SAVER is not None:
        return _SAVER
    async with _SAVER_LOCK:
        # Re-check under the lock: the first coroutine through may have already built it.
        if _SAVER is not None:
            return _SAVER
        import aiosqlite

        s = get_settings()
        s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(s.sqlite_path.as_posix())
        # WAL + a busy timeout so the agent checkpointer and the sync watermarks connection
        # (same SQLite file) don't trip "database is locked" under concurrent access.
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        _SAVER = AsyncSqliteSaver(conn)
        # Create the checkpoint tables once. Idempotent.
        await _SAVER.setup()
        return _SAVER


def reset_checkpointer_cache() -> None:
    """Test/reconnect helper: drop the cached saver so the next call opens a fresh connection."""
    global _SAVER
    _SAVER = None