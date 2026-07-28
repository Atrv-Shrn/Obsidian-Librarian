"""Tests for the async ``AsyncSqliteSaver`` checkpointer singleton.

``get_checkpointer`` is double-checked-locked and caches the saver in a module global; the
real ``aiosqlite`` + ``AsyncSqliteSaver`` are stubbed away (heavy deps), so we inject a fake
``aiosqlite`` module and a fake saver class and assert the singleton + reset contract:

* two calls in the same process return the *same* saver,
* after ``reset_checkpointer_cache`` the next call builds a *new* one.

Everything runs under a single ``asyncio.run`` so the module-level ``_SAVER_LOCK`` binds to one
live event loop (a per-call ``asyncio.run`` would bind it to a closed loop and crash on call 3).
"""

from __future__ import annotations

import asyncio
import sys
import types

from obsidian_librarian.agent import memory


def test_get_checkpointer_singleton_then_reset(monkeypatch):
    class _FakeConn:
        async def execute(self, sql, *a, **k):
            return None

        async def close(self):
            pass

    fake_aiosqlite = types.ModuleType("aiosqlite")

    async def _connect(path):
        return _FakeConn()

    fake_aiosqlite.connect = _connect
    monkeypatch.setitem(sys.modules, "aiosqlite", fake_aiosqlite)

    class _FakeSaver:
        def __init__(self, conn):
            self.conn = conn

        async def setup(self):
            pass

    monkeypatch.setattr(memory, "AsyncSqliteSaver", _FakeSaver)

    async def scenario():
        memory.reset_checkpointer_cache()
        a = await memory.get_checkpointer()
        b = await memory.get_checkpointer()
        # Singleton: the second call reuses the cached saver, no second setup().
        assert a is b
        memory.reset_checkpointer_cache()
        c = await memory.get_checkpointer()
        # Reset drops the cache → a fresh saver is built.
        assert c is not a

    asyncio.run(scenario())


def test_reset_checkpointer_cache_clears_singleton(monkeypatch):
    # reset_checkpointer_cache is the test/reconnect hook; it must zero the global so the
    # next get_checkpointer rebuilds (e.g. after a settings/sqlite path change).
    built = {"n": 0}

    class _FakeConn:
        async def execute(self, sql, *a, **k):
            return None

    fake_aiosqlite = types.ModuleType("aiosqlite")

    async def _connect(path):
        return _FakeConn()

    fake_aiosqlite.connect = _connect
    monkeypatch.setitem(sys.modules, "aiosqlite", fake_aiosqlite)

    class _FakeSaver:
        def __init__(self, conn):
            built["n"] += 1
            self.conn = conn

        async def setup(self):
            pass

    monkeypatch.setattr(memory, "AsyncSqliteSaver", _FakeSaver)

    async def scenario():
        memory.reset_checkpointer_cache()
        await memory.get_checkpointer()
        assert built["n"] == 1
        # Reset without a follow-up build must not construct anything new.
        memory.reset_checkpointer_cache()
        assert built["n"] == 1
        await memory.get_checkpointer()
        assert built["n"] == 2

    asyncio.run(scenario())