"""Tests for :mod:`obsidian_librarian.agent.observability` — the Langfuse enablement gate.

The autouse env fixture sets no ``LANGFUSE_*`` keys, so the disabled path is the default; the
enabled path is exercised by setting keys and stubbing the v3 handler import so we never make a
real Langfuse client call.
"""

from __future__ import annotations

import sys
import types

import pytest

from obsidian_librarian.agent import observability


def test_langfuse_enabled_false_without_keys():
    # Autouse fixture sets no LANGFUSE_* keys.
    assert observability.langfuse_enabled() is False


def test_langfuse_enabled_true_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert observability.langfuse_enabled() is True


def test_get_langfuse_handler_none_when_disabled():
    assert observability.get_langfuse_handler() is None


def test_get_langfuse_handler_enabled_returns_v3_handler(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    # Don't run the real Langfuse(...) init; force success and stub the v3 handler import.
    monkeypatch.setattr(observability, "_LANGFUSE_INIT", False)
    monkeypatch.setattr(observability, "_init_langfuse_client", lambda: True)

    class _V3Handler:
        tag = "v3"

        def __init__(self, *a, **k):
            pass

    fake = types.ModuleType("langfuse.langchain")
    fake.CallbackHandler = _V3Handler
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake)

    h = observability.get_langfuse_handler()
    assert h is not None
    assert isinstance(h, _V3Handler)


def test_get_langfuse_handler_never_raises_on_import_failure(monkeypatch):
    # Keys set, but both v3 and v2 imports blow up → must return None, not raise.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(observability, "_LANGFUSE_INIT", False)
    monkeypatch.setattr(observability, "_init_langfuse_client", lambda: False)
    # Force v2 import to fail too by pointing it at a broken module.
    broken = types.ModuleType("langfuse.callback")

    def _raise(*a, **k):
        raise ImportError("no v2")

    broken.CallbackHandler = _raise
    monkeypatch.setitem(sys.modules, "langfuse.callback", broken)

    assert observability.get_langfuse_handler() is None