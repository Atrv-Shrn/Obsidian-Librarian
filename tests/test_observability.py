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


# --------------------------------------------------------------------------- scoring


class _FakeClient:
    """Stand-in for the v3 Langfuse client, recording create_score calls."""

    def __init__(self, trace_id="tr-abc"):
        self.scores: list = []
        self._trace_id = trace_id
        self.flushed = 0

    def create_score(self, *, name, value, trace_id=None, comment=None, **kw):
        self.scores.append({"name": name, "value": value, "trace_id": trace_id, "comment": comment})

    def get_current_trace_id(self):
        return self._trace_id

    def flush(self):
        self.flushed += 1


def _enable(monkeypatch, client):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: client)


def test_score_trace_uses_create_score_not_the_v2_score_method(monkeypatch):
    """v3 has no ``lf.score()`` — only ``create_score``.

    The old eval called ``Langfuse().score(...)``, which raised ``AttributeError`` on v3 and was
    swallowed into a log warning, so **no score ever reached Langfuse**. Pin the v3 API.
    """
    client = _FakeClient()
    _enable(monkeypatch, client)

    assert observability.score_trace("agent_task_pass", 1.0, "tr-123", comment="task-a") is True
    assert client.scores == [
        {"name": "agent_task_pass", "value": 1.0, "trace_id": "tr-123", "comment": "task-a"}
    ]


def test_score_trace_refuses_to_submit_without_a_trace_id(monkeypatch):
    """An empty trace id is what orphaned every score; refuse rather than send ``""``."""
    client = _FakeClient()
    _enable(monkeypatch, client)

    assert observability.score_trace("agent_task_pass", 1.0, None) is False
    assert observability.score_trace("agent_task_pass", 1.0, "") is False
    assert client.scores == []


def test_score_trace_noop_when_langfuse_disabled():
    # No keys (autouse fixture) → no client, no raise, reports not-submitted.
    assert observability.score_trace("agent_task_pass", 1.0, "tr-1") is False


def test_score_trace_never_raises_when_backend_errors(monkeypatch):
    class _Boom(_FakeClient):
        def create_score(self, **kw):
            raise RuntimeError("langfuse down")

    _enable(monkeypatch, _Boom())
    # Telemetry must never break the caller.
    assert observability.score_trace("agent_task_pass", 1.0, "tr-1") is False


def test_current_trace_id_reads_from_the_live_client(monkeypatch):
    """v3's CallbackHandler has no ``get_trace_id()`` — the id lives in the OTel context.

    The old code called ``handler.get_trace_id()``, which always raised ``AttributeError``.
    """
    _enable(monkeypatch, _FakeClient(trace_id="tr-live"))
    assert observability.current_trace_id() == "tr-live"


def test_v3_callback_handler_really_has_no_get_trace_id():
    """Pin the upstream fact that motivated the fix, if the real SDK is installed.

    If a future langfuse release adds ``get_trace_id`` back, this fails and we can simplify.
    """
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        pytest.skip("langfuse langchain integration not importable here")
    assert not hasattr(CallbackHandler, "get_trace_id")


def test_flush_is_called_through_and_safe_when_disabled(monkeypatch):
    client = _FakeClient()
    _enable(monkeypatch, client)
    observability.flush()
    assert client.flushed == 1

    # Disabled → no client → must not raise.
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: None)
    observability.flush()