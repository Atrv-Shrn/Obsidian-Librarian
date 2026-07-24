"""Tests for the dependency-free config + history-reconciliation helpers in the agent graph.

We never build the real agent (that needs MCP servers + langgraph.prebuilt). Instead we drive
``_reconcile_new_messages`` with a tiny fake agent exposing ``aget_state``, and ``_invoke_config``
with monkeypatched observability hooks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from obsidian_librarian.agent import graph


class _Msg:
    """Minimal stand-in for a LangChain message: identity = (type, content)."""

    def __init__(self, type, content="", tool_calls=None, tool_call_id=None):
        self.type = type
        self.content = content
        if tool_calls is not None:
            self.tool_calls = tool_calls
        if tool_call_id is not None:
            self.tool_call_id = tool_call_id


class _FakeAgent:
    """Exposes ``aget_state`` + ``aupdate_state``; returns ``SimpleNamespace(values={...})``."""

    def __init__(self, messages, raise_on_state=False):
        self._messages = messages
        self._raise = raise_on_state
        self.updates = []  # records aupdate_state calls (for the repair test)

    async def aget_state(self, cfg):
        if self._raise:
            raise RuntimeError("no checkpoint backend")
        return SimpleNamespace(values={"messages": list(self._messages)})

    async def aupdate_state(self, cfg, values):
        # Mimic the add_messages reducer: append.
        self.updates.append(values)
        self._messages = list(self._messages) + list(values.get("messages", []))


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# --------------------------------------------------------------------------- _reconcile_new_messages


def test_reconcile_empty_state_returns_full():
    agent = _FakeAgent(messages=[])
    msgs = [_Msg("human", "hi"), _Msg("ai", "hello")]
    assert _run(graph._reconcile_new_messages(agent, msgs, "t")) is msgs


def test_reconcile_aget_state_raises_propagates():
    # A fresh thread returns empty state (no raise); a raise means the backend is broken.
    # Propagate it rather than silently feeding duplicates into a partially-checkpointed thread.
    agent = _FakeAgent(messages=[_Msg("ai", "x")], raise_on_state=True)
    msgs = [_Msg("human", "hi")]
    with pytest.raises(RuntimeError):
        _run(graph._reconcile_new_messages(agent, msgs, "t"))


def test_reconcile_drops_already_checkpointed_when_history_strips_tool_msgs():
    # Chat clients resend only user/assistant turns — tool + tool-call noise is stripped.
    # The checkpoint state carries the full turn-1 sequence (incl. the tool round), so a
    # positional prefix match would misalign; content-membership dedup must still drop the
    # resent human+ai pair and pass only the new "yes".
    existing = [_Msg("human", "q"), _Msg("ai", "<toolcall>"), _Msg("tool", "result"), _Msg("ai", "plan")]
    agent = _FakeAgent(messages=existing)
    msgs = [_Msg("human", "q"), _Msg("ai", "plan"), _Msg("human", "yes")]
    out = _run(graph._reconcile_new_messages(agent, msgs, "t"))
    assert [m.content for m in out] == ["yes"]


def test_reconcile_prefix_match_returns_suffix():
    existing = [_Msg("human", "hi"), _Msg("ai", "hello")]
    agent = _FakeAgent(messages=existing)
    msgs = [_Msg("human", "hi"), _Msg("ai", "hello"), _Msg("human", "again?")]
    out = _run(graph._reconcile_new_messages(agent, msgs, "t"))
    assert len(out) == 1
    assert out[0].content == "again?"


def test_reconcile_no_overlap_returns_full():
    # First message differs → n=0 → full list returned.
    agent = _FakeAgent(messages=[_Msg("human", "different")])
    msgs = [_Msg("human", "hi"), _Msg("ai", "hello")]
    assert _run(graph._reconcile_new_messages(agent, msgs, "t")) is msgs


def test_reconcile_partial_prefix_match():
    existing = [_Msg("human", "hi"), _Msg("ai", "hello")]
    agent = _FakeAgent(messages=existing)
    # Diverges at index 1.
    msgs = [_Msg("human", "hi"), _Msg("ai", "DIFFERENT"), _Msg("human", "q")]
    out = _run(graph._reconcile_new_messages(agent, msgs, "t"))
    assert len(out) == 2
    assert out[0].content == "DIFFERENT"
    assert out[1].content == "q"


# --------------------------------------------------------------------------- _invoke_config


def test_invoke_config_disabled_no_callbacks():
    # No Langfuse keys set by the autouse fixture → langfuse_enabled() is False.
    config, handler = graph._invoke_config(None)
    assert handler is None
    assert "callbacks" not in config
    assert config["recursion_limit"] == 30
    assert config["configurable"]["thread_id"] == "obsidian-librarian-default"


def test_invoke_config_thread_id_override():
    config, handler = graph._invoke_config("custom-thread")
    assert config["configurable"]["thread_id"] == "custom-thread"
    assert handler is None


def test_invoke_config_enabled_attaches_handler(monkeypatch):
    class _FakeHandler:
        def get_trace_id(self):
            return "trace-xyz"

    monkeypatch.setattr(graph, "langfuse_enabled", lambda: True)
    monkeypatch.setattr(graph, "get_langfuse_handler", lambda: _FakeHandler())

    config, handler = graph._invoke_config("t")
    assert handler is not None
    assert handler.get_trace_id() == "trace-xyz"
    assert config["callbacks"] == [handler]


def test_invoke_config_enabled_but_handler_none_omits_callbacks(monkeypatch):
    # langfuse enabled but get_langfuse_handler() returned None → no callbacks key.
    monkeypatch.setattr(graph, "langfuse_enabled", lambda: True)
    monkeypatch.setattr(graph, "get_langfuse_handler", lambda: None)
    config, handler = graph._invoke_config("t")
    assert handler is None
    assert "callbacks" not in config

# --------------------------------------------------------------------------- list-content keys
#
# Regression guard for the "[stream error]" a couple of turns into a tool-using conversation:
# LangChain stores tool-call / content-block messages with a LIST content, which is unhashable,
# so Counter(key(m) ...) raised TypeError: unhashable type: 'list'. key() must normalize.


def test_reconcile_handles_list_content_in_checkpoint():
    # An AI message whose content is a list of content blocks (as tool-using turns produce).
    existing = [
        _Msg("human", "question"),
        _Msg("ai", [{"type": "text", "text": "thinking"}, {"type": "tool_use", "name": "search"}]),
    ]
    agent = _FakeAgent(messages=existing)
    # A fresh user turn resent as a string; must not crash on the unhashable checkpoint entry.
    msgs = [_Msg("human", "question"), _Msg("human", "follow-up")]
    out = _run(graph._reconcile_new_messages(agent, msgs, "t"))
    # The already-seen "question" is dropped; the genuinely new "follow-up" survives.
    assert [m.content for m in out] == ["follow-up"]


def test_reconcile_list_content_message_dedups_against_itself():
    # Same list-content message on both sides must dedup one-for-one (stable repr key).
    blocks = [{"type": "text", "text": "x"}]
    existing = [_Msg("ai", blocks)]
    agent = _FakeAgent(messages=existing)
    msgs = [_Msg("ai", list(blocks))]  # equal-by-value list content
    out = _run(graph._reconcile_new_messages(agent, msgs, "t"))
    assert out == []


# --------------------------------------------------------------------------- poisoned checkpoint
#
# Regression guard for the poisoned-checkpoint bug: a turn that errors after the model emits
# tool_calls but before ToolMessages are written leaves the checkpoint with dangling tool_calls,
# and every later turn on that thread_id 500s with "Found AIMessages with tool_calls that do not
# have a corresponding ToolMessage". _repair_dangling_tool_calls completes them with error
# ToolMessages so the thread is usable again.


def test_repair_appends_toolmessage_for_dangling_call():
    # AIMessage with a tool_call, NO following ToolMessage -> poisoned.
    existing = [
        _Msg("human", "q"),
        _Msg("ai", "", tool_calls=[{"id": "call_x", "name": "search", "args": {}}]),
    ]
    agent = _FakeAgent(messages=existing)
    n = _run(graph._repair_dangling_tool_calls(agent, "t"))
    assert n == 1
    assert len(agent.updates) == 1
    appended = agent.updates[0]["messages"]
    assert len(appended) == 1
    tm = appended[0]
    assert tm.type == "tool" and tm.tool_call_id == "call_x"
    assert getattr(tm, "status", None) == "error"


def test_repair_noop_when_all_calls_answered():
    existing = [
        _Msg("ai", "", tool_calls=[{"id": "call_x", "name": "search", "args": {}}]),
        _Msg("tool", "result", tool_call_id="call_x"),
    ]
    agent = _FakeAgent(messages=existing)
    n = _run(graph._repair_dangling_tool_calls(agent, "t"))
    assert n == 0
    assert agent.updates == []


def test_repair_noop_on_empty_or_plain_history():
    assert _run(graph._repair_dangling_tool_calls(_FakeAgent(messages=[]), "t")) == 0
    plain = _FakeAgent(messages=[_Msg("human", "hi"), _Msg("ai", "hello")])
    assert _run(graph._repair_dangling_tool_calls(plain, "t")) == 0
    assert plain.updates == []


def test_repair_multiple_dangling_calls_one_message_each():
    existing = [
        _Msg("ai", "", tool_calls=[
            {"id": "a", "name": "search", "args": {}},
            {"id": "b", "name": "get_note", "args": {}},
        ]),
        _Msg("tool", "answered", tool_call_id="a"),  # only 'a' answered
    ]
    agent = _FakeAgent(messages=existing)
    n = _run(graph._repair_dangling_tool_calls(agent, "t"))
    assert n == 1  # only 'b' was dangling
    assert agent.updates[0]["messages"][0].tool_call_id == "b"
