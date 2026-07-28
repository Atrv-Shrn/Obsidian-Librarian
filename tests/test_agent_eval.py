"""Tests for the pure assertion helpers in the agent eval harness.

``_last_ai_text``, ``_check_read`` and ``_check_marker`` are dependency-free; we drive them
with fake task dicts and SimpleNamespace message stand-ins. No agent is invoked.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from obsidian_librarian.evals import agent_eval
from obsidian_librarian.evals.agent_eval import (
    _check_marker,
    _check_read,
    _is_write_tool,
    _last_ai_text,
)


def _msg(type, content="", tool_calls=None):
    return SimpleNamespace(type=type, content=content, tool_calls=tool_calls or [])


# --------------------------------------------------------------------------- _last_ai_text


def test_last_ai_text_returns_last_ai_content():
    result = {
        "messages": [
            _msg("human", "q"),
            _msg("ai", "first"),
            _msg("human", "again"),
            _msg("ai", "second"),
        ]
    }
    assert _last_ai_text(result) == "second"


def test_last_ai_text_empty_when_no_ai():
    assert _last_ai_text({"messages": [_msg("human", "q")]}) == ""


# --------------------------------------------------------------------------- _check_read


def test_check_read_pass_with_citation_no_marker():
    r = _check_read(
        {"must_not_marker": True, "expects_citation": "Note A"},
        reply="See Note A for details.",
    )
    assert r["pass"] is True
    assert r["notes"] == []


def test_check_read_fails_on_unexpected_marker():
    r = _check_read({"must_not_marker": True}, reply="ok [PENDING_WRITE] target=x")
    assert r["pass"] is False
    assert any("PENDING_WRITE" in n for n in r["notes"])


def test_check_read_fails_on_missing_citation():
    r = _check_read({"expects_citation": "Note A"}, reply="nothing relevant here")
    assert r["pass"] is False
    assert any("citation" in n for n in r["notes"])


def test_check_read_negative_passes_with_signal_word():
    r = _check_read({"expects_no_answer": True}, reply="I don't have that in the vault.")
    assert r["pass"] is True


def test_check_read_negative_fails_without_signal():
    r = _check_read({"expects_no_answer": True}, reply="The answer is 42.")
    assert r["pass"] is False
    assert any("negative" in n for n in r["notes"])


def test_check_read_negative_bare_vault_word_does_not_pass():
    # The word "vault" alone appears in almost every reply and must NOT satisfy the
    # negative-signal check — the agent has to actually signal a missing answer.
    r = _check_read({"expects_no_answer": True}, reply="Searching the vault now, here is the answer.")
    assert r["pass"] is False


def test_check_read_negative_curly_apostrophe_normalizes():
    # A model often emits a curly ' (U+2019) in "don't"; NFKC folds it to a straight apostrophe
    # so the negative-signal match still fires.
    r = _check_read({"expects_no_answer": True}, reply="I don’t have that note.")
    assert r["pass"] is True


# --------------------------------------------------------------------------- _is_write_tool


def test_is_write_tool_classifies_create_patch_delete():
    assert _is_write_tool("create_note") is True
    assert _is_write_tool("patch_note") is True
    assert _is_write_tool("append_to_note") is True
    assert _is_write_tool("delete_note") is True


def test_is_write_tool_rejects_read_tools():
    assert _is_write_tool("search_notes") is False
    assert _is_write_tool("query_notes") is False
    assert _is_write_tool("get_note") is False
    assert _is_write_tool("list_notes") is False
    assert _is_write_tool("get_recent") is False
    assert _is_write_tool("reindex") is False


def test_check_read_ground_truth_missing_with_no_answer_does_not_fail():
    # expects_no_answer means a missing substring is expected; ok stays True, but it's noted.
    r = _check_read(
        {"ground_truth_contains": "Secret", "expects_no_answer": True},
        reply="I don't have that in the vault.",
    )
    assert r["pass"] is True
    assert any("substring" in n for n in r["notes"])


def test_check_read_ground_truth_missing_without_no_answer_fails():
    r = _check_read({"ground_truth_contains": "Secret"}, reply="here is an answer")
    assert r["pass"] is False


# --------------------------------------------------------------------------- _check_marker


def test_check_marker_pass_full():
    reply = "[PENDING_WRITE] target=notes/x.md op=create body stuff"
    r = _check_marker({"marker_op": "create", "marker_target_contains": "notes/x.md"}, reply)
    assert r["pass"] is True
    assert r["notes"] == []


def test_check_marker_missing_marker_fails():
    r = _check_marker({"marker_op": "create"}, reply="I will create a note now")
    assert r["pass"] is False
    assert any("missing" in n.lower() for n in r["notes"])


def test_check_marker_wrong_op_fails():
    reply = "[PENDING_WRITE] target=x.md op=edit something"
    r = _check_marker({"marker_op": "create"}, reply)
    assert r["pass"] is False
    assert any("op mismatch" in n for n in r["notes"])


def test_check_marker_wrong_target_fails():
    reply = "[PENDING_WRITE] target=other.md op=create"
    r = _check_marker({"marker_op": "create", "marker_target_contains": "notes/x.md"}, reply)
    assert r["pass"] is False
    assert any("target" in n for n in r["notes"])


def test_check_marker_present_but_no_op_requirement_passes():
    reply = "[PENDING_WRITE] target=x.md something"
    r = _check_marker({}, reply)
    assert r["pass"] is True


# --------------------------------------------------------------------------- _eval_task turn-2 slice


def test_eval_write_task_counts_only_turn2_write_calls(monkeypatch):
    """The turn-2 slice must count ONLY tool calls from turn-2's own AI messages. ``r2["messages"]``
    is the full thread state (turn-1 + turn-2), so naively scanning it would re-count turn-1's read
    tool calls and taint ``executed_write``. Here turn-1 calls ``search_notes`` (a read) and turn-2
    calls ``create_note`` (a write); the slice must drop the read and keep only the write.
    """
    import asyncio

    def _msg(t, content="", tool_calls=None):
        return SimpleNamespace(type=t, content=content, tool_calls=tool_calls or [])

    r1 = {"messages": [
        _msg("human", "create a note about X"),
        _msg("ai", "[PENDING_WRITE] target=notes/x.md op=create",
             tool_calls=[{"name": "search_notes"}]),
    ]}
    r2 = {"messages": r1["messages"] + [
        _msg("human", "yes"),
        _msg("ai", "done", tool_calls=[{"name": "create_note"}]),
    ]}

    calls = {"n": 0}

    async def fake_ainvoke(messages, thread_id=None):
        calls["n"] += 1
        return r1 if calls["n"] == 1 else r2

    monkeypatch.setattr(agent_eval.agent_graph, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(agent_eval.agent_graph, "obsidian_writes_available", lambda: True)

    task = {
        "id": "w1", "kind": "write", "task": "create a note about X",
        "marker_op": "create", "marker_target_contains": "notes/x.md",
        "confirm_reply": "yes",
    }
    out = asyncio.run(agent_eval._eval_task(task))

    # The turn-1 read tool call is NOT re-counted in turn-2's slice.
    assert out["write_calls_turn2"] == ["create_note"]
    assert "search_notes" not in out["tool_calls_turn2"]
    assert out["executed_write"] is True
    assert out["pass"] is True