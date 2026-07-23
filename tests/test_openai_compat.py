"""Tests for the pure helpers in the OpenAI-compatible API layer.

We exercise message conversion, thread-id derivation, and SSE chunk formatting — none of which
need the agent or FastAPI server to be running. ``langchain_core.messages`` is a real installed
dep, so the converted messages are real LangChain objects we can assert on.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from obsidian_librarian.api import openai_compat as oc
from obsidian_librarian.api.openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    _sse_chunk,
    _thread_id,
    _to_lc_messages,
)


# --------------------------------------------------------------------------- ChatMessage


def test_chat_message_accepts_tool_calls():
    m = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    )
    assert m.role == "assistant"
    assert m.tool_calls[0]["function"]["name"] == "f"


# --------------------------------------------------------------------------- _to_lc_messages


def test_to_lc_messages_system_user():
    out = _to_lc_messages(
        [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]
    )
    assert isinstance(out[0], SystemMessage) and out[0].content == "sys"
    assert isinstance(out[1], HumanMessage) and out[1].content == "hi"


def test_to_lc_messages_assistant_with_tool_calls_parses_args():
    out = _to_lc_messages(
        [
            ChatMessage(
                role="assistant",
                content="thinking...",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            )
        ]
    )
    ai = out[0]
    assert isinstance(ai, AIMessage)
    assert ai.content == "thinking..."
    assert ai.tool_calls == [
        {"name": "search", "args": {"q": "x"}, "id": "call_1", "type": "tool_call"}
    ]


def test_to_lc_messages_assistant_without_tool_calls_is_empty_list():
    out = _to_lc_messages([ChatMessage(role="assistant", content="answer")])
    ai = out[0]
    assert isinstance(ai, AIMessage)
    # AIMessage.tool_calls is a list (never None); no calls → empty list.
    assert ai.tool_calls == []


def test_to_lc_messages_tool_uses_tool_call_id():
    out = _to_lc_messages([ChatMessage(role="tool", content="result", tool_call_id="call_1")])
    assert isinstance(out[0], ToolMessage)
    assert out[0].tool_call_id == "call_1"
    assert out[0].content == "result"


def test_to_lc_messages_tool_falls_back_to_name():
    # No tool_call_id → fall back to ``name`` so the tool result still binds to a call.
    out = _to_lc_messages([ChatMessage(role="tool", content="r", name="call_1")])
    assert isinstance(out[0], ToolMessage)
    assert out[0].tool_call_id == "call_1"


def test_to_lc_messages_unknown_role_becomes_human():
    out = _to_lc_messages([ChatMessage(role="weird", content="x")])
    assert isinstance(out[0], HumanMessage)


# --------------------------------------------------------------------------- _thread_id


def _req(messages, user=None):
    return ChatCompletionRequest(messages=messages, user=user)


def test_thread_id_stable_for_same_opener_and_user():
    msgs = [ChatMessage(role="user", content="Hello there"), ChatMessage(role="assistant", content="Hi")]
    a = _thread_id(_req(msgs, user="alice"))
    b = _thread_id(_req(msgs, user="alice"))
    assert a == b
    # The full first user message is hashed (no truncation); short opener → key is exactly that.
    assert a == hashlib.sha1("alice::Hello there".encode("utf-8")).hexdigest()[:16]


def test_thread_id_differs_across_users_or_openers():
    msgs_a = [ChatMessage(role="user", content="Hello there")]
    msgs_b = [ChatMessage(role="user", content="Hello there")]
    assert _thread_id(_req(msgs_a, user="alice")) != _thread_id(_req(msgs_b, user="bob"))
    assert _thread_id(_req([ChatMessage(role="user", content="A")], user="u")) != _thread_id(
        _req([ChatMessage(role="user", content="B")], user="u")
    )


def test_thread_id_no_user_message_falls_back_to_default():
    msgs = [ChatMessage(role="system", content="sys"), ChatMessage(role="assistant", content="hi")]
    assert _thread_id(_req(msgs)) == "obsidian-librarian-default"


def test_thread_id_conversation_id_takes_precedence():
    # An explicit X-Conversation-Id wins over the opener-derived hash, so two conversations
    # that share an opening line still get distinct threads when the client supplies ids.
    msgs = [ChatMessage(role="user", content="Hello there")]
    a = _thread_id(_req(msgs, user="alice"), conversation_id="conv-42")
    b = _thread_id(_req([ChatMessage(role="user", content="totally different")], user="bob"),
                   conversation_id="conv-42")
    assert a == b == hashlib.sha1(b"conv::conv-42").hexdigest()[:16]
    # And differs from the opener-derived thread for the same request.
    assert a != _thread_id(_req(msgs, user="alice"))


def test_thread_id_full_message_hash_distinguishes_long_shared_prefixes():
    # Two long openers that share their first 200 characters but diverge after must NOT collide.
    # (A truncating hash would route both to the same thread and leak pending writes between them.)
    shared = "x" * 250
    long_a = shared + "AAA"
    long_b = shared + "BBB"
    a = _thread_id(_req([ChatMessage(role="user", content=long_a)], user="u"))
    b = _thread_id(_req([ChatMessage(role="user", content=long_b)], user="u"))
    assert a != b
    # Sanity: the full message (not the truncated prefix) drives the hash.
    assert a == hashlib.sha1(f"u::{long_a}".encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- _sse_chunk


def test_sse_chunk_with_content_has_delta():
    line = _sse_chunk("m", content="hi", finish="stop", cid="c1")
    assert line.startswith("data: ") and line.endswith("\n\n")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["id"] == "c1"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["model"] == "m"
    assert payload["choices"][0]["delta"] == {"content": "hi"}
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_sse_chunk_without_content_has_empty_delta():
    line = _sse_chunk("m", finish="stop", cid="c1")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["choices"][0]["delta"] == {}


def test_sse_chunk_finish_error():
    line = _sse_chunk("m", finish="error", cid="c1")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["choices"][0]["finish_reason"] == "error"


# --------------------------------------------------------------------------- _stream_agent containment


def test_stream_agent_contains_midstream_error(monkeypatch):
    """A failure mid-stream must be contained as an SSE error chunk + ``[DONE]`` rather than
    escaping into the response (FastAPI can't turn a StreamingResponse into a 500 once headers
    are sent). We drive ``astream`` to raise on the first event and assert the error envelope."""
    import asyncio

    async def _raising_astream(messages, thread_id=None):
        raise RuntimeError("boom")
        yield  # pragma: no cover  makes this an async generator

    monkeypatch.setattr(oc.agent_graph, "astream", _raising_astream)

    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")], stream=True)

    out: list = []

    async def collect():
        async for chunk in oc._stream_agent(req, "m", "cid1", "tid"):
            out.append(chunk)

    asyncio.run(collect())

    # Exactly: [stream error] content chunk → finish="error" chunk → [DONE].
    assert len(out) == 3
    assert out[2] == "data: [DONE]\n\n"
    p0 = json.loads(out[0][len("data: "):].strip())
    assert p0["choices"][0]["delta"]["content"] == "[stream error]"
    p1 = json.loads(out[1][len("data: "):].strip())
    assert p1["choices"][0]["finish_reason"] == "error"


# --------------------------------------------------------------------------- CORS default


def test_configure_cors_default_is_localhost_only(monkeypatch):
    """With no ``API_CORS_ORIGINS`` env, the write-capable endpoint must default to localhost
    loopback only (any port) — never a wide-open origin that an external site could drive
    cross-origin. Asserts the ``allow_origin_regex`` lands on the middleware."""
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware

    monkeypatch.delenv("API_CORS_ORIGINS", raising=False)
    app = FastAPI()
    oc._configure_cors(app)

    cors = None
    for m in app.user_middleware:
        if getattr(m, "cls", None) is CORSMiddleware:
            cors = m
            break
    assert cors is not None
    opts = getattr(cors, "options", None)
    if opts is None:
        opts = getattr(cors, "kwargs", {})
    assert opts.get("allow_origin_regex") == r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    # The wide-open default must NOT be set when the regex is in effect.
    assert "allow_origins" not in opts


def test_configure_cors_star_opts_into_all_origins(monkeypatch):
    from fastapi import FastAPI
    from starlette.middleware.cors import CORSMiddleware

    monkeypatch.setenv("API_CORS_ORIGINS", "*")
    app = FastAPI()
    oc._configure_cors(app)

    cors = next(m for m in app.user_middleware if getattr(m, "cls", None) is CORSMiddleware)
    opts = getattr(cors, "options", None) or getattr(cors, "kwargs", {})
    assert opts.get("allow_origins") == ["*"]

# --------------------------------------------------------------------------- /health probes
#
# Regression guard for the first-real-run failure: /health reported {"status": "ok"} purely
# from settings while Qdrant sat in supervisord BACKOFF for the whole session. The endpoint
# must now probe the two hard dependencies and degrade (503) when either is down.


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_health_ok_when_both_dependencies_up(monkeypatch):
    monkeypatch.setattr(oc, "_probe_qdrant", lambda: None)
    monkeypatch.setattr(oc, "_probe_redis", lambda: None)
    body = _run(oc.health())
    assert body["status"] == "ok"
    assert body["checks"]["qdrant"]["ok"] is True
    assert body["checks"]["redis"]["ok"] is True
    # A healthy response is the plain dict, not a 503 JSONResponse.
    assert isinstance(body, dict)


def test_health_degrades_with_503_when_qdrant_down(monkeypatch):
    monkeypatch.setattr(oc, "_probe_qdrant", lambda: "ConnectError")
    monkeypatch.setattr(oc, "_probe_redis", lambda: None)
    resp = _run(oc.health())
    # This is the case that silently passed before: qdrant dead, everything else fine.
    assert resp.status_code == 503
    payload = json.loads(bytes(resp.body).decode())
    assert payload["status"] == "degraded"
    assert payload["checks"]["qdrant"] == {"ok": False, "error": "ConnectError"}
    assert payload["checks"]["redis"]["ok"] is True


def test_health_degrades_when_redis_down(monkeypatch):
    monkeypatch.setattr(oc, "_probe_qdrant", lambda: None)
    monkeypatch.setattr(oc, "_probe_redis", lambda: "TimeoutError")
    resp = _run(oc.health())
    assert resp.status_code == 503
    payload = json.loads(bytes(resp.body).decode())
    assert payload["checks"]["redis"]["error"] == "TimeoutError"


def test_health_obsidian_absence_does_not_degrade(monkeypatch):
    # Obsidian is a soft dependency — only up while the user has Obsidian open. Its absence
    # must not flip the container to unhealthy.
    monkeypatch.setattr(oc, "_probe_qdrant", lambda: None)
    monkeypatch.setattr(oc, "_probe_redis", lambda: None)
    monkeypatch.setattr(oc.agent_graph, "obsidian_writes_available", lambda: False)
    body = _run(oc.health())
    assert body["status"] == "ok"
    assert body["obsidian_writes"] is False


def test_probe_returns_error_name_not_traceback(monkeypatch):
    # Probe failures must surface a short type name, never an exception string that could
    # leak connection details (URLs/credentials) to an unauthenticated caller.
    import httpx

    def _boom(*a, **kw):
        raise httpx.ConnectError("connection refused to redis://user:pw@host:6379")

    monkeypatch.setattr(httpx, "get", _boom)
    assert oc._probe_qdrant() == "ConnectError"
