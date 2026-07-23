"""Seam 2 — OpenAI-compatible FastAPI endpoint (``/v1``).

Exposes the LangGraph agent to chat clients (Open WebUI, Copilot, anything that speaks the
OpenAI Chat Completions API):

* ``POST /v1/chat/completions`` — drives the agent. Supports ``stream: true`` (SSE) and
  non-streaming. Tool/intermediate noise is hidden; we surface the final assistant text as
  OpenAI delta chunks.
* ``GET /v1/models`` — lists the agent's generation model so clients can show it in a picker.

Messages arrive as a full history each turn (chat clients are stateless); we pass them straight
to the agent and derive a ``thread_id`` so the checkpointer can carry pending writes across turns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..config import get_settings
from ..agent import graph as agent_graph

log = logging.getLogger(__name__)


def _configure_cors(application: FastAPI) -> None:
    """Attach CORS middleware with origins from ``API_CORS_ORIGINS`` (via the Settings model).

    Read through :class:`Settings` (pydantic-settings, ``.env``-aware) rather than
    ``os.environ`` directly — ``config.py``'s contract is that nothing reads env vars directly,
    and a value set only in ``.env`` would otherwise be silently ignored (pydantic-settings
    populates the model fields but does NOT export them back to ``os.environ``). We construct a
    transient ``Settings()`` instead of calling the cached :func:`get_settings`: this runs at
    module import time, before the test isolation fixture has redirected paths to tmp dirs, and
    ``get_settings()`` would eagerly ``ensure_dirs()`` (mkdir the production ``/data``).
    ``Settings()`` alone reads env + ``.env`` without that side effect. The endpoint is
    unauthenticated and write-capable, so the default is **localhost loopback only** (any port);
    a comma-separated origin list opts into more. ``*`` is accepted but discouraged.
    """
    from ..config import Settings

    raw = Settings().api_cors_origins.strip()
    if raw == "*":
        kwargs: Dict[str, Any] = {"allow_origins": ["*"]}
    elif raw:
        kwargs = {"allow_origins": [o.strip() for o in raw.split(",") if o.strip()]}
    else:
        # Default: any http(s) localhost / 127.0.0.1 origin, any port. Blocks external sites
        # from driving the write-capable endpoint cross-origin.
        kwargs = {"allow_origin_regex": r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"}
    application.add_middleware(
        CORSMiddleware,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Conversation-Id"],
        **kwargs,
    )


app = FastAPI(title="Obsidian-Librarian", version="0.1.0")
_configure_cors(app)


# --------------------------------------------------------------------------- models


class ChatMessage(BaseModel):
    role: str
    content: str | None = ""
    name: Optional[str] = None
    # OpenAI assistant messages may carry tool_calls; tool messages must reference the
    # call they answer. Without these fields we'd silently drop tool-call continuity on
    # a resent history (pydantic v2 ignores extras), corrupting multi-turn tool flows.
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # OpenAI clients send these; we accept & ignore what doesn't map to the agent.
    top_p: Optional[float] = None
    n: Optional[int] = None
    user: Optional[str] = None


def _to_lc_messages(msgs: List[ChatMessage]) -> list:
    """OpenAI messages → LangChain messages, preserving tool-call continuity."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    out: list = []
    for m in msgs:
        c = m.content or ""
        if m.role == "system":
            out.append(SystemMessage(content=c))
        elif m.role == "assistant":
            tool_calls = []
            for tc in m.tool_calls or []:
                # OpenAI shape: {"id": "...", "type": "function",
                #                "function": {"name": "...", "arguments": "..."}}
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except Exception:
                        args = {}
                tool_calls.append(
                    {
                        "name": fn.get("name", ""),
                        "args": args or {},
                        "id": tc.get("id", ""),
                        "type": "tool_call",
                    }
                )
            # ``tool_calls`` is always a list here (possibly empty). Passing ``None`` would
            # crash AIMessage (pydantic expects a list), so never coerce empties to None.
            out.append(AIMessage(content=c, tool_calls=tool_calls))
        elif m.role == "tool":
            # The tool_call_id ties a tool result back to the assistant call it answers.
            out.append(ToolMessage(content=c, tool_call_id=m.tool_call_id or m.name or ""))
        else:  # user (and unknown → user)
            out.append(HumanMessage(content=c))
    return out


def _thread_id(req: ChatCompletionRequest, conversation_id: Optional[str] = None) -> str:
    """Stable per-conversation id so the checkpointer carries pending writes across turns.

    Precedence:
    1. Explicit ``conversation_id`` (from the ``X-Conversation-Id`` header) — the caller knows
       which conversation this belongs to, so trust it. Two different conversations that happen
       to share an opening line must NOT collide, and an explicit id is the cleanest way to
       guarantee that.
    2. Otherwise hash the **full** first user message + the optional ``user`` field. The whole
       message is hashed (not just the first 200 chars) so a long note that shares its first 200
       characters with a different note gets a distinct thread — a truncated hash would route
       both to the same pending write and leak one into the other.
    3. Fall back to the configured default only when there is no user message at all.

    Same conversation → same thread → pending write survives across turns.
    """
    s = get_settings()
    if conversation_id:
        return hashlib.sha1(f"conv::{conversation_id}".encode("utf-8")).hexdigest()[:16]
    opener = next((m.content for m in req.messages if m.role == "user"), "") or ""
    if not opener:
        return s.api_default_thread_id
    key = f"{req.user or ''}::{opener}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _model_id() -> str:
    return get_settings().generation_model


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    s = get_settings()
    return {
        "object": "list",
        "data": [
            {
                "id": s.generation_model,
                "object": "model",
                "created": 0,
                "owned_by": "obsidian-librarian",
            }
        ],
    }


def _probe_qdrant() -> Optional[str]:
    """Return None if Qdrant is reachable and ready, else a short error string."""
    import httpx

    s = get_settings()
    try:
        r = httpx.get(f"{s.qdrant_url.rstrip('/')}/readyz", timeout=_PROBE_TIMEOUT)
        return None if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as e:
        return type(e).__name__


def _probe_redis() -> Optional[str]:
    """Return None if Redis answers PING, else a short error string."""
    from redis import Redis

    s = get_settings()
    client = None
    try:
        client = Redis.from_url(
            s.redis_url, socket_connect_timeout=_PROBE_TIMEOUT, socket_timeout=_PROBE_TIMEOUT
        )
        return None if client.ping() else "PING returned falsy"
    except Exception as e:
        return type(e).__name__
    finally:
        try:
            if client is not None:
                client.close()
        except Exception:  # pragma: no cover - close is best-effort
            pass


# Kept short so a hung dependency can't stall the health endpoint (which orchestrators poll
# on a timer). A dependency that can't answer in this window is down for our purposes.
_PROBE_TIMEOUT = 2.0


@app.get("/health")
async def health() -> Any:
    """Liveness + **dependency readiness**.

    This used to report ``{"status": "ok"}`` purely from settings, never touching the stores.
    That made it actively misleading: on the first container run Qdrant crash-looped in
    supervisord ``BACKOFF`` for the entire session while ``/health`` cheerfully returned ``ok``
    and the API served requests against a vector store that wasn't there. The pipeline's two
    hard dependencies are now probed for real, and a failure is reported as ``degraded`` with
    HTTP **503** so a Docker healthcheck / orchestrator actually notices.

    Obsidian is deliberately NOT a hard dependency — it is only up while the user has Obsidian
    open, and the agent is designed to run read-only without it (see ``graph._load_tools``).
    """
    s = get_settings()
    # Probes are blocking socket calls; keep them off the event loop.
    qdrant_err, redis_err = await asyncio.gather(
        asyncio.to_thread(_probe_qdrant),
        asyncio.to_thread(_probe_redis),
    )
    checks = {
        "qdrant": {"ok": qdrant_err is None, **({"error": qdrant_err} if qdrant_err else {})},
        "redis": {"ok": redis_err is None, **({"error": redis_err} if redis_err else {})},
    }
    healthy = qdrant_err is None and redis_err is None
    body: Dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "model": s.generation_model,
        "obsidian_writes": agent_graph.obsidian_writes_available(),
        "write_confirm": s.write_confirm,
        "checks": checks,
    }
    return body if healthy else JSONResponse(status_code=503, content=body)


# --------------------------------------------------------------------------- chat completions


def _sse_chunk(model: str, content: str = "", finish: Optional[str] = None, cid: str = "") -> str:
    delta: Dict[str, Any] = {}
    if content:
        delta["content"] = content
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_agent(req: ChatCompletionRequest, model: str, cid: str, thread: str):
    """Drive the agent and emit OpenAI-style SSE chunks of the final assistant text.

    A ReAct agent may invoke the LLM several times per turn (tool-calling rounds). Only the
    *last* LLM run's prose is the answer; intermediate runs carry tool-call payloads (and any
    reasoning prose) that must not leak to the chat client. We buffer the current LLM run's
    prose and discard it whenever a tool is invoked (that run was intermediate); whatever
    survives at the end is the final answer. This trades token-by-token streaming for
    leak-free output, which is what the spec calls for ("surface the final assistant text").

    Errors raised mid-stream are contained here as an SSE error chunk rather than escaping
    into the response (FastAPI can't turn a StreamingResponse into a 500 once headers sent).
    """
    from langchain_core.messages import AIMessageChunk

    candidate: list[str] = []
    current_run: Optional[str] = None
    try:
        # Convert inside the try: ``_stream_agent`` is an async generator, so this line only
        # runs on Starlette's first ``__anext__`` — AFTER ``StreamingResponse`` has sent the
        # 200 + ``text/event-stream`` headers. If ``_to_lc_messages`` raised here outside the
        # try, the exception would escape the generator, Starlette would log + close the
        # connection, and the client would get a 200 with an empty/truncated SSE stream and no
        # ``data: error`` / ``data: [DONE]``. Contain it as an SSE error chunk instead.
        lc_msgs = _to_lc_messages(req.messages)
        async for ev in agent_graph.astream(lc_msgs, thread_id=thread):
            kind = ev.get("event")
            if kind == "on_chat_model_start":
                r = ev.get("run_id")
                if r != current_run:
                    current_run = r
                    candidate = []
            elif kind in ("on_tool_start", "on_tool_end"):
                # The preceding LLM run was a tool-calling turn — its prose is not the answer.
                candidate = []
                current_run = None
            elif kind == "on_chat_model_stream":
                chunk = ev.get("data", {}).get("chunk")
                if not isinstance(chunk, AIMessageChunk) or not chunk.content:
                    continue
                # Skip chunks carrying tool-call payloads, not answer prose.
                if getattr(chunk, "tool_call_chunks", None):
                    continue
                if isinstance(chunk.content, str):
                    candidate.append(chunk.content)
    except Exception:  # contain mid-stream failures — never leak internal text cross-origin
        log.exception("agent stream failed")
        yield _sse_chunk(model, content="[stream error]", cid=cid)
        yield _sse_chunk(model, finish="error", cid=cid)
        yield "data: [DONE]\n\n"
        return

    answer = "".join(candidate)
    if answer:
        yield _sse_chunk(model, content=answer, cid=cid)
    yield _sse_chunk(model, finish="stop", cid=cid)
    yield "data: [DONE]\n\n"


async def _run_agent_nonstream(req: ChatCompletionRequest, model: str, cid: str, thread: str) -> Dict[str, Any]:
    lc_msgs = _to_lc_messages(req.messages)
    result = await agent_graph.ainvoke(lc_msgs, thread_id=thread)
    ai_msgs = [m for m in result["messages"] if m.type == "ai"]
    answer = ai_msgs[-1].content if ai_msgs else ""
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-Id"),
):
    s = get_settings()
    # Accept any model id the client sends; we always run the configured agent.
    model = req.model or s.generation_model
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    thread = _thread_id(req, conversation_id=x_conversation_id)

    try:
        if req.stream:
            return StreamingResponse(
                _stream_agent(req, model, cid, thread),
                media_type="text/event-stream",
            )
        return await _run_agent_nonstream(req, model, cid, thread)
    except Exception:  # surface a clean 500 — never leak internal text cross-origin
        log.exception("agent invocation failed")
        raise HTTPException(status_code=500, detail="agent error")


def main() -> None:  # pragma: no cover - CLI entry
    import uvicorn

    s = get_settings()
    uvicorn.run(app, host=s.api_host, port=s.api_port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()