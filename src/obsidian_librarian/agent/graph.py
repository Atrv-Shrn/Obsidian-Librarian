"""The LangGraph agent — ``create_react_agent`` over MCP tools + checkpointer.

Two MCP backends, combined into one toolset:
- **RAG-MCP** (always): our read-only pipeline at ``127.0.0.1:8765/mcp`` (streamable-http).
- **Obsidian plugin MCP** (best-effort): writes via the user's Local REST API plugin at
  ``host.docker.internal:27124/mcp``. Obsidian only runs while the user is at their desk, so
  this connection is wrapped in try/except — if it's down, the agent still answers questions
  and simply can't write. We never hard-fail boot on Obsidian being absent.

The propose-then-confirm HITL contract lives in the system prompt (see ``prompts.py``); the
graph itself is a standard ReAct loop. ``recursion_limit`` is the hard guard against tool
loops and is passed in the invoke config by the API/CLI layers, not baked into the graph.

Langfuse, if configured, attaches as a callback handler on each invocation (agent only).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings
from .llm import get_agent_llm
from .memory import get_checkpointer
from .observability import (
    current_trace_id,
    get_langfuse_client,
    get_langfuse_handler,
    langfuse_enabled,
)
from .prompts import build_system_prompt

log = logging.getLogger(__name__)

_AGENT: Optional[Any] = None
# Guards the check-then-set build, mirroring ``memory._SAVER_LOCK``: the agent eval runs
# golden tasks concurrently (``asyncio.gather``), so without a lock several coroutines
# could all see ``_AGENT is None`` on the first turn and each build + connect to the MCP
# servers (leaking the losers' connections). Python 3.11+ defers loop binding for
# ``asyncio.Lock``, so creating it at import is safe.
_AGENT_LOCK = asyncio.Lock()
_OBSIDIAN_AVAILABLE: bool = False


def _rag_server_spec() -> dict:
    s = get_settings()
    return {
        "url": s.rag_mcp_endpoint,
        "transport": "streamable_http",
        "timeout": 60,
    }


def _insecure_httpx_factory():
    """httpx factory that skips TLS verification — the Obsidian plugin ships a self-signed cert."""

    def _factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout or httpx.Timeout(30.0),
            auth=auth,
            verify=False,
        )

    return _factory


def _obsidian_server_spec() -> dict:
    s = get_settings()
    headers: dict[str, str] = {}
    if s.obsidian_api_key:
        headers["Authorization"] = f"Bearer {s.obsidian_api_key}"
    spec: dict[str, Any] = {
        "url": s.obsidian_mcp_url,
        "transport": "streamable_http",
        "timeout": 30,
    }
    if headers:
        spec["headers"] = headers
    if s.obsidian_tls_insecure:
        spec["httpx_client_factory"] = _insecure_httpx_factory()
    return spec


async def _load_tools() -> list:
    """Connect to RAG-MCP (required) + Obsidian MCP (best-effort) and return combined tools."""
    s = get_settings()

    rag_client = MultiServerMCPClient({"rag": _rag_server_spec()})
    tools = list(await rag_client.get_tools())
    log.info("Loaded %d RAG-MCP tools", len(tools))

    global _OBSIDIAN_AVAILABLE
    obsidian_tools: list = []
    try:
        obs_client = MultiServerMCPClient({"obsidian": _obsidian_server_spec()})
        obsidian_tools = list(await obs_client.get_tools())
        _OBSIDIAN_AVAILABLE = bool(obsidian_tools)
        if _OBSIDIAN_AVAILABLE:
            log.info("Loaded %d Obsidian write tools", len(obsidian_tools))
        else:
            log.warning("Obsidian MCP reachable but exposed no tools — writes disabled.")
    except Exception as e:  # best-effort: Obsidian is only up while the user is present
        _OBSIDIAN_AVAILABLE = False
        log.warning("Obsidian MCP unavailable (%s); agent will run read-only.", type(e).__name__)

    return tools + obsidian_tools


def obsidian_writes_available() -> bool:
    """True if the Obsidian plugin MCP responded with write tools at boot."""
    return _OBSIDIAN_AVAILABLE


async def build_agent():
    """Construct (or return cached) the compiled ReAct agent.

    Double-checked locking (see ``_AGENT_LOCK``): the first coroutine through builds the
    agent + connects to MCP; concurrent first-callers wait on the lock, then see the
    cached agent and reuse it instead of each opening their own connections.
    """
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    async with _AGENT_LOCK:
        # Re-check under the lock: the first coroutine through may have already built it.
        if _AGENT is not None:
            return _AGENT

        from langgraph.prebuilt import create_react_agent

        s = get_settings()
        llm = get_agent_llm()
        tools = await _load_tools()
        checkpointer = await get_checkpointer()
        system_prompt = build_system_prompt()

        kwargs: dict[str, Any] = {
            "model": llm,
            "tools": tools,
            "checkpointer": checkpointer,
        }
        # langgraph renamed `prompt` -> `state_modifier` -> `prompt` across versions; try both.
        try:
            _AGENT = create_react_agent(prompt=system_prompt, **kwargs)
        except TypeError:
            _AGENT = create_react_agent(state_modifier=system_prompt, **kwargs)
        log.info(
            "Agent built: model=%s tools=%d writes=%s confirm=%s",
            s.generation_model,
            len(tools),
            _OBSIDIAN_AVAILABLE,
            s.write_confirm,
        )
        return _AGENT


def _invoke_config(thread_id: Optional[str] = None) -> tuple[dict, Optional[Any]]:
    """Build the per-call config and return ``(config, langfuse_handler_or_None)``.

    The handler is returned so callers that want the Langfuse trace id (e.g. the eval
    scorer) can read ``handler.get_trace_id()`` from the *same* instance attached as a
    callback — a fresh handler built elsewhere would have a different trace.
    """
    s = get_settings()
    tid = thread_id or s.api_default_thread_id
    config: dict[str, Any] = {
        "recursion_limit": s.recursion_limit,
        "configurable": {"thread_id": tid},
    }
    handler = None
    if langfuse_enabled():
        handler = get_langfuse_handler()
        if handler is not None:
            config["callbacks"] = [handler]
    return config, handler


def get_invoke_config(thread_id: Optional[str] = None) -> dict:
    """Per-call config: recursion limit + thread id + Langfuse callback (if configured)."""
    return _invoke_config(thread_id)[0]


async def _reconcile_new_messages(agent, messages: list, thread_id: str) -> list:
    """Return only the messages in ``messages`` not already in checkpoint state.

    Chat clients (Open WebUI, Copilot, our eval harness) resend the *full* conversation
    each turn. With a checkpointer keyed by a stable ``thread_id``, feeding that full
    list straight to ``ainvoke`` would make ``add_messages`` append every resent message
    as a new one (resent messages carry no ``id`` → fresh UUIDs → no dedup) and the
    thread's state would balloon with duplicates.

    We diff the resent history against the checkpointed state by ``(type, content)`` and
    drop every resent message that already has a matching entry in the checkpoint, passing
    only what's genuinely new (preserving input order). This is a multiset subtraction, not a
    longest-common-prefix: chat clients strip tool/tool-call noise from the history they
    resend (they only carry user/assistant turns), so the resent list is rarely a positional
    prefix of the checkpoint state — a prefix match would misalign, re-feed an old answer as
    a new message, or drop the user's actual new turn. Dropping by content membership is
    robust to that stripping.

    A fresh thread (no checkpoint) returns the full list unchanged. A real checkpoint *read*
    failure is propagated, not swallowed: a fresh thread returns empty state and does NOT
    raise, so an exception here means the backend is broken — silently returning the full
    list would feed duplicates into a partially-checkpointed thread and corrupt it.

    **The final message is never dropped.** In an OpenAI Chat Completions request the last
    message is, by construction, the turn the user just typed. Subtracting it because its text
    matched something already in the checkpoint is how this function used to silently swallow
    an entire turn: re-ask a question (or open a new chat with a familiar first line, which the
    content-derived thread id routed onto the *same* thread), and every resent message matched,
    leaving ``[]`` to invoke the graph with. The agent then replayed a state that already ended
    in an AI answer, produced no new content, and the API returned an empty 200 — the client
    renders nothing and the assistant appears to ignore you. Reconciling only the *prefix*
    makes that structurally impossible: whatever else happens, the agent sees what you just
    sent. A genuine duplicate (the user really did re-ask the same thing) is the correct thing
    to append anyway.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(cfg)
    existing = list(state.values.get("messages", []))
    if not existing or not messages:
        return messages

    # Only the prefix is eligible for subtraction; the last message always passes through.
    head, tail = messages[:-1], messages[-1:]

    from collections import Counter

    def key(m: Any) -> tuple:
        # ``content`` is not always a string: LangChain stores tool-call / content-block
        # messages with a LIST content (e.g. ``[{"type": "text", ...}, {"type": "tool_use", ...}]``),
        # and a list is unhashable → ``Counter`` blew up with ``TypeError: unhashable type:
        # 'list'`` once a prior turn's tool-using AI message landed in the checkpoint. That
        # surfaced as a "[stream error]" a couple of turns into any conversation where the agent
        # called a tool. Normalize non-string content to a stable string so the key is hashable;
        # a deterministic ``repr`` is enough here (we only need equality within one reconcile).
        c = getattr(m, "content", "")
        if not isinstance(c, str):
            c = repr(c)
        return (m.type, c)

    # Count existing messages by key so duplicates dedup one-for-one rather than all-or-none.
    remaining = Counter(key(m) for m in existing)
    out: list = []
    for m in head:
        k = key(m)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1  # already checkpointed → drop this resent copy
        else:
            out.append(m)
    out.extend(tail)
    # Nothing dropped → return the original list object (identity preserved for callers/tests).
    return messages if len(out) == len(messages) else out


async def _new_messages_for_turn(agent, messages: list, thread_id: str) -> list:
    """Reconcile against the checkpoint, upholding "never invoke the graph with nothing".

    ``_reconcile_new_messages`` already guarantees this by always passing the last message
    through, so the fallback here should be unreachable. It stays as a structural backstop:
    invoking the agent with an empty message list is the single failure mode that produces a
    silent, empty, HTTP-200 answer — the graph replays stale state, emits no new content, and
    the user sees the assistant ignore them. Whatever else goes wrong, we would rather send the
    full history twice than send nothing.
    """
    new_msgs = await _reconcile_new_messages(agent, messages, thread_id)
    if messages and not new_msgs:
        log.warning(
            "Reconciliation emptied a non-empty turn on thread %s; sending full history "
            "instead of invoking the agent with nothing.",
            thread_id,
        )
        return messages
    return new_msgs


async def _repair_dangling_tool_calls(agent, thread_id: str) -> int:
    """Complete any checkpointed tool_calls that never received a ToolMessage. Returns count.

    If a turn errors *after* the model emits an AIMessage with tool_calls but *before* the
    tools node writes the results (e.g. an MCP ``ConnectError`` while Obsidian is flapping, or
    a recursion-limit abort), the checkpoint is left with **dangling** tool_calls. On the next
    turn LangGraph reloads that state and ``create_react_agent``'s ``_validate_chat_history``
    raises ``Found AIMessages with tool_calls that do not have a corresponding ToolMessage`` —
    which 500s **every** subsequent message on that ``thread_id``. Because the thread id is
    derived from the conversation, one transient hiccup poisons that chat permanently.

    We repair it by appending a synthetic *error* ToolMessage for each unanswered tool_call, so
    the history validates and the model simply sees "that tool failed" and proceeds. Appended
    via ``aupdate_state`` (the ``add_messages`` reducer), so they land right after the dangling
    AIMessage and before this turn's new input.
    """
    from langchain_core.messages import ToolMessage

    cfg = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(cfg)
    messages = list(state.values.get("messages", []))
    if not messages:
        return 0
    answered = {
        m.tool_call_id
        for m in messages
        if getattr(m, "type", None) == "tool" and getattr(m, "tool_call_id", None)
    }
    repairs = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id and tc_id not in answered:
                repairs.append(
                    ToolMessage(
                        content=(
                            "Error: this tool call did not complete because the previous turn "
                            "was interrupted. Disregard it and continue."
                        ),
                        tool_call_id=tc_id,
                        status="error",
                    )
                )
                answered.add(tc_id)  # never synthesize two results for the same id
    if repairs:
        log.warning(
            "Repaired %d dangling tool call(s) on thread %s (poisoned checkpoint).",
            len(repairs),
            thread_id,
        )
        await agent.aupdate_state(cfg, {"messages": repairs})
    return len(repairs)


async def ainvoke(messages: list, thread_id: Optional[str] = None) -> dict:
    """Run one agent turn. ``messages`` is a full LangChain message list (history resent).

    The returned state dict carries an extra ``_langfuse_trace_id`` key (when Langfuse is
    configured) so callers can attach scores to the right trace instead of orphaning them.
    """
    agent = await build_agent()
    tid = thread_id or get_settings().api_default_thread_id
    await _repair_dangling_tool_calls(agent, tid)
    new_msgs = await _new_messages_for_turn(agent, messages, tid)
    config, handler = _invoke_config(thread_id)
    if handler is None:
        return await agent.ainvoke({"messages": new_msgs}, config=config)

    # Wrap the run in an explicit Langfuse span so we own a trace id we can score against.
    # The v3 SDK is OpenTelemetry-based: the trace id lives in the *active context*, so it can
    # only be read while the call is on the stack — reading it after ``ainvoke`` returns (as we
    # used to, via the handler) yields nothing, because the span has already closed. The old
    # code called ``handler.get_trace_id()``, which doesn't exist in v3 at all, so the trace id
    # was always ``None`` and every score was orphaned.
    client = get_langfuse_client()
    if client is None:
        return await agent.ainvoke({"messages": new_msgs}, config=config)
    with client.start_as_current_span(name="agent-turn") as span:
        trace_id = current_trace_id()
        result = await agent.ainvoke({"messages": new_msgs}, config=config)
        try:
            span.update_trace(session_id=tid)
        except Exception:  # pragma: no cover - tracing is best-effort
            pass
    result["_langfuse_trace_id"] = trace_id
    return result


async def astream(messages: list, thread_id: Optional[str] = None):
    """Stream agent events for one turn (used by the SSE FastAPI layer).

    Wrapped in the same explicit Langfuse span as :func:`ainvoke` so a streamed chat turn
    produces one grouped trace tagged with its thread, rather than loose spans. The span must
    stay open for the whole generator — closing it before the stream drains would orphan every
    event after the first.
    """
    agent = await build_agent()
    tid = thread_id or get_settings().api_default_thread_id
    await _repair_dangling_tool_calls(agent, tid)
    new_msgs = await _new_messages_for_turn(agent, messages, tid)
    config = _invoke_config(thread_id)[0]

    client = get_langfuse_client() if langfuse_enabled() else None
    if client is None:
        async for event in agent.astream_events({"messages": new_msgs}, config=config, version="v2"):
            yield event
        return

    with client.start_as_current_span(name="agent-turn") as span:
        try:
            span.update_trace(session_id=tid)
        except Exception:  # pragma: no cover - tracing is best-effort
            pass
        async for event in agent.astream_events({"messages": new_msgs}, config=config, version="v2"):
            yield event


def reset_agent_cache() -> None:
    """Drop the cached agent so the next build reconnects to MCP servers (used on reconnect).

    Also clears the LLM and system-prompt singletons so a settings change (model swap,
    WRITE_CONFIRM toggle) is picked up on the next build rather than served stale.
    """
    global _AGENT, _OBSIDIAN_AVAILABLE
    _AGENT = None
    _OBSIDIAN_AVAILABLE = False
    # Clear the lru_cached singletons in sibling modules.
    from .llm import get_agent_llm
    from .prompts import build_system_prompt

    get_agent_llm.cache_clear()
    build_system_prompt.cache_clear()