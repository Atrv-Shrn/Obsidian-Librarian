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
from .observability import get_langfuse_handler, langfuse_enabled
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
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    state = await agent.aget_state(cfg)
    existing = list(state.values.get("messages", []))
    if not existing:
        return messages

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
    for m in messages:
        k = key(m)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1  # already checkpointed → drop this resent copy
        else:
            out.append(m)
    # Nothing dropped → return the original list object (identity preserved for callers/tests).
    return messages if len(out) == len(messages) else out


async def ainvoke(messages: list, thread_id: Optional[str] = None) -> dict:
    """Run one agent turn. ``messages`` is a full LangChain message list (history resent).

    The returned state dict carries an extra ``_langfuse_trace_id`` key (when Langfuse is
    configured) so callers can attach scores to the right trace instead of orphaning them.
    """
    agent = await build_agent()
    tid = thread_id or get_settings().api_default_thread_id
    new_msgs = await _reconcile_new_messages(agent, messages, tid)
    config, handler = _invoke_config(thread_id)
    result = await agent.ainvoke({"messages": new_msgs}, config=config)
    if handler is not None:
        try:
            result["_langfuse_trace_id"] = handler.get_trace_id()
        except Exception:  # pragma: no cover - tracing is best-effort
            pass
    return result


async def astream(messages: list, thread_id: Optional[str] = None):
    """Stream agent events for one turn (used by the SSE FastAPI layer)."""
    agent = await build_agent()
    tid = thread_id or get_settings().api_default_thread_id
    new_msgs = await _reconcile_new_messages(agent, messages, tid)
    async for event in agent.astream_events(
        {"messages": new_msgs}, config=_invoke_config(thread_id)[0], version="v2"
    ):
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