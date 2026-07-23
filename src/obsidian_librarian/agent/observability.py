"""Observability — Langfuse, **agent only**. The RAG pipeline never uses this.

If ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set, we expose a Langfuse
``CallbackHandler`` that the agent graph invokes with, so every run is traced and scored.
If the keys are absent (e.g. local dev), :func:`get_langfuse_handler` returns ``None`` and
the agent runs unobserved — no hard dependency on Langfuse to start.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_settings

# Whether we've successfully constructed the Langfuse v3 client singleton.
_LANGFUSE_INIT: bool = False


def langfuse_enabled() -> bool:
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def _init_langfuse_client() -> bool:
    """Construct the Langfuse v3 client singleton once. Idempotent; returns False on failure."""
    global _LANGFUSE_INIT
    if _LANGFUSE_INIT:
        return True
    s = get_settings()
    try:
        from langfuse import Langfuse

        # In v3 the credentials live on the client, not the handler. Constructing
        # ``Langfuse(...)`` registers the singleton that ``CallbackHandler()`` then uses.
        Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        _LANGFUSE_INIT = True
        return True
    except Exception:  # pragma: no cover - optional dependency / config drift
        return False


def get_langfuse_handler():
    """Return a Langfuse CallbackHandler, or None if not configured/unavailable.

    Tries the v3 SDK first (``langfuse.langchain.CallbackHandler`` + a separately
    constructed client singleton) and falls back to the v2 SDK
    (``langfuse.callback.CallbackHandler`` with creds on the handler) for older installs.
    Never raises — tracing is best-effort and must not break the agent.
    """
    if not langfuse_enabled():
        return None
    s = get_settings()
    # v3
    try:
        if _init_langfuse_client():
            from langfuse.langchain import CallbackHandler as V3Handler

            return V3Handler()
    except Exception:  # pragma: no cover
        pass
    # v2 fallback
    try:
        from langfuse.callback import CallbackHandler as V2Handler

        return V2Handler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    except Exception:  # pragma: no cover - optional dependency
        return None