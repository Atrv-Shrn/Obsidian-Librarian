"""Observability — Langfuse, **agent only**. The RAG pipeline never uses this.

If ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are set, we expose a Langfuse
``CallbackHandler`` that the agent graph invokes with, so every run is traced and scored.
If the keys are absent (e.g. local dev), :func:`get_langfuse_handler` returns ``None`` and
the agent runs unobserved — no hard dependency on Langfuse to start.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import get_settings

log = logging.getLogger(__name__)

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


def get_langfuse_client() -> Optional[Any]:
    """Return the initialized Langfuse v3 client singleton, or ``None``.

    Scores must be created through the *same* client that the ``CallbackHandler`` traces
    through, otherwise they land on a different (or no) trace. v3 exposes
    ``langfuse.get_client()`` for exactly this; we initialize first so the singleton carries
    our configured credentials rather than falling back to bare env vars.
    """
    if not langfuse_enabled() or not _init_langfuse_client():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:  # pragma: no cover - optional dependency / version drift
        return None


def current_trace_id() -> Optional[str]:
    """Trace id of the span currently in context, or ``None``.

    **Must be called while the traced call is still on the stack.** The v3 SDK is
    OpenTelemetry-based: the trace id lives in the active OTel context, not on the handler.
    The v3 ``CallbackHandler`` has no ``get_trace_id()`` method at all (it only holds
    ``.client``), so the old ``handler.get_trace_id()`` raised ``AttributeError`` on every
    call — silently, because the caller swallowed it — and every score was written with an
    empty trace id, i.e. orphaned. Reading the id from the live OTel context is the v3 way.
    """
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return client.get_current_trace_id()
    except Exception:  # pragma: no cover - tracing is best-effort
        return None


def score_trace(name: str, value: float, trace_id: Optional[str], comment: str = "") -> bool:
    """Attach a numeric score to ``trace_id``. Returns True if it was submitted.

    Uses v3's ``create_score`` (the v2 ``lf.score()`` method does not exist in v3 — calling it
    raised ``AttributeError``, which the eval swallowed into a warning, so no score ever landed).
    A falsy ``trace_id`` is refused rather than sent as ``""``: an empty trace id is what
    produced orphaned scores in the first place, and silently writing one is worse than
    reporting that we couldn't score.
    """
    if not langfuse_enabled():
        return False
    if not trace_id:
        log.warning("langfuse: no trace id for %r — score not submitted (would be orphaned)", comment or name)
        return False
    client = get_langfuse_client()
    if client is None:
        return False
    try:
        client.create_score(name=name, value=value, trace_id=trace_id, comment=comment or None)
        return True
    except Exception as e:  # never fail the caller on telemetry
        log.warning("langfuse create_score failed: %s", e)
        return False


def flush() -> None:
    """Flush buffered traces/scores to Langfuse. Safe to call when disabled.

    The v3 SDK batches over OTel and ships asynchronously, so a short-lived process (our eval
    harness, the CLI) can exit before anything is sent — scores would silently never appear.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:  # pragma: no cover - best effort
        log.warning("langfuse flush failed: %s", e)


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