"""Seam 2 — the user↔agent front door.

:mod:`openai_compat` exposes an OpenAI-compatible FastAPI app:
``POST /v1/chat/completions`` (SSE streaming) and ``GET /v1/models``, so Obsidian
Copilot and Open WebUI can point one base URL at ``:8000/v1`` and just talk.
"""