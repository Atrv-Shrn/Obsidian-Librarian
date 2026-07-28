"""The agent's voice — ``deepseek-v4-pro:cloud`` via Ollama Cloud.

We call Ollama Cloud with :class:`langchain_ollama.ChatOllama` — the SPEC's "organs" table
locks the model call as ``ChatOllama("deepseek-v4-pro:cloud")`` over ``langchain-ollama``.
Ollama Cloud is reached by pointing ``base_url`` at the cloud host and passing the
``OLLAMA_API_KEY`` as a bearer token in the client headers (``client_kwargs={"headers": ...}``);
that maps to the underlying ``ollama.Client(host=..., headers=...)`` call. Token streaming to
the chat client is via LangChain's ``.astream()`` / LangGraph's ``.astream_events()`` — no
``streaming=`` constructor flag is needed for ``ChatOllama``.

The **judge** model (``glm-5.2:cloud``) is a separate concern used only by Ragas (Part A
evals); it never lives here. Embeddings are local Ollama, also not here.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from ..config import get_settings


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOllama:
    s = get_settings()
    api_key = s.ollama_api_key or "EMPTY"
    return ChatOllama(
        model=s.generation_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        # Ollama Cloud auth: a bearer token in the headers of the underlying ollama client.
        # client_kwargs are forwarded to httpx (sync + async), so the timeout rides along too.
        client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": s.llm_request_timeout,
        },
    )