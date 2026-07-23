"""Shared pytest fixtures + heavy-dependency stubbing.

The full stack (Qdrant, Redis, LlamaIndex, langchain-openai, langchain-mcp-adapters,
ragas, apscheduler, fastmcp, httpx) is not installed in every environment that runs these
tests (e.g. CI without the Docker image, or a bare dev box). Several project modules import
those packages at top level, so importing them would raise ``ModuleNotFoundError`` before a
test can touch the *pure* helpers we actually want to unit-test.

Rather than refactor production code, we install a :data:`sys.meta_path` finder that returns a
permissive stub module for any not-installed heavy package (and a couple of optional
submodules of installed packages, like ``langgraph.checkpoint.sqlite``). The stubs satisfy
``from heavy import Name`` at import time; they are never *exercised* by the unit tests, which
call only dependency-free helpers. This lets us test the real code in isolation.

The stub object returned for any name is a plain class that is both **callable** and
**subclassable** (with a metaclass that yields more stubs for any attribute access), so
``class Foo(StubbedBase):`` and ``StubbedThing(...)`` both work at import time.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys
import types
from typing import Set

import pytest


# Heavy top-level packages whose absence should be papered over for import-only purposes.
# (fastembed / mcp / langchain_core / langgraph / langfuse / pydantic / fastapi / typer / rich
#  are real deps that are typically installed and are NOT stubbed when present.)
_HEAVY_TOPLEVEL = [
    "qdrant_client",
    "redis",
    "llama_index",
    "langchain_openai",
    "langchain_ollama",
    "langchain_mcp_adapters",
    "ragas",
    "apscheduler",
    "fastmcp",
    "httpx",
    "datasets",
]

# Submodules of *installed* packages that may be missing (optional extras).
_HEAVY_SUBMODULES = [
    "langgraph.prebuilt",
    "langgraph.checkpoint.sqlite",
    "langgraph.checkpoint.sqlite.aio",
]


# --------------------------------------------------------------------------- stub object


class _StubMeta(type):
    """Metaclass so any class-attribute access on a stub yields another stub."""

    def __getattr__(cls, name):  # noqa: D401
        return _Stub


class _Stub(metaclass=_StubMeta):
    """A callable, subclassable, attribute-permissive stand-in.

    Used both as the return value of module ``__getattr__`` (so ``from heavy import X``
    binds a subclassable/callable thing) and for chained attribute access.
    """

    def __init__(self, *args, **kwargs):
        # Stash kwargs so tests that *do* route through a stub can still read them back.
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):  # noqa: D401
        return _Stub

    def __call__(self, *args, **kwargs):  # noqa: D401
        return _Stub(*args, **kwargs)


def _stub_getattr(name: str):
    return _Stub


# --------------------------------------------------------------------------- detection


def _detect_missing() -> tuple[Set[str], Set[str]]:
    missing_top: Set[str] = set()
    for name in _HEAVY_TOPLEVEL:
        try:
            importlib.import_module(name)
        except Exception:
            missing_top.add(name)
    missing_sub: Set[str] = set()
    for name in _HEAVY_SUBMODULES:
        try:
            importlib.import_module(name)
        except Exception:
            missing_sub.add(name)
    return missing_top, missing_sub


_MISSING_TOP, _MISSING_SUB = _detect_missing()


# --------------------------------------------------------------------------- finder


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):  # noqa: D401
        return None  # use default module creation

    def exec_module(self, module) -> None:
        # Mark as a package so ``import heavy.sub`` works, and install PEP 562 __getattr__.
        module.__path__ = []  # type: ignore[attr-defined]
        module.__all__ = []  # type: ignore[attr-defined]
        module.__getattr__ = _stub_getattr  # type: ignore[attr-defined]


class _StubFinder(importlib.abc.MetaPathFinder):
    """Return a stub spec for any missing heavy fullname; None otherwise (normal import)."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in _MISSING_TOP or fullname in _MISSING_SUB:
            return importlib.util.spec_from_loader(fullname, _StubLoader(), is_package=True)
        return None


_STUB_FINDER = _StubFinder()
if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _STUB_FINDER)


# --------------------------------------------------------------------------- fixtures


def _clear_lru_singletons() -> None:
    """Clear the project's ``@lru_cache`` singletons so a settings change is picked up.

    Several modules cache a heavy object (embedder, reranker, generation LLM, system prompt,
    MCP vault doc cache, …) built from the *first* ``get_settings()`` they saw. The autouse
    fixture resets settings per test, but without clearing these the next test could reuse a
    singleton bound to the previous env. Importing the modules is safe — heavy deps are
    stubbed — and each ``cache_clear`` is best-effort.
    """
    targets = [
        ("obsidian_librarian.rag.embeddings", ["get_dense_embed_model",
                                                "get_sparse_embed_model", "get_reranker"]),
        ("obsidian_librarian.rag.query_engine", ["get_generation_llm"]),
        ("obsidian_librarian.agent.prompts", ["build_system_prompt"]),
        ("obsidian_librarian.agent.llm", ["get_agent_llm"]),
        ("obsidian_librarian.mcp.rag_server", ["_vault_docs"]),
    ]
    for modname, names in targets:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for n in names:
            fn = getattr(mod, n, None)
            clear = getattr(fn, "cache_clear", None)
            if callable(clear):
                clear()


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Point vault/data/sqlite at a per-test tmp dir and reset the settings cache.

    Production defaults (``/vault``, ``/data``) are unwritable on Windows / non-root CI, and
    ``get_settings()`` calls ``ensure_dirs()`` which mkdirs them. Every test gets fresh,
    writable paths and a clean settings singleton so env-driven tests don't bleed into each other.
    """
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv("DATA_PATH", str(data))
    monkeypatch.setenv("SQLITE_PATH", str(data / "librarian.db"))

    from obsidian_librarian.config import reset_settings_cache

    reset_settings_cache()
    _clear_lru_singletons()
    yield
    reset_settings_cache()
    _clear_lru_singletons()