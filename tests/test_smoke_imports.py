"""Smoke test: every project module imports cleanly under the conftest stub regime.

This catches top-level import breakages (a stubbed heavy package used in a way our stub doesn't
satisfy, a real installed dep with API drift, a bad relative import) without needing the live
stack. It does *not* exercise behavior — the unit-test modules do that.
"""

from __future__ import annotations

import importlib

import pytest

_PROJECT_MODULES = [
    "obsidian_librarian.config",
    "obsidian_librarian.cli",
    "obsidian_librarian.rag.embeddings",
    "obsidian_librarian.rag.reader",
    "obsidian_librarian.rag.ingest",
    "obsidian_librarian.rag.retrieve",
    "obsidian_librarian.rag.query_engine",
    "obsidian_librarian.rag.sync.watermarks",
    "obsidian_librarian.rag.sync.scheduler",
    "obsidian_librarian.mcp.rag_server",
    "obsidian_librarian.agent.llm",
    "obsidian_librarian.agent.prompts",
    "obsidian_librarian.agent.memory",
    "obsidian_librarian.agent.observability",
    "obsidian_librarian.agent.graph",
    "obsidian_librarian.api.openai_compat",
    "obsidian_librarian.evals.rag_eval",
    "obsidian_librarian.evals.agent_eval",
]


@pytest.mark.parametrize("modname", _PROJECT_MODULES)
def test_module_imports_cleanly(modname):
    mod = importlib.import_module(modname)
    assert mod is not None
    # Sanity: the module's name matches what we asked for (not a re-export stub).
    assert mod.__name__ == modname