"""Obsidian-Librarian — a personal AI librarian for your Obsidian vault.

Two halves (see `docs/SPEC.md`):

* :mod:`obsidian_librarian.rag`   — read-only RAG pipeline (parse → split → embed →
  retrieve → rerank → generate). Never writes; never uses Langfuse.
* :mod:`obsidian_librarian.agent` — LangGraph tool-calling agent that reasons over
  the RAG tools (read) and the Obsidian plugin tools (write), and controls the vault.

The user reaches the agent through :mod:`obsidian_librarian.api` (OpenAI-compatible
FastAPI endpoint, Seam 2). The agent reaches everything through MCP (Seam 1): our
read-only RAG-MCP (:mod:`obsidian_librarian.mcp`) and the host Obsidian plugin MCP.
"""

__version__ = "0.1.0"