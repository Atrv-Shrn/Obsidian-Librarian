"""Seam 1 (read side) — the RAG-MCP server.

:mod:`rag_server` exposes the pipeline's read-only tools over FastMCP
(streamable_http at ``127.0.0.1:8765/mcp``): ``search_notes``, ``query_notes``,
``get_note``, ``list_notes``, ``get_backlinks``, ``get_recent``, ``reindex``.

The agent is the only consumer. The pipeline has no FastAPI endpoint and no Langfuse.
"""