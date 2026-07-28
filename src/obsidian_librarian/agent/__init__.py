"""Part B — the LangGraph agent.

Submodules:
* :mod:`llm`            — ``ChatOllama`` pointed at Ollama Cloud (``deepseek-v4-pro:cloud``).
* :mod:`prompts`        — assembles the system prompt, loading the Obsidian skill file.
* :mod:`skills.obsidian`— markdown knowledge that makes the agent an Obsidian specialist.
* :mod:`memory`         — ``AsyncSqliteSaver`` checkpointer (carries pending writes across turns).
* :mod:`observability`  — Langfuse ``CallbackHandler`` (agent only; never the pipeline).
* :mod:`graph`          — the graph: ``MultiServerMCPClient(rag + obsidian)`` → tools →
  ``create_react_agent`` with a ``recursion_limit`` guard.

The agent owns **all** vault writes (via the Obsidian plugin MCP) and all Langfuse
tracing. Writes use propose-then-confirm HITL over chat.
"""