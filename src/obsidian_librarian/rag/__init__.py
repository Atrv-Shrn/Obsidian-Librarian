"""Part A — the read-only RAG pipeline.

Submodules:
* :mod:`embeddings`  — Ollama nomic dense + FastEmbed BM25 sparse / cross-encoder rerank.
* :mod:`reader`      — wraps LlamaHub ``ObsidianReader`` (+ tags/frontmatter extension).
* :mod:`ingest`      — parse → split → embed (dense+sparse) → Qdrant; Redis raw + dedup.
* :mod:`retrieve`    — Qdrant hybrid retrieve (RRF) + cross-encoder rerank.
* :mod:`query_engine`— retrieve → rerank → synthesize a grounded answer + citations.
* :mod:`sync`        — SQLite watermarks + APScheduler-driven periodic re-sync.

This package **never writes to the vault and never calls Langfuse**.
"""