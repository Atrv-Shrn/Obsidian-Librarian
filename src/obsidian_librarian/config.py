"""Central configuration & contracts for Obsidian-Librarian.

This module is the **single source of truth** for every tunable and every external
endpoint in the system. Every other module imports settings from here — nothing
reads env vars or hardcodes URLs/ports/models directly.

Two runtime model layers (see SPEC.md, stack table):

* **Local Ollama** (`OLLAMA_LOCAL_BASE_URL`, default `http://127.0.0.1:11434`) serves the
  dense embedding model `nomic-embed-text`. Embeddings stay in-container; the vault
  never leaves the box. Used by our custom `NomicEmbedding` (a LlamaIndex
  `BaseEmbedding` subclass) and the Ragas embedding-based metrics.
* **Ollama Cloud** (`OLLAMA_BASE_URL`, default `https://ollama.com`) is reached via
  `ChatOllama` (`langchain-ollama`): the cloud host goes in `base_url`, `OLLAMA_API_KEY`
  rides as a bearer header in `client_kwargs`. The host carries **no** `/v1` suffix —
  ChatOllama uses the native ollama client (`{base_url}/api/chat`), and a `/v1` would make
  that `/v1/api/chat` → 404. Two distinct cloud models live here, on purpose:
    - `GENERATION_MODEL` (default `deepseek-v4-pro:cloud`) — agent reasoning + RAG
      synthesis (the generator).
    - `JUDGE_MODEL` (default `glm-5.2:cloud`) — Ragas LLM-metric judge. Judge ≠
      generator so it never grades its own output (no self-preference bias).

The RAG pipeline is **read-only** and never uses Langfuse; the agent owns all writes
and all Langfuse tracing. This module carries no opinion about that — it just exposes
the knobs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-driven settings. Loaded once via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Paths -------------------------------------------------------------
    vault_path: Path = Field(
        default=Path("/vault"),
        description="Bind-mounted Obsidian vault (read-write on host; pipeline reads it).",
    )
    data_path: Path = Field(
        default=Path("/data"),
        description="Persistent state: Qdrant, Redis dump, SQLite, Ollama models, checkpointer.",
    )

    # --- Local Ollama (embeddings) ----------------------------------------
    ollama_local_base_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768  # nomic-embed-text output dimensionality
    # Timeout for a single local-Ollama embedding call. Generous on purpose: the FIRST embed
    # after a container start pays a cold model load (Ollama maps ~274 MB off disk and spins up
    # a CPU runner), which routinely blows past a 60 s budget on a small container — observed as
    # `sync error: <note>: timed out` on the very first sync, with the note only picked up on the
    # next scheduled tick. Steady-state embeds take well under a second, so a high ceiling costs
    # nothing; it only bounds a genuinely stuck request.
    embed_request_timeout: float = 300.0

    # --- Ollama Cloud (LLM + judge) ---------------------------------------
    # Native Ollama API host. ChatOllama (langchain-ollama) uses the native ollama client, which
    # appends ``/api/chat`` to this base — so it MUST NOT carry a ``/v1`` suffix (that is the
    # OpenAI-compat path and yields a 404 on ``/v1/api/chat``). Our own OpenAI-compatible ``/v1``
    # endpoint is a separate thing we expose; it is unrelated to how we call Ollama Cloud.
    ollama_base_url: str = "https://ollama.com"
    ollama_api_key: Optional[str] = Field(
        default=None, description="Ollama Cloud API key (routes both deepseek + glm)."
    )
    generation_model: str = "deepseek-v4-pro:cloud"
    judge_model: str = "glm-5.2:cloud"
    llm_temperature: float = 0.2
    llm_request_timeout: float = 120.0

    # --- Qdrant (dense + sparse, server-side RRF fusion) ------------------
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "obsidian_librarian"
    qdrant_api_key: Optional[str] = None

    # --- Redis (raw docstore + content-hash dedup) ------------------------
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_docstore_namespace: str = "librarian/docstore"

    # --- SQLite (sync watermarks + agent checkpointer) --------------------
    sqlite_path: Path = Field(default=Path("/data/librarian.db"))

    # --- Chunking / retrieval knobs ---------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64  # ~12% of chunk_size
    retrieval_top_n: int = 20  # candidates pulled before rerank
    rerank_top_k: int = 6  # nodes passed to the synthesizer
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # FastEmbed cross-encoder
    sparse_model: str = "Qdrant/bm25"  # FastEmbed BM25 sparse encoder
    # Where FastEmbed caches its ONNX models (BM25 sparse + cross-encoder). Defaults to the
    # persistent /data volume so models download once across container recreations.
    fastembed_cache_path: Path = Field(default=Path("/data/caches/fastembed"))

    # nomic task prefixes — prepended to text before embedding
    embed_doc_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    # --- RAG-MCP (read-only, agent-only) ----------------------------------
    rag_mcp_host: str = "127.0.0.1"
    rag_mcp_port: int = 8765
    rag_mcp_path: str = "/mcp"

    # --- Agent -------------------------------------------------------------
    # Hard guard against infinite tool loops. 30, not 12: a legit multi-step write (e.g.
    # "duplicate this note, rename it, rewrite the body") spends several laps — read, copy,
    # re-read, patch — and 12 tripped LangGraph's limit mid-task, returning the confusing
    # "Sorry, need more steps to process this request." even though the writes had landed.
    recursion_limit: int = 30
    write_confirm: bool = True  # propose-then-confirm before any vault write
    pending_write_marker: str = "[PENDING_WRITE]"

    # --- Obsidian Local REST API plugin MCP (writes) ----------------------
    obsidian_mcp_url: str = "https://host.docker.internal:27124/mcp/"
    obsidian_api_key: Optional[str] = Field(default=None, description="Plugin bearer token.")
    obsidian_tls_insecure: bool = True  # plugin ships a self-signed cert

    # --- FastAPI (Seam 2: user ↔ agent) -----------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_default_thread_id: str = "obsidian-librarian-default"
    # Comma-separated list of allowed CORS origins for the OpenAI-compatible endpoint.
    # Empty (default) → restrict to localhost loopback on any port (the endpoint is
    # unauthenticated and write-capable, so "*" would leak pending writes cross-origin).
    # Set to "*" only if you understand the risk and have network-level isolation.
    api_cors_origins: str = ""

    # --- APScheduler (periodic vault re-sync) -----------------------------
    sync_interval_minutes: int = 15
    sync_on_start: bool = True

    # --- Langfuse (agent ONLY — never the pipeline) -----------------------
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ------------------------------------------------------------------ helpers

    @field_validator("vault_path", "data_path", "sqlite_path", "fastembed_cache_path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def rag_mcp_endpoint(self) -> str:
        return f"http://{self.rag_mcp_host}:{self.rag_mcp_port}{self.rag_mcp_path}"

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_path.as_posix()}"

    @property
    def generation_kwargs(self) -> dict:
        return {"temperature": self.llm_temperature, "timeout": self.llm_request_timeout}

    def ensure_dirs(self) -> None:
        """Create the on-disk dirs we need (no-op if they exist). Safe to call repeatedly."""
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings (reads env + .env once)."""
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    """Test helper: drop the cached settings so the next get_settings() re-reads env."""
    get_settings.cache_clear()