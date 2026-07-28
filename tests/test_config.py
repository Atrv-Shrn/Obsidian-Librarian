"""Tests for :mod:`obsidian_librarian.config` — env loading, properties, validators."""

from __future__ import annotations

from pathlib import Path

from obsidian_librarian.config import Settings, get_settings, reset_settings_cache


def test_defaults_when_env_unset(monkeypatch):
    # The autouse fixture sets VAULT_PATH/DATA_PATH/SQLITE_PATH; clear them to see defaults.
    for k in ("VAULT_PATH", "DATA_PATH", "SQLITE_PATH"):
        monkeypatch.delenv(k, raising=False)
    s = Settings()
    # FastEmbed model id (in-process ONNX), not the old Ollama `nomic-embed-text` tag.
    assert s.embed_model == "nomic-ai/nomic-embed-text-v1.5-Q"
    assert s.embed_dim == 768
    assert s.generation_model == "deepseek-v4-pro:cloud"
    assert s.judge_model == "glm-5.2:cloud"
    assert s.qdrant_collection == "obsidian_librarian"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.retrieval_top_n == 20
    assert s.rerank_top_k == 6
    assert s.write_confirm is True
    assert s.pending_write_marker == "[PENDING_WRITE]"
    assert s.recursion_limit == 30
    assert s.api_default_thread_id == "obsidian-librarian-default"
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "other-model:cloud")
    monkeypatch.setenv("EMBED_DIM", "1024")
    monkeypatch.setenv("CHUNK_SIZE", "128")
    monkeypatch.setenv("WRITE_CONFIRM", "false")
    monkeypatch.setenv("RECURSION_LIMIT", "7")
    monkeypatch.setenv("RAG_MCP_PORT", "9999")
    s = Settings()
    assert s.generation_model == "other-model:cloud"
    assert s.embed_dim == 1024
    assert s.chunk_size == 128
    assert s.write_confirm is False
    assert s.recursion_limit == 7
    assert s.rag_mcp_port == 9999


def test_rag_mcp_endpoint_property():
    s = Settings(rag_mcp_host="127.0.0.1", rag_mcp_port=8765, rag_mcp_path="/mcp")
    assert s.rag_mcp_endpoint == "http://127.0.0.1:8765/mcp"


def test_sqlite_url_property(tmp_path):
    s = Settings(sqlite_path=tmp_path / "librarian.db")
    assert s.sqlite_url == f"sqlite:///{(tmp_path / 'librarian.db').as_posix()}"


def test_generation_kwargs_property():
    s = Settings(llm_temperature=0.5, llm_request_timeout=30.0)
    assert s.generation_kwargs == {"temperature": 0.5, "timeout": 30.0}


def test_path_validator_expandsuser(monkeypatch):
    # The validator calls Path.expanduser(); with no ~ it is a no-op but must return a Path.
    monkeypatch.setenv("SQLITE_PATH", str(Path("./x.db")))
    s = Settings()
    assert isinstance(s.sqlite_path, Path)


def test_ensure_dirs_creates_data_path(tmp_path, monkeypatch):
    for k in ("VAULT_PATH", "DATA_PATH", "SQLITE_PATH"):
        monkeypatch.delenv(k, raising=False)
    data = tmp_path / "freshdata"
    assert not data.exists()
    s = Settings(data_path=data, sqlite_path=data / "librarian.db")
    s.ensure_dirs()
    assert data.exists()
    assert (data).is_dir()
    assert s.sqlite_path.parent.exists()


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "cached-model:cloud")
    reset_settings_cache()
    a = get_settings()
    b = get_settings()
    assert a is b  # lru_cache singleton
    assert a.generation_model == "cached-model:cloud"


def test_reset_settings_cache_rereads_env(monkeypatch):
    reset_settings_cache()
    monkeypatch.setenv("GENERATION_MODEL", "first:cloud")
    assert get_settings().generation_model == "first:cloud"
    reset_settings_cache()
    monkeypatch.setenv("GENERATION_MODEL", "second:cloud")
    assert get_settings().generation_model == "second:cloud"


def test_extra_env_ignored(monkeypatch):
    # case_sensitive=False + extra="ignore" → unknown keys don't error.
    monkeypatch.setenv("SOMETHING_UNRELATED", "x")
    s = Settings()
    assert not hasattr(s, "something_unrelated")