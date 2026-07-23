"""RAG-MCP — the read-only tool surface the agent consumes (Seam 1, read side).

FastMCP server, streamable-http at ``127.0.0.1:8765/mcp``. Tools:

* ``search_notes``  — retrieve + rerank (no generation) → ranked chunks + citations.
* ``query_notes``   — full RAG (retrieve + rerank + generate) → grounded answer + sources.
* ``get_note``      — full raw note by path.
* ``list_notes``    — enumerate indexed paths (optional prefix filter).
* ``get_backlinks`` — notes that link to a given note.
* ``get_recent``    — most-recently-modified notes.
* ``reindex``       — force a sync (whole vault or one path).

The pipeline is **read-only**: no tool here writes to the vault. The agent does all
writing through the Obsidian plugin MCP (Part B). This server has no FastAPI endpoint
and never calls Langfuse.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

from ..config import get_settings
from ..rag import ingest
from ..rag.query_engine import query_notes as _query_notes
from ..rag.query_engine import search as _search
from ..rag.reader import load_note, load_vault

mcp = FastMCP("obsidian-librarian-rag")


# --------------------------------------------------------------------------- tools


@mcp.tool()
def search_notes(
    query: str,
    k: int = 6,
    path_prefix: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """Search the vault: hybrid retrieve + cross-encoder rerank. Returns ranked chunks
    (path, title, heading, text, tags, score). No answer generation."""
    return _search(query, k=k, path_prefix=path_prefix, tags=tags)


@mcp.tool()
def query_notes(
    query: str,
    top_k: int = 6,
    path_prefix: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Ask the vault a question and get a grounded answer with [[wikilink]] citations.
    Full RAG (retrieve + rerank + synthesize). This is the evaluated path."""
    return _query_notes(query, top_k=top_k, path_prefix=path_prefix, tags=tags)


@mcp.tool()
def get_note(path: str) -> dict:
    """Return a full raw note by relative path: text + title + tags + wikilinks + frontmatter."""
    doc = load_note(path)
    if doc is None:
        return {"error": f"note not found: {path}"}
    return {
        "path": doc.metadata.get("file_path", path),
        "title": doc.metadata.get("title", ""),
        "text": doc.text,
        "tags": doc.metadata.get("tags", []),
        "wikilinks": doc.metadata.get("wikilinks", []),
        "frontmatter": doc.metadata.get("frontmatter", {}),
    }


@mcp.tool()
def list_notes(path_prefix: Optional[str] = None) -> list[str]:
    """List indexed note paths, optionally filtered by a path prefix."""
    paths = ingest.get_indexed_paths()
    if path_prefix:
        paths = [p for p in paths if p.startswith(path_prefix)]
    return sorted(paths)


@mcp.tool()
def get_backlinks(path: str) -> list[str]:
    """Return paths of notes that link to the given note (by title or filename).

    Mirrors :func:`reader.compute_backlinks`: the indexer matches a wikilink's target against
    the note's frontmatter ``title`` (defaulting to the stem) **and** its stem, so a note
    ``bayes.md`` titled ``Bayesian Reasoning`` is backlinked by ``[[Bayesian Reasoning]]``.
    Matching only the stem here would return an incomplete set that contradicts the
    ``backlinks`` field the Qdrant payload carries, so we resolve the target note's title from
    the vault snapshot and test link targets against it too.
    """
    docs = _vault_docs()
    target_stem = Path(path).stem
    target_name = Path(path).name
    target_title = ""
    for d in docs:
        if d.metadata.get("file_path", "") == path:
            target_title = str(d.metadata.get("title", "") or "")
            break
    back: list[str] = []
    for d in docs:
        links = d.metadata.get("wikilinks", [])
        for link in links:
            t = link.split("|")[0].split("#")[0].split("^")[0].strip()
            if (
                t == target_stem
                or t == path
                or t == target_name
                or (bool(target_title) and t == target_title)
            ):
                back.append(d.metadata.get("file_path", ""))
                break
    return sorted(set(b for b in back if b))


@mcp.tool()
def get_recent(n: int = 10) -> list[str]:
    """Return the paths of the N most-recently-modified notes (newest first).

    Per SPEC, ``get_recent`` returns paths (like ``list_notes`` / ``get_backlinks``); use
    ``get_note`` to read a path's full text/title/tags when needed.
    """
    docs = _vault_docs()
    ranked = sorted(docs, key=lambda d: d.metadata.get("mtime", 0.0), reverse=True)[:n]
    return [d.metadata.get("file_path", "") for d in ranked]


@mcp.tool()
def reindex(path: Optional[str] = None) -> dict:
    """Force a vault re-sync. Pass ``path`` to re-index a single note; omit for the whole vault.

    Invalidates the cached vault read so ``get_backlinks`` / ``get_recent`` see the post-sync
    notes on their next call rather than the pre-sync snapshot.
    """
    result = ingest.reindex(path=path)
    _reset_vault_docs_cache()
    return result


# --------------------------------------------------------------------------- helpers


# TTL cache for the vault read. ``get_backlinks`` / ``get_recent`` are cheap lookups over the
# full vault document set; re-reading the vault on every call would be wasteful, but a process-
# lifetime cache (the old ``@lru_cache(maxsize=1)``) serves a stale snapshot until the next full
# ``reindex`` — so edits made through the agent wouldn't show up in backlinks/recency until then.
# Instead we keep the snapshot for ``sync_interval_minutes`` (the same cadence the scheduler
# uses), and ``reindex`` forces an immediate reset. ``time.monotonic`` is immune to wall-clock
# jumps. Held in a module-level mutable so the test isolation fixture can reset it.
_VAULT_DOCS_CACHE: Optional[tuple[float, list]] = None


def _reset_vault_docs_cache() -> None:
    """Drop the cached vault snapshot so the next ``_vault_docs()`` re-reads the vault."""
    global _VAULT_DOCS_CACHE
    _VAULT_DOCS_CACHE = None


def _vault_docs() -> list:
    """Cached vault read, refreshed every ``sync_interval_minutes`` (TTL, not lru_cache)."""
    global _VAULT_DOCS_CACHE
    s = get_settings()
    ttl = max(1, int(s.sync_interval_minutes)) * 60.0
    now = time.monotonic()
    if _VAULT_DOCS_CACHE is not None:
        expires_at, docs = _VAULT_DOCS_CACHE
        if now < expires_at:
            return docs
    docs = load_vault()
    _VAULT_DOCS_CACHE = (now + ttl, docs)
    return docs


# Expose a ``cache_clear``-compatible reset so the shared test-isolation fixture
# (which clears the project's lru singletons via ``getattr(fn, "cache_clear", None)``)
# also resets this TTL cache without needing a special case.
_vault_docs.cache_clear = _reset_vault_docs_cache  # type: ignore[attr-defined]


def main() -> None:
    s = get_settings()
    mcp.run(
        transport="streamable-http",
        host=s.rag_mcp_host,
        port=s.rag_mcp_port,
        path=s.rag_mcp_path,
    )


if __name__ == "__main__":
    main()