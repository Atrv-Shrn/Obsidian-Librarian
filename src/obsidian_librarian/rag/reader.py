"""Vault reader — wraps LlamaHub ``ObsidianReader`` with a safe local fallback.

Prefers ``llama-index-readers-obsidian``'s ``ObsidianReader`` (wikilinks, backlinks,
folder, tasks → metadata). Because LlamaHub import paths vary across versions, we
attempt several and fall back to :class:`_LocalObsidianReader` — a small self-contained
reader that produces the same metadata shape (``file_path``, ``title``, ``tags``,
``wikilinks``, ``backlinks``, ``frontmatter``, ``mtime``). Either way the pipeline gets
LlamaIndex :class:`Document` objects.

The local fallback is not a downgrade for correctness — it parses YAML frontmatter,
inline ``#tags``, ``[[wikilinks]]``/``[[Note|alias]]``/``[[Note#^block]]``, and computes
backlinks across the vault. ObsidianReader is used when available because it also
handles tasks and folder structure natively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.schema import Document

from ..config import get_settings

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_TAG_RE = re.compile(r"(?<![\w&])#([A-Za-z0-9_\-/]+)")


def _try_import_obsidian_reader() -> Optional[type]:
    for mod_path, attr in (
        ("llama_index.readers.obsidian", "ObsidianReader"),
        ("llama_index.readers.obsidian.base", "ObsidianReader"),
        ("llama_index_readers_obsidian", "ObsidianReader"),
        ("llama_hub.obsidian.base", "ObsidianReader"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[attr])
            return getattr(mod, attr)
        except Exception:
            continue
    return None


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split a leading ``---`` YAML block. Minimal parser (no pyyaml dependency)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    fm: Dict[str, Any] = {}
    key: Optional[str] = None
    for line in block.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.startswith("  ") and key:
            # list item under current key
            val = line.strip().lstrip("-").strip()
            if isinstance(fm.get(key), list):
                fm[key].append(val)
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            key = k.strip()
            v = v.strip()
            if v == "":
                fm[key] = []
            elif v.startswith("[") and v.endswith("]"):
                fm[key] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
            else:
                fm[key] = v.strip("'\"")
    return fm, body


def _extract_wikilinks(text: str) -> List[str]:
    return [m.group(1) for m in _WIKILINK_RE.finditer(text)]


def _extract_tags(text: str) -> List[str]:
    return [m.group(1) for m in _TAG_RE.finditer(text)]


@dataclass
class _LocalObsidianReader:
    """Self-contained fallback reader producing LlamaIndex Documents with rich metadata."""

    vault_path: Path
    _backlinks: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)

    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        vault = self.vault_path
        docs: List[Document] = []
        raw_links: Dict[str, List[str]] = {}  # file -> wikilinks
        for md in sorted(vault.rglob("*.md")):
            if any(p in md.parts for p in (".obsidian", ".trash", ".git")):
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
            fm, body = _parse_frontmatter(text)
            rel = md.relative_to(vault).as_posix()
            wikilinks = _extract_wikilinks(text)
            tags = set(_extract_tags(body))
            tags.update(_to_list(fm.get("tags")))
            title = str(fm.get("title", md.stem))
            doc = Document(
                text=body,
                metadata={
                    "file_path": rel,
                    "title": title,
                    "tags": sorted(tags),
                    "wikilinks": wikilinks,
                    "frontmatter": fm,
                    "mtime": md.stat().st_mtime,
                },
            )
            docs.append(doc)
            raw_links[rel] = wikilinks
        # backlinks: for each note, who links to it (by title/filename match)
        by_target: Dict[str, List[str]] = {}
        for src, links in raw_links.items():
            for link in links:
                target = link.split("|")[0].split("#")[0].split("^")[0].strip()
                if not target:
                    continue
                by_target.setdefault(target, []).append(src)
        for doc in docs:
            title = doc.metadata["title"]
            stem = Path(doc.metadata["file_path"]).stem
            doc.metadata["backlinks"] = sorted(
                set(by_target.get(title, []) + by_target.get(stem, []))
            )
        self._backlinks = by_target
        return docs


def _to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def compute_backlinks(vault_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Vault-wide backlink map: relative path → sorted list of source paths that link to it.

    This is the lightweight scan the sync path uses to enrich single-note reads. ``load_vault``
    computes backlinks as a side effect of reading everything; the incremental sync path reads
    one note at a time via :func:`load_note` (which cannot know backlinks in isolation), so it
    calls this once per sync run and injects the result into each note's metadata. Only
    wikilinks are scanned here — no frontmatter/parse — so it is cheap even for large vaults.
    """
    vault = vault_path or get_settings().vault_path
    if not vault.exists():
        return {}
    raw_links: Dict[str, List[str]] = {}
    titles: Dict[str, str] = {}
    by_target: Dict[str, List[str]] = {}
    for md in sorted(vault.rglob("*.md")):
        if any(p in md.parts for p in (".obsidian", ".trash", ".git")):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = md.relative_to(vault).as_posix()
        fm, _body = _parse_frontmatter(text)
        titles[rel] = str(fm.get("title", md.stem))
        links = _extract_wikilinks(text)
        raw_links[rel] = links
        for link in links:
            target = link.split("|")[0].split("#")[0].split("^")[0].strip()
            if target:
                by_target.setdefault(target, []).append(rel)
    out: Dict[str, List[str]] = {}
    for rel in raw_links:
        title = titles[rel]
        stem = Path(rel).stem
        out[rel] = sorted(set(by_target.get(title, []) + by_target.get(stem, [])))
    return out


def load_vault(vault_path: Optional[Path] = None) -> List[Document]:
    """Read every ``.md`` in the vault into LlamaIndex Documents with rich metadata."""
    vault = vault_path or get_settings().vault_path
    Reader = _try_import_obsidian_reader()
    if Reader is not None:
        try:
            reader = Reader(vault_path=str(vault))  # type: ignore[call-arg]
            docs = reader.load_data()
            # Normalize metadata keys so downstream code can rely on them.
            for d in docs:
                m = d.metadata or {}
                m.setdefault("file_path", m.get("path") or m.get("file_name") or "")
                m.setdefault("tags", _to_list(m.get("tags")))
                m.setdefault("wikilinks", _to_list(m.get("wikilinks")))
                m.setdefault("backlinks", _to_list(m.get("backlinks")))
                m.setdefault("frontmatter", {})
                d.metadata = m
            return docs
        except Exception:
            pass  # fall through to local reader
    return _LocalObsidianReader(vault_path=vault).load_data()


def load_note(path: str, vault_path: Optional[Path] = None) -> Optional[Document]:
    """Read a single note by relative path (used by the ``get_note`` MCP tool).

    The path is resolved and checked for vault containment before any file is read: a caller
    (or a crafted agent request) passing ``../secret.md`` would otherwise walk out of the vault
    and read arbitrary host files. Resolve both sides against the same root and require the
    resolved note to live inside the resolved vault.
    """
    vault = vault_path or get_settings().vault_path
    md = vault / path
    try:
        inside = md.resolve().is_relative_to(vault.resolve())
    except OSError:  # broken symlink / unreadable path → treat as not-in-vault
        return None
    if not inside or not md.exists() or md.suffix != ".md":
        return None
    text = md.read_text(encoding="utf-8", errors="replace")
    fm, body = _parse_frontmatter(text)
    return Document(
        text=body,
        metadata={
            "file_path": md.relative_to(vault).as_posix(),
            "title": str(fm.get("title", md.stem)),
            "tags": sorted(set(_extract_tags(body)) | set(_to_list(fm.get("tags")))),
            "wikilinks": _extract_wikilinks(text),
            "frontmatter": fm,
            "mtime": md.stat().st_mtime,
        },
    )