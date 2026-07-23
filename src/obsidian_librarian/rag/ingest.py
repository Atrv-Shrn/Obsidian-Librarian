"""Ingestion — parse → split → embed (dense + sparse) → index. Read-only over the vault.

This module owns the write side of the **index** (Qdrant points, Redis docstore, SQLite
watermarks). It never writes to the vault itself.

Design (see ``docs/SPEC.md``):

* **Chunking**: header-aware via ``MarkdownNodeParser``; long sections are then size-split
  to ≈ ``chunk_size`` tokens (approximated as ``chunk_size * 4`` chars) with ``chunk_overlap``
  overlap. Each chunk carries its heading path + the note's tags/wikilinks/frontmatter.
* **IDs are deterministic**: ``uuid5(NAMESPACE, f"{rel_path}::{chunk_index}")`` → re-indexing
  is idempotent (upsert overwrites the same points).
* **Qdrant**: one collection with a named dense vector (``dense``, nomic cosine) and a named
  sparse vector (``sparse``, BM25) per point — the shape the hybrid retriever fuses.
* **Redis**: raw note text + per-node docstore + a content-hash set for dedup + a per-path
  node-id set so an update/delete can purge exactly that note's points.
* **SQLite**: watermarks drive the sync diff (see :mod:`.sync.watermarks`).

We drive ``qdrant-client`` directly for the hybrid mechanics (named dense+sparse + server-side
RRF in :mod:`retrieve`) rather than LlamaIndex's vector-store wrapper, so we control the exact
fusion the spec calls for. LlamaIndex still owns parse / split / synthesize.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document, TextNode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from redis import Redis

from ..config import get_settings
from .embeddings import (
    get_dense_embed_model,
    get_reranker,  # noqa: F401  (re-exported for convenience)
    get_sparse_embed_model,
    sparse_to_qdrant,
)
from .reader import compute_backlinks, load_note, load_vault
from .sync import watermarks

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
_NAMESPACE = uuid.NAMESPACE_URL

# Files/dirs to skip when scanning the vault.
_SKIP_PARTS = {".obsidian", ".trash", ".git"}


def _now_utc_iso() -> str:
    """TZ-aware UTC timestamp for the ``indexed_at`` watermark column.

    Naive ``time.strftime`` would stamp with container-local time and no offset —
    ambiguous across hosts/DST. ISO-8601 UTC is unambiguous and sorts correctly.
    """
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- ids


def node_uuid(rel_path: str, chunk_index: int) -> str:
    """Stable UUID for a chunk → Qdrant point id. Re-indexing overwrites the same point."""
    return str(uuid.uuid5(_NAMESPACE, f"{rel_path}::{chunk_index}"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- stores


@dataclass
class QdrantStore:
    client: QdrantClient
    collection: str
    dim: int

    @classmethod
    def from_settings(cls) -> "QdrantStore":
        s = get_settings()
        client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=30)
        return cls(client=client, collection=s.qdrant_collection, dim=s.embed_dim)

    def ensure_collection(self) -> None:
        cols = {c.name for c in self.client.get_collections().collections}
        if self.collection not in cols:
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={DENSE_NAME: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
                    sparse_vectors_config={SPARSE_NAME: qm.SparseVectorParams()},
                )
            except Exception:
                # Create-then-index race: another worker may have created the collection
                # in the window between our ``get_collections`` and ``create_collection``.
                # Re-check; if it now exists, fall through to payload-index setup. Otherwise
                # the failure is real — re-raise so the caller sees it.
                cols = {c.name for c in self.client.get_collections().collections}
                if self.collection not in cols:
                    raise
        # Full-text index on ``path`` so ``MatchText`` filters (path_prefix) work.
        # The PREFIX tokenizer indexes every prefix of each path token, giving us
        # prefix-style narrowing. Best-effort: re-creating an existing index throws
        # and is swallowed; path_prefix is an optional narrow, not a load-bearing path.
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="path",
                field_schema=qm.TextIndexParams(
                    type=qm.TextIndexType.TEXT,
                    tokenizer=qm.TokenizerType.PREFIX,
                    lowercase=True,
                    min_token_len=2,
                ),
            )
        except Exception:
            pass

    def upsert_points(self, points: List[qm.PointStruct]) -> None:
        if not points:
            return
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_by_path(self, rel_path: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="path", match=qm.MatchValue(value=rel_path))])
            ),
            wait=True,
        )


@dataclass
class RedisStore:
    client: Redis
    ns: str

    @classmethod
    def from_settings(cls) -> "RedisStore":
        s = get_settings()
        return cls(client=Redis.from_url(s.redis_url, decode_responses=True), ns=s.redis_docstore_namespace)

    def _k(self, *parts: str) -> str:
        return "/".join((self.ns, *parts))

    def store_node(self, node_id: str, payload: Dict[str, Any]) -> None:
        self.client.set(self._k("node", node_id), json.dumps(payload))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._k("node", node_id))
        return json.loads(raw) if raw else None

    def store_raw(self, rel_path: str, text: str) -> None:
        self.client.set(self._k("raw", rel_path), text)

    def get_raw(self, rel_path: str) -> Optional[str]:
        return self.client.get(self._k("raw", rel_path))

    def add_path_nodes(self, rel_path: str, node_ids: List[str]) -> None:
        if node_ids:
            self.client.sadd(self._k("path", rel_path), *node_ids)

    def path_nodes(self, rel_path: str) -> List[str]:
        return list(self.client.smembers(self._k("path", rel_path)))

    def purge_path(self, rel_path: str) -> None:
        for nid in self.path_nodes(rel_path):
            self.client.delete(self._k("node", nid))
        self.client.delete(self._k("path", rel_path), self._k("raw", rel_path))

    def add_hash(self, sha: str) -> bool:
        """Returns True if the hash was new (not seen before) → not a dup."""
        return bool(self.client.sadd(self._k("hashes"), sha))

    def remove_hash(self, sha: str) -> None:
        self.client.srem(self._k("hashes"), sha)


# --------------------------------------------------------------------------- chunking


def _heading_path(node_metadata: Dict[str, Any]) -> str:
    parts = [node_metadata.get(f"Header {i}") for i in range(1, 7)]
    return " > ".join(str(p) for p in parts if p)


def _split_to_size(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap if end - overlap > start else end
    return out


def chunk_documents(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[TextNode]:
    """Header-aware split, then size-split long sections. Assigns chunk_index per file."""
    parser = MarkdownNodeParser()
    max_chars = max(chunk_size * 4, 512)
    overlap_chars = max(chunk_overlap * 4, 64)
    nodes: List[TextNode] = []
    for doc in documents:
        rel = doc.metadata.get("file_path", "")
        header_nodes = parser.get_nodes_from_documents([doc])
        idx = 0
        for hn in header_nodes:
            heading = _heading_path(hn.metadata or {})
            for piece in _split_to_size(hn.text, max_chars, overlap_chars):
                if not piece.strip():
                    continue
                md = {
                    "path": rel,
                    "title": doc.metadata.get("title", Path(rel).stem),
                    "heading_path": heading,
                    "tags": list(doc.metadata.get("tags", [])),
                    "wikilinks": list(doc.metadata.get("wikilinks", [])),
                    "backlinks": list(doc.metadata.get("backlinks", [])),
                    "frontmatter": doc.metadata.get("frontmatter", {}),
                    "chunk_index": idx,
                }
                nodes.append(TextNode(text=piece, metadata=md))
                idx += 1
    return nodes


# --------------------------------------------------------------------------- upsert


def _breadcrumb(node: TextNode) -> str:
    """Title / heading-path / tags breadcrumb prepended to the chunk text before embedding.

    SPEC: chunks embed a ``title / heading-path / tags`` breadcrumb with the text so the
    dense + sparse vectors carry topical context the raw prose alone lacks. The payload's
    ``text`` stays the *raw* chunk text (citations and the synthesizer see prose, not the
    breadcrumb); only the *embedded* text gets the prefix.
    """
    title = node.metadata.get("title") or ""
    heading = node.metadata.get("heading_path", "")
    tags = node.metadata.get("tags", []) or []
    parts = [p for p in (title, heading) if p]
    if tags:
        parts.append("tags:" + ",".join(tags))
    return " | ".join(parts)


def _embed_text(node: TextNode) -> str:
    bc = _breadcrumb(node)
    return f"{bc}\n{node.text}" if bc else node.text


def _build_point(node: TextNode, dense_model, sparse_model) -> qm.PointStruct:
    rel = node.metadata["path"]
    nid = node_uuid(rel, node.metadata["chunk_index"])
    emb = _embed_text(node)
    dense = dense_model._get_text_embedding(emb)
    sparse = next(sparse_model.embed([emb]))
    payload = {
        "text": node.text,
        "path": rel,
        "title": node.metadata.get("title"),
        "heading_path": node.metadata.get("heading_path", ""),
        "tags": node.metadata.get("tags", []),
        "wikilinks": node.metadata.get("wikilinks", []),
        "backlinks": node.metadata.get("backlinks", []),
        "frontmatter": node.metadata.get("frontmatter", {}),
        "chunk_index": node.metadata.get("chunk_index", 0),
    }
    return qm.PointStruct(
        id=nid,
        vector={DENSE_NAME: dense, SPARSE_NAME: sparse_to_qdrant(sparse)},
        payload=payload,
    )


def index_documents(documents: List[Document]) -> int:
    """Chunk, embed (dense+sparse), and upsert a batch of Documents. Returns chunk count."""
    s = get_settings()
    qs = QdrantStore.from_settings()
    qs.ensure_collection()
    rs = RedisStore.from_settings()
    dense = get_dense_embed_model()
    sparse = get_sparse_embed_model()

    nodes = chunk_documents(documents, s.chunk_size, s.chunk_overlap)
    # group by file so per-path node-id sets + raw store stay consistent
    by_path: Dict[str, List[TextNode]] = {}
    for n in nodes:
        by_path.setdefault(n.metadata["path"], []).append(n)

    points: List[qm.PointStruct] = []
    for rel, file_nodes in by_path.items():
        # Purge the previous copy of this file from BOTH stores before re-adding. Redis
        # purge drops the docstore node/raw/hash entries; the Qdrant delete drops every
        # point whose payload ``path`` == rel. The Qdrant delete is the load-bearing one
        # for correctness: point ids are ``uuid5(path::chunk_index)``, so when a note
        # shrinks from N to M chunks, ids ::M..N-1 are never overwritten by the upsert
        # below and would otherwise linger as stale, still-citable text. Deleting by path
        # first guarantees only the current chunk set is retrievable after a modify/reindex.
        rs.purge_path(rel)
        qs.delete_by_path(rel)
        ids: List[str] = []
        for n in file_nodes:
            p = _build_point(n, dense, sparse)
            points.append(p)
            nid = p.id
            ids.append(nid)
            rs.store_node(
                nid,
                {
                    "text": n.text,
                    "path": rel,
                    "title": n.metadata.get("title"),
                    "heading_path": n.metadata.get("heading_path", ""),
                    "tags": n.metadata.get("tags", []),
                    "wikilinks": n.metadata.get("wikilinks", []),
                    "backlinks": n.metadata.get("backlinks", []),
                    "frontmatter": n.metadata.get("frontmatter", {}),
                },
            )
        rs.add_path_nodes(rel, ids)
        # Raw docstore: SPEC says the Redis raw store holds the *full raw markdown* (with the
        # leading ``---`` frontmatter block) for citations. The source Document.text is the
        # frontmatter-stripped body (``load_note``/``load_vault`` strip the YAML block), so
        # re-read the verbatim file from disk rather than storing ``src.text`` — otherwise the
        # raw store would hold only the body and silently diverge from the spec. Fall back to
        # the stripped body only if the file can't be re-read (e.g. deleted mid-sync).
        src = next((d for d in documents if d.metadata.get("file_path") == rel), None)
        if src is not None:
            try:
                raw_md = (get_settings().vault_path / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw_md = src.text
            rs.store_raw(rel, raw_md)

    qs.upsert_points(points)
    return len(points)


# --------------------------------------------------------------------------- scan + sync


def scan_vault(vault_path: Optional[Path] = None) -> Dict[str, Tuple[str, float]]:
    """Live filesystem scan: relative path → (sha256, mtime)."""
    vault = vault_path or get_settings().vault_path
    out: Dict[str, Tuple[str, float]] = {}
    if not vault.exists():
        return out
    for md in vault.rglob("*.md"):
        if any(p in _SKIP_PARTS for p in md.parts):
            continue
        rel = md.relative_to(vault).as_posix()
        out[rel] = (file_sha256(md), md.stat().st_mtime)
    return out


@dataclass
class SyncReport:
    added: int = 0
    modified: int = 0
    deleted: int = 0
    skipped_dup: int = 0
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "skipped_dup": self.skipped_dup,
            "errors": self.errors,
        }


def sync_vault(vault_path: Optional[Path] = None) -> SyncReport:
    """Diff the filesystem against watermarks and apply index CRUD. Index-only, never touches files."""
    s = get_settings()
    vault = vault_path or s.vault_path
    rs = RedisStore.from_settings()
    qs = QdrantStore.from_settings()
    qs.ensure_collection()

    scan = scan_vault(vault)
    # Backlinks are a vault-graph property: a single note cannot know who links to it without
    # seeing every other note. ``load_note`` (per-file, used below) cannot compute them, so we
    # scan the vault's wikilinks once and inject each note's backlinks into its metadata before
    # indexing — otherwise the sync path (the production path) would index empty backlinks for
    # every note while the full ``load_vault`` reindex path would populate them.
    backlinks = compute_backlinks(vault)
    with watermarks.connection() as conn:
        wm = watermarks.load_all(conn)
        d = watermarks.diff(scan, wm)
        report = SyncReport()

        for rel in d.added + d.modified:
            try:
                sha, mtime = scan[rel]
                doc = load_note(rel, vault)
                if doc is None:
                    continue
                # Enrich the single-note read with the vault-wide backlink map (load_note sets
                # ``backlinks`` to [] because it can't see the rest of the vault).
                doc.metadata["backlinks"] = backlinks.get(rel, [])
                # Maintain the content-hash dedup set. On a modify, drop the previous hash
                # first so a content revert doesn't read as a dup of a stale entry.
                if rel in d.modified and rel in wm:
                    rs.remove_hash(wm[rel][0])
                # Index FIRST, register the hash only once it succeeded. Registering before
                # meant a failed index (e.g. the embed timing out on a cold model load) still
                # left its hash in the set, so the next sync's retry saw ``add_hash`` return
                # False and mis-reported the file as ``skipped_dup`` — a phantom duplicate for
                # what was really a first successful index. Observed on the very first run:
                # `errs=1` then `+a=1 dup=1` on the retry for the same note.
                index_documents([doc])
                is_new_hash = rs.add_hash(sha)
                # Points are path-keyed (uuid5(path::chunk_index)), so even identical content
                # under a different path gets its own retrievable points — we always index, and
                # only *count* the content duplicate for reporting.
                if not is_new_hash:
                    report.skipped_dup += 1
                watermarks.upsert(conn, rel, sha, mtime, _now_utc_iso())
                if rel in d.added:
                    report.added += 1
                else:
                    report.modified += 1
            except Exception as e:  # pragma: no cover - per-file resilience
                report.errors.append(f"{rel}: {e}")

        for rel in d.deleted:
            try:
                qs.delete_by_path(rel)
                rs.purge_path(rel)
                sha = wm[rel][0]
                rs.remove_hash(sha)
                watermarks.delete(conn, rel)
                report.deleted += 1
            except Exception as e:  # pragma: no cover
                report.errors.append(f"delete {rel}: {e}")

    return report


def reindex(path: Optional[str] = None, vault_path: Optional[Path] = None) -> Dict[str, Any]:
    """Force a sync (whole vault) or re-index a single path. Exposed via the ``reindex`` MCP tool."""
    if path is None:
        return sync_vault(vault_path).as_dict()
    vault = vault_path or get_settings().vault_path
    md = vault / path
    # Containment before existence: confirm the resolved path stays under the vault *first*,
    # so this endpoint can't be turned into a traversal existence oracle — a caller probing
    # ``../secrets`` would otherwise learn whether an out-of-vault path exists. Mirrors the
    # guard inside ``load_note``.
    if not md.resolve().is_relative_to(vault.resolve()):
        return {"error": f"not in vault: {path}"}
    if not md.exists():
        return {"error": f"not found: {path}"}
    doc = load_note(path, vault)
    if doc is None:
        return {"error": f"could not read: {path}"}
    # A forced single-note reindex still needs the vault-wide backlink map (load_note can't
    # compute backlinks in isolation); inject it so the reindexed note carries real backlinks.
    doc.metadata["backlinks"] = compute_backlinks(vault).get(path, [])
    n = index_documents([doc])
    rs = RedisStore.from_settings()
    with watermarks.connection() as conn:
        # Keep the content-hash set consistent with a forced reindex: drop the prior hash
        # (if any) then add the current one.
        wm = watermarks.load_all(conn)
        if path in wm:
            rs.remove_hash(wm[path][0])
        sha = file_sha256(md)
        rs.add_hash(sha)
        watermarks.upsert(conn, path, sha, md.stat().st_mtime, _now_utc_iso())
    return {"path": path, "chunks_indexed": n}


def get_indexed_paths() -> List[str]:
    """Return all paths currently in the watermark table (for ``list_notes``)."""
    with watermarks.connection() as conn:
        return list(watermarks.load_all(conn).keys())