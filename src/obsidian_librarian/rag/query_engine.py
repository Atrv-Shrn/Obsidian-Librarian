"""Query engine — retrieve → rerank → synthesize a grounded answer + citations.

This is the generating path (``query_notes``) that Ragas evaluates. The synthesizer is the
cloud generation model (``deepseek-v4-pro:cloud`` via Ollama Cloud, called through
``langchain_ollama.ChatOllama``), fed the reranked nodes' text + metadata and instructed to
answer **grounded**, cite sources as ``[[wikilink]]``s, and say plainly when the vault
doesn't contain the answer.

``search_notes`` is the non-generating sibling (retrieve + rerank only) — see :func:`search`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from llama_index.core.schema import NodeWithScore

from ..config import get_settings
from .retrieve import retrieve

__all__ = ["get_generation_llm", "query_notes", "search", "format_sources"]

_CITATION_INSTRUCTIONS = """\
You are the Obsidian Librarian. Answer the user's question using ONLY the provided vault
excerpts. Rules:
1. Ground every claim in the excerpts; if they don't contain the answer, say \
"I don't have enough in the vault to answer that." — do not guess.
2. Cite sources inline as Obsidian wikilinks using each excerpt's **source** path or title, \
e.g. [[Bayesian Reasoning]]. Use the note's title (filename without .md) when available.
3. Prefer a concise, well-structured answer (short paragraphs or bullets).
4. Preserve exact names, tags, and jargon from the notes.

Vault excerpts (each prefixed with its source):
{context}

User question: {question}
Answer:"""


@lru_cache(maxsize=1)
def get_generation_llm() -> ChatOllama:
    """The pipeline synthesizer: deepseek via Ollama Cloud (``ChatOllama`` over ``langchain-ollama``).

    SPEC's "organs" table locks the model call as ``ChatOllama``; the cloud host is passed via
    ``base_url`` and the ``OLLAMA_API_KEY`` as a bearer header in ``client_kwargs``.
    """
    s = get_settings()
    api_key = s.ollama_api_key or "EMPTY"
    return ChatOllama(
        model=s.generation_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": s.llm_request_timeout,
        },
    )


def _context_block(nodes: List[NodeWithScore]) -> str:
    lines: List[str] = []
    for i, nws in enumerate(nodes, 1):
        nd = nws.node
        md = nd.metadata or {}
        src = md.get("title") or md.get("path") or "unknown"
        # SPEC: the reranked nodes pass *all* their metadata to the synthesizer — path,
        # heading path, tags, wikilinks, backlinks, frontmatter — not just source + heading.
        meta_bits: List[str] = []
        heading = md.get("heading_path", "")
        if heading:
            meta_bits.append(f"heading: {heading}")
        tags = md.get("tags", []) or []
        if tags:
            meta_bits.append("tags: " + ",".join(tags))
        wikilinks = md.get("wikilinks", []) or []
        if wikilinks:
            meta_bits.append("wikilinks: " + ",".join(wikilinks))
        backlinks = md.get("backlinks", []) or []
        if backlinks:
            meta_bits.append("backlinks: " + ",".join(backlinks))
        fm = md.get("frontmatter", {}) or {}
        if fm:
            fm_bits = [f"{k}={v!r}" for k, v in fm.items() if k not in ("tags", "title")]
            if fm_bits:
                meta_bits.append("frontmatter: " + "; ".join(fm_bits))
        meta = f" ({'; '.join(meta_bits)})" if meta_bits else ""
        lines.append(f"[{i}] source: [[{src}]]{meta}\n{nd.text}")
    return "\n\n".join(lines)


def format_sources(nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    sources: List[Dict[str, Any]] = []
    for nws in nodes:
        p = nws.node.metadata.get("path", "")
        if not p or p in seen:
            continue
        seen.add(p)
        sources.append(
            {
                "path": p,
                "title": nws.node.metadata.get("title", ""),
                "heading_path": nws.node.metadata.get("heading_path", ""),
                "tags": nws.node.metadata.get("tags", []),
                "score": float(nws.score or 0.0),
            }
        )
    return sources


def search(
    query: str,
    k: Optional[int] = None,
    path_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve + rerank, NO generation. Returns ranked chunks with citations."""
    nodes = retrieve(query, top_k=k, path_prefix=path_prefix, tags=tags)
    return [
        {
            "path": n.node.metadata.get("path", ""),
            "title": n.node.metadata.get("title", ""),
            "heading_path": n.node.metadata.get("heading_path", ""),
            "text": n.node.text,
            "tags": n.node.metadata.get("tags", []),
            "score": float(n.score or 0.0),
        }
        for n in nodes
    ]


def query_notes(
    query: str,
    top_k: Optional[int] = None,
    path_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Full RAG: retrieve → rerank → synthesize. Returns answer + sources + raw contexts.

    This is the path Ragas evaluates (answer scored against question + contexts + ground_truth).
    """
    nodes = retrieve(query, top_k=top_k, path_prefix=path_prefix, tags=tags)
    if not nodes:
        return {
            "answer": "I don't have enough in the vault to answer that.",
            "sources": [],
            "contexts": [],
        }
    context = _context_block(nodes)
    prompt = _CITATION_INSTRUCTIONS.format(context=context, question=query)
    resp = get_generation_llm().invoke([HumanMessage(content=prompt)])
    answer = str(resp.content).strip()
    return {
        "answer": answer,
        "sources": format_sources(nodes),
        "contexts": [n.node.text for n in nodes],
    }