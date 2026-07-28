"""Command-line interface — ``librarian`` entrypoint (see ``pyproject.toml`` scripts).

Subcommands map to the two halves of the system:

* ``librarian sync`` — index the vault (M1/M7 path; also what APScheduler calls).
* ``librarian search`` / ``librarian query`` — hit the RAG pipeline directly (no agent).
* ``librarian chat`` — talk to the LangGraph agent over MCP tools (M4).
* ``librarian rag-serve`` — run the read-only RAG-MCP streamable-http server (M3).
* ``librarian api-serve`` — run the OpenAI-compatible FastAPI endpoint (M6).
* ``librarian eval`` — run the eval suites (M8).
"""

from __future__ import annotations

import asyncio
import json as _json
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from .config import get_settings

app = typer.Typer(add_completion=False, help="Obsidian-Librarian — personal AI librarian for Obsidian.")
console = Console()


@app.command()
def sync() -> None:
    """Index the vault (parse→split→embed→upsert). Idempotent; diff-driven by watermarks."""
    from .rag.ingest import sync_vault

    report = sync_vault()
    console.print(Panel.fit(
        f"added={report.added}  modified={report.modified}  deleted={report.deleted}  "
        f"skipped_dup={report.skipped_dup}  errors={len(report.errors)}",
        title="sync report",
    ))
    for e in report.errors:
        console.print(f"[red]error[/red] {e}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: Optional[int] = typer.Option(None, "--k", "-k", help="Top-K chunks after rerank."),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Restrict to paths under this prefix."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Restrict to notes with these tags."),
) -> None:
    """Retrieve + rerank only (no generation). Prints ranked chunks."""
    from .rag.query_engine import search as _search

    hits = _search(query, k=k, path_prefix=prefix, tags=tags)
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return
    for i, h in enumerate(hits, 1):
        console.print(Panel(
            f"[bold]{h['title']}[/bold]  [dim]{h['path']}[/dim]\n"
            f"[dim]heading: {h.get('heading_path','')} · score {h['score']:.3f}[/dim]\n\n"
            f"{h['text'][:600]}",
            title=f"#{i}",
        ))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer from the vault."),
    top_k: Optional[int] = typer.Option(None, "--k", "-k", help="Top-K chunks passed to the synthesizer."),
    prefix: Optional[str] = typer.Option(None, "--prefix", help="Restrict to paths under this prefix."),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Restrict to notes with these tags."),
) -> None:
    """Full RAG: retrieve → rerank → synthesize a grounded answer with [[citations]]."""
    from .rag.query_engine import query_notes

    res = query_notes(question, top_k=top_k, path_prefix=prefix, tags=tags)
    console.print(Panel(res["answer"], title="answer"))
    if res["sources"]:
        console.print("[bold]sources[/bold]")
        for s in res["sources"]:
            console.print(f"  · [[{s['title']}]]  [dim]{s['path']}[/dim]  score={s['score']:.3f}")


@app.command()
def chat() -> None:
    """Interactive chat with the LangGraph agent (reads via RAG-MCP, writes via Obsidian MCP)."""
    from langchain_core.messages import HumanMessage

    from .agent import graph as agent_graph

    async def _run() -> None:
        s = get_settings()
        thread = s.api_default_thread_id
        console.print(Panel.fit(
            f"Obsidian Librarian chat · model={s.generation_model} · confirm={s.write_confirm}\n"
            "Ctrl-D / empty line to exit. Type 'yes' to confirm a pending write.",
            title="librarian chat",
        ))
        history: list = []
        while True:
            try:
                line = console.input("[bold]you>[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye.[/dim]")
                break
            if not line:
                break
            history.append(HumanMessage(content=line))
            console.print("[bold]librarian>[/bold] ", end="")
            try:
                result = await agent_graph.ainvoke(history, thread_id=thread)
            except Exception as e:
                # Contain per-turn failures: one bad turn (tool/MCP/LLM blip) must not kill the
                # whole REPL. Drop this turn's just-added user message so history stays
                # consistent, surface the error, and keep the loop alive for the next line.
                console.print(f"[red]turn failed: {e}[/red]")
                history.pop()
                continue
            ai_msgs = [m for m in result["messages"] if m.type == "ai"]
            reply = ai_msgs[-1].content if ai_msgs else ""
            console.print(reply)
            history = result["messages"]

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


@app.command()
def rag_serve() -> None:
    """Run the read-only RAG-MCP streamable-http server (:8765/mcp)."""
    from .mcp.rag_server import main

    main()


@app.command()
def api_serve() -> None:
    """Run the OpenAI-compatible FastAPI endpoint (:8000/v1)."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "obsidian_librarian.api.openai_compat:app",
        host=s.api_host,
        port=s.api_port,
        log_level="info",
    )


@app.command("eval")
def eval_cmd(
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Run RAG evals (Ragas + retrieval)."),
    agent: bool = typer.Option(False, "--agent/--no-agent", help="Run agent evals (golden + Langfuse)."),
) -> None:
    """Run the eval suites. Defaults to RAG only (offline, no Langfuse)."""
    results: dict = {}
    if rag:
        from .evals.rag_eval import run as run_rag

        results["rag"] = run_rag()
    if agent:
        from .evals.agent_eval import run as run_agent

        results["agent"] = run_agent()
    console.print(Syntax(_json.dumps(results, indent=2, default=str), "json", theme="ansi_dark"))


@app.command()
def info() -> None:
    """Print resolved settings (never prints secret values)."""
    s = get_settings()
    redacted = {
        "vault_path": str(s.vault_path),
        "data_path": str(s.data_path),
        "embed_model": s.embed_model,
        "generation_model": s.generation_model,
        "judge_model": s.judge_model,
        "ollama_base_url": s.ollama_base_url,
        "ollama_api_key": "***" if s.ollama_api_key else None,
        "qdrant_url": s.qdrant_url,
        "redis_url": s.redis_url,
        "rag_mcp_endpoint": s.rag_mcp_endpoint,
        "obsidian_mcp_url": s.obsidian_mcp_url,
        "obsidian_api_key": "***" if s.obsidian_api_key else None,
        "write_confirm": s.write_confirm,
        "langfuse": bool(s.langfuse_public_key and s.langfuse_secret_key),
    }
    console.print(Syntax(_json.dumps(redacted, indent=2), "json", theme="ansi_dark"))


if __name__ == "__main__":  # pragma: no cover
    app()