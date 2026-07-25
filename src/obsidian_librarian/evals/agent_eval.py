"""Agent evaluation — golden tasks + Langfuse, with file/marker assertions for writes.

Unlike the RAG eval (offline, no Langfuse), the agent eval **does** use Langfuse when
``LANGFUSE_*`` keys are set: every golden task runs as a traced agent turn and we record
pass/fail as a Langfuse score. If Langfuse isn't configured, results are still computed and
returned — just not pushed anywhere.

Assertion strategy:
- **Read tasks**: the agent's reply must contain the expected citation/substring and must NOT
  contain the ``[PENDING_WRITE]`` marker. Q&A answer *correctness* is scored either by a
  ``glm-5.2:cloud`` judge or by **semantic match** (SPEC line 481) — we use semantic match
  (cosine similarity of nomic embeddings of reply vs ``ground_truth``), the offline-friendly
  branch SPEC permits, so eval doesn't require a live judge call. When a task carries a full
  ``ground_truth`` field the semantic score gates ``pass``; when it only has
  ``ground_truth_contains`` (a substring gate) the structural checks stand on their own.
- **Write tasks (propose-then-confirm)**: turn 1 must emit the marker with the expected
  ``op`` and target; a confirm turn ("yes") must lead to a write tool call (when Obsidian is
  available), and a decline turn ("no") must NOT execute the write.
- **Negative tasks**: the agent should say the vault doesn't contain the answer.

Actual vault file assertions are best-effort: writes go through the Obsidian plugin MCP, which
is only up while the user has Obsidian running. If ``obsidian_writes_available()`` is false at
eval time, write-execution assertions are recorded as ``skipped`` (deferred), not failed — the
marker/HITL behavior is still fully checked because that lives in the prompt, not the plugin.
"""

from __future__ import annotations

import asyncio
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from ..config import get_settings
from ..agent import graph as agent_graph
from ..agent.observability import (
    flush as langfuse_flush,
    get_langfuse_handler,
    langfuse_enabled,
    score_trace,
)

log = logging.getLogger(__name__)

_TASKS = Path(__file__).parent / "agent_tasks.jsonl"

# Substrings that identify an Obsidian *write* tool by name. The plugin exposes
# create/patch/append/delete (+ active-note + commands); reads come from our RAG-MCP
# (search_notes, query_notes, get_note, list_notes, get_backlinks, get_recent, reindex).
# A turn-2 read tool call must NOT count as "executed the write".
_WRITE_TOOL_HINTS = ("create", "patch", "append", "delete", "write", "put", "update", "rename")

# Cosine-similarity threshold for the semantic answer-correctness check (SPEC line 481). Mirrors
# the RAG eval's ``answer_correctness`` threshold rather than the stricter ``semantic_similarity``
# one — a paraphrased agent reply that's substantively correct should pass.
_SEMANTIC_THRESHOLD = 0.6


def _is_write_tool(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _WRITE_TOOL_HINTS)


def _semantic_similarity(reply: str, ground_truth: str) -> Optional[float]:
    """Cosine similarity of nomic embeddings of ``reply`` vs ``ground_truth`` (SPEC line 481).

    The offline-friendly branch of "glm-5.2 judge **or** semantic match": reuses the local
    nomic embedder already in the pipeline, so no live judge call is needed. Returns ``None``
    when the embedder is unavailable (e.g. heavy deps stubbed in unit tests, or fastembed not
    installed) so ``_check_read`` can skip the semantic gate rather than fail the whole eval —
    the structural substring/citation checks still run.
    """
    if not reply or not ground_truth:
        return None
    try:
        import math

        from ..rag.embeddings import get_dense_embed_model

        model = get_dense_embed_model()
        a = [float(x) for x in model._get_text_embedding(reply)]
        b = [float(x) for x in model._get_text_embedding(ground_truth)]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return None
        return dot / (na * nb)
    except Exception as e:  # embedder unavailable / stubbed → skip semantic gate, don't crash eval
        log.debug("semantic match skipped: %s", e)
        return None


def _norm(s: str) -> str:
    """Fold curly quotes/apostrophes to ASCII so ``don’t`` (U+2019) matches ``don't`` (U+0027).

    NFKC alone does NOT fold U+2018/U+2019 (they have no compatibility decomposition), so we
    translate the common typographic punctuation explicitly before checking signal words.
    """
    return (
        unicodedata.normalize("NFKC", s)
        .replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def _load_tasks() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in _TASKS.read_text(encoding="utf-8").splitlines() if l.strip()]


def _last_ai_text(result: Dict[str, Any]) -> str:
    ai = [m for m in result["messages"] if m.type == "ai"]
    return ai[-1].content if ai else ""


async def _invoke(messages: List, thread: str) -> Dict[str, Any]:
    return await agent_graph.ainvoke(messages, thread_id=thread)


def _check_read(task: Dict[str, Any], reply: str) -> Dict[str, Any]:
    s = get_settings()
    ok = True
    notes: List[str] = []
    out: Dict[str, Any] = {}
    if task.get("must_not_marker") and s.pending_write_marker in reply:
        ok = False
        notes.append("unexpected PENDING_WRITE marker on a read task")
    if task.get("expects_citation") and task["expects_citation"].lower() not in reply.lower():
        ok = False
        notes.append(f"missing citation: {task['expects_citation']}")
    if task.get("ground_truth_contains") and task["ground_truth_contains"].lower() not in reply.lower():
        if not task.get("expects_no_answer"):
            ok = False
        notes.append(f"missing expected substring: {task['ground_truth_contains']}")
    if task.get("expects_no_answer"):
        # Negative: the agent should signal the vault doesn't answer. Normalize (NFKC) so a
        # curly apostrophe in "don't" matches a straight one, and don't count the bare word
        # "vault" — it appears in almost every reply ("searching the vault…") and so trivially
        # passes, masking a failure to actually signal a missing answer.
        r = _norm(reply).lower()
        if not any(w in r for w in ("don't", "don't have", "no note", "not enough", "couldn't find", "no matching", "doesn't contain", "not in the vault")):
            ok = False
            notes.append("negative task: agent didn't signal missing answer")
    # Answer correctness (SPEC line 481): when a full ``ground_truth`` is present, score the
    # reply by semantic match (nomic cosine). This is the offline-friendly alternative to a
    # glm-5.2 judge call. If the embedder is unavailable the gate is skipped (noted) so the
    # structural checks above still decide ``pass`` on their own.
    gt = task.get("ground_truth")
    if gt:
        sim = _semantic_similarity(reply, str(gt))
        out["semantic_similarity"] = sim
        if sim is None:
            notes.append("semantic answer-correctness skipped (embedder unavailable)")
        elif sim < _SEMANTIC_THRESHOLD:
            ok = False
            notes.append(f"semantic answer-correctness below threshold: {sim:.3f} < {_SEMANTIC_THRESHOLD}")
    out["pass"] = ok
    out["notes"] = notes
    out["reply_excerpt"] = reply[:300]
    return out


def _check_marker(task: Dict[str, Any], reply: str) -> Dict[str, Any]:
    """Turn-1 check: did the agent propose with the right marker/op/target?"""
    s = get_settings()
    marker = s.pending_write_marker
    ok = marker in reply
    notes: List[str] = []
    if not ok:
        notes.append("missing PENDING_WRITE marker")
    if ok and task.get("marker_op"):
        if f"op={task['marker_op']}" not in reply:
            ok = False
            notes.append(f"marker op mismatch (want op={task['marker_op']})")
    if ok and task.get("marker_target_contains"):
        if task["marker_target_contains"].lower() not in reply.lower():
            ok = False
            notes.append("marker target mismatch")
    return {"pass": ok, "notes": notes, "reply_excerpt": reply[:300]}


async def _eval_task(task: Dict[str, Any]) -> Dict[str, Any]:
    s = get_settings()
    thread = f"eval-{task['id']}"
    kind = task.get("kind", "read")
    out: Dict[str, Any] = {"id": task["id"], "kind": kind}

    if kind == "read" or kind == "negative":
        result = await _invoke([HumanMessage(content=task["task"])], thread)
        out.update(_check_read(task, _last_ai_text(result)))
        out["_langfuse_trace_id"] = result.get("_langfuse_trace_id")
        return out

    # write: turn 1 = propose
    r1 = await _invoke([HumanMessage(content=task["task"])], thread)
    reply1 = _last_ai_text(r1)
    out["propose"] = _check_marker(task, reply1)

    # turn 2 = confirm or decline
    confirm = task.get("confirm_reply", "yes")
    history = r1["messages"] + [HumanMessage(content=confirm)]
    r2 = await _invoke(history, thread)
    reply2 = _last_ai_text(r2)
    # Only count tool calls from turn-2's OWN AI messages. ``r2["messages"]`` is the full
    # thread state (turn-1 messages + turn-2 messages), so naively scanning all of it would
    # re-count turn-1's read tools and taint both ``decline_honored`` and ``executed_write``.
    r1_ai_count = sum(1 for m in r1["messages"] if m.type == "ai")
    turn2_ai = [m for m in r2["messages"] if m.type == "ai"][r1_ai_count:]
    all_calls = [tc.get("name", "") for m in turn2_ai for tc in (m.tool_calls or [])]
    # Only WRITE tool calls count as "executed the write". A turn-2 read tool call (the agent
    # re-grounding before executing) must not satisfy ``executed_write``, and on a decline task
    # a read call must not flip ``decline_honored`` to False.
    write_calls = [c for c in all_calls if _is_write_tool(c)]
    wrote = bool(write_calls) and not task.get("expects_no_write", False)
    if task.get("expects_no_write"):
        out["decline_honored"] = not write_calls
        out["pass"] = bool(out["propose"]["pass"]) and out["decline_honored"]
    else:
        writes_avail = agent_graph.obsidian_writes_available()
        if writes_avail:
            out["executed_write"] = bool(write_calls)
            out["pass"] = bool(out["propose"]["pass"]) and out["executed_write"]
        else:
            out["executed_write"] = None
            out["skipped"] = "Obsidian plugin not available — write execution deferred"
            # Marker/HITL behavior is still validated:
            out["pass"] = bool(out["propose"]["pass"])
    out["reply2_excerpt"] = reply2[:300]
    out["tool_calls_turn2"] = all_calls
    out["write_calls_turn2"] = write_calls
    # Score against the turn-2 trace (the confirm/decline turn being asserted on).
    out["_langfuse_trace_id"] = r2.get("_langfuse_trace_id")
    return out


def _score_langfuse(task_id: str, passed: bool, trace_id: Optional[str]) -> bool:
    """Attach the task's pass/fail to its Langfuse trace. Returns True if submitted.

    Delegates to :func:`observability.score_trace`, which uses the v3 ``create_score`` API on
    the shared client singleton. The previous implementation constructed a bare ``Langfuse()``
    (no credentials — they live on the singleton) and called ``lf.score(...)``, a v2 method that
    does not exist in v3; the resulting ``AttributeError`` was swallowed into a warning, so no
    score ever reached Langfuse. It also passed ``trace_id or ""``, which would have orphaned
    the score even if the call had worked.
    """
    return score_trace(
        name="agent_task_pass",
        value=1.0 if passed else 0.0,
        trace_id=trace_id,
        comment=task_id,
    )


def _run_async() -> Dict[str, Any]:
    tasks = _load_tasks()
    log.info("agent eval over %d tasks (langfuse=%s)", len(tasks), langfuse_enabled())

    async def _all():
        return await asyncio.gather(*[_eval_task(t) for t in tasks])

    results = asyncio.run(_all())
    passed = sum(1 for r in results if r.get("pass"))
    scored = 0
    for r in results:
        # Attach the score to the same trace the agent run produced, so it isn't orphaned.
        if _score_langfuse(r["id"], bool(r.get("pass")), r.get("_langfuse_trace_id")):
            scored += 1
    # The v3 SDK ships spans/scores asynchronously over OTel; without an explicit flush this
    # short-lived process can exit before anything is sent and the scores never appear.
    langfuse_flush()

    return {
        "n": len(tasks),
        "passed": passed,
        "failed": len(tasks) - passed,
        "pass_rate": passed / max(1, len(tasks)),
        "overall_pass": passed == len(tasks),
        # Surfaced so a silent telemetry regression is visible in the eval output rather than
        # only in logs: langfuse on + scored == 0 means scoring is broken again.
        "langfuse_enabled": langfuse_enabled(),
        "langfuse_scored": scored,
        "results": results,
    }


def run() -> Dict[str, Any]:
    try:
        return _run_async()
    except Exception as e:
        return {"error": str(e), "overall_pass": False}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2, default=str))