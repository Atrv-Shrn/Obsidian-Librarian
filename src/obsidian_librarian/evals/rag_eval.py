"""RAG evaluation — Ragas (non-LLM + LLM metrics) + LlamaIndex retrieval metrics. Offline.

Per SPEC.md (Delta 1), three models, three roles:

* ``nomic-embed-text`` (local Ollama) — pipeline embeddings **and** Ragas embedding-based metrics.
* ``deepseek-v4-pro:cloud`` — the pipeline's generator (what we're scoring).
* ``glm-5.2:cloud`` — the **Ragas LLM judge** (≠ generator → no self-preference bias).

So the judge is a ``ChatOllama`` pointed at Ollama Cloud with ``judge_model``, wrapped in
``LangchainLLMWrapper``; embeddings are a thin LangChain wrapper around our local nomic model.

We run **both** kinds of Ragas metrics:
- non-LLM (no judge call, deterministic): ``NonLLMContextRecall``,
  ``NonLLMContextPrecisionWithReference``, ``BleuScore``, ``RougeScore``, ``ExactMatch``,
  ``StringPresence``, ``SemanticSimilarity``.
- LLM (judge = glm-5.2:cloud): ``Faithfulness``, ``ResponseRelevancy``, ``LLMContextRecall``,
  ``LLMContextPrecisionWithReference``, ``AnswerCorrectness``, ``AspectCritic``.

Plus LlamaIndex retrieval metrics (hit-rate, MRR) computed directly from retrieved paths vs
``relevant_paths``. **No Langfuse here** — this is the pipeline, which never uses Langfuse.

Every metric runs in its own try/except so an API drift on one metric doesn't kill the suite;
the result dict records which metrics ran and which were skipped (deferred) with the error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from ..config import get_settings

log = logging.getLogger(__name__)

_GOLDEN = Path(__file__).parent / "golden_set.jsonl"

# Pass/fail thresholds (SPEC.md). A metric below threshold flags the run.
# ``context_recall`` / ``context_precision`` gate the Ragas LLM/non-LLM context metrics; the
# path-overlap retrieval variants are gated under their own ``retrieval_context_*`` keys so the
# two families are scored independently (see the merge in :func:`run`).
_THRESHOLDS = {
    "faithfulness": 0.7,
    "response_relevancy": 0.6,
    "answer_correctness": 0.6,
    "context_recall": 0.6,
    "context_precision": 0.6,
    "retrieval_context_recall": 0.6,
    "retrieval_context_precision": 0.6,
    "semantic_similarity": 0.7,
    "hit_rate": 0.6,
    "mrr": 0.6,
}


# --------------------------------------------------------------------------- wrappers


class _NomicLangchainEmbeddings:
    """LangChain-style embeddings backed by our local nomic model (for Ragas)."""

    def __init__(self) -> None:
        from ..rag.embeddings import get_dense_embed_model

        self._model = get_dense_embed_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model._get_text_embeddings(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(map(float, self._model._get_text_embedding(text)))


def _judge_llm():
    """glm-5.2:cloud judge via Ollama Cloud → LangchainLLMWrapper.

    SPEC: the judge is ``ChatOllama("glm-5.2:cloud")`` (≠ generator → no self-preference bias),
    reached over ``langchain-ollama`` with the cloud host in ``base_url`` and the API key as a
    bearer header in ``client_kwargs``.
    """
    from langchain_ollama import ChatOllama

    s = get_settings()
    api_key = s.ollama_api_key or "EMPTY"
    chat = ChatOllama(
        model=s.judge_model,
        base_url=s.ollama_base_url,
        temperature=0.0,
        client_kwargs={
            "headers": {"Authorization": f"Bearer {api_key}"},
            "timeout": s.llm_request_timeout,
        },
    )
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(chat)
    except Exception:  # older ragas
        return chat


def _embeddings_wrapper():
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(_NomicLangchainEmbeddings())
    except Exception:
        return _NomicLangchainEmbeddings()


# --------------------------------------------------------------------------- data


def _load_golden() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in _GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_pipeline(question: str) -> Dict[str, Any]:
    """Get the pipeline's answer + retrieved contexts for one golden question."""
    from ..rag.query_engine import query_notes

    res = query_notes(question)
    return res


def _path_matches(p: str, ref: str) -> bool:
    """Fuzzy path equality: exact, or one path is a path-component suffix of the other.

    Handles dir-prefix differences (``notes/bayes.md`` vs ``bayes.md``) the same way a bare
    ``str.endswith`` would, but compares *path components* instead of raw characters: a bare
    ``endswith`` over-matches (``"notes.md".endswith("es.md")`` is True even though ``es.md`` and
    ``notes.md`` are unrelated notes), which would inflate hit-rate/MRR/context-precision/recall
    whenever a reference path happens to be a character-tail of a retrieved path or vice versa.
    """
    if p == ref:
        return True
    pp = PurePosixPath(p).parts
    rp = PurePosixPath(ref).parts
    if len(pp) >= len(rp) and pp[-len(rp):] == rp:
        return True
    if len(rp) >= len(pp) and rp[-len(pp):] == pp:
        return True
    return False


def _retrieval_paths(question: str, relevant_paths: List[str]) -> Dict[str, Optional[float]]:
    """Hit-rate, MRR, and context precision/recall of retrieval vs ``relevant_paths``.

    SPEC line 290: retrieval metrics are hit-rate, MRR, **and** context precision/recall.
    Context precision/recall are computed as path-set overlap with the same fuzzy path match
    hit-rate uses: recall = |retrieved ∩ relevant| / |relevant| (of the notes we expected, how
    many surfaced); precision = |retrieved ∩ relevant| / |retrieved| (of what surfaced, how
    much was relevant).

    Returns ``None`` for every metric when there are no ``relevant_paths`` (a golden item with
    no ground-truth retrieval expectation). Such items are **skipped** from the averages rather
    than counted as 1.0 (which would inflate the mean) or 0.0 (which would penalize unfairly).
    """
    from ..rag.query_engine import search

    if not relevant_paths:
        return {"hit_rate": None, "mrr": None, "context_precision": None, "context_recall": None}
    hits = search(question, k=20)
    retrieved = [h["path"] for h in hits]
    hit = 0
    rr = 0.0
    for rank, p in enumerate(retrieved, 1):
        for ref in relevant_paths:
            if _path_matches(p, ref):
                hit = 1
                if rr == 0.0:
                    rr = 1.0 / rank
                break
    recall_count = sum(1 for ref in relevant_paths if any(_path_matches(p, ref) for p in retrieved))
    precision_count = sum(1 for p in retrieved if any(_path_matches(p, ref) for ref in relevant_paths))
    context_recall = recall_count / len(relevant_paths)
    context_precision = (precision_count / len(retrieved)) if retrieved else 0.0
    return {
        "hit_rate": float(hit),
        "mrr": rr,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


# --------------------------------------------------------------------------- metrics


def _build_ragas_dataset(samples: List[Dict[str, Any]]) -> Any:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    items = []
    for s in samples:
        items.append(
            SingleTurnSample(
                user_input=s["question"],
                response=s["_answer"],
                retrieved_contexts=s["_contexts"],
                reference=s["ground_truth"],
                reference_contexts=s.get("reference_contexts", []),
            )
        )
    return EvaluationDataset(items)


def _metric(metric_name: str, **kwargs: Any):
    """Construct a Ragas metric by class name.

    The first param is ``metric_name`` (not ``name``) on purpose: ``AspectCritic`` is built with
    ``_metric("AspectCritic", name="helpfulness", definition=...)``, and a parameter literally
    named ``name`` would collide with that ``name=`` kwarg — Python raises ``TypeError: got
    multiple values for argument 'name'`` at *call* time, before this function's body runs, so the
    internal try/except can't catch it. The TypeError would propagate to :func:`run`'s outer
    ``except`` and abort the whole Ragas block (the LLM loop mid-iteration, before the non-LLM
    ``evaluate`` even runs), leaving both ``ragas_non_llm`` and ``ragas_llm`` empty. Renaming the
    param keeps the ``name=`` kwarg flowing through to the metric constructor as ragas expects.
    """
    try:
        import importlib

        mod = importlib.import_module("ragas.metrics")
        cls = getattr(mod, metric_name)
        return cls(**kwargs)
    except Exception as e:
        log.warning("metric %s unavailable: %s", metric_name, e)
        return None


# AspectCritic needs an explicit name + definition at construction (ragas v0.4 enforces this).
_ASPECT_DEFINITION = (
    "Is the response helpful and grounded in the retrieved context, without fabricating "
    "note contents or straying from the user's question?"
)


def _coerce_score(v: Any) -> Optional[float]:
    """ragas v0.4 returns ``MetricResult`` objects (``.value``); v0.3 returns floats."""
    val = getattr(v, "value", v)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_ragas_scores(out: Any) -> Dict[str, float]:
    """EvaluationResult (v0.3 or v0.4) → ``{metric_name: mean_score}``.

    ragas v0.4's ``EvaluationResult`` has no ``.items()``; ``to_pandas()`` is the stable
    cross-version path. Each cell may be a ``MetricResult`` (v0.4) or a raw float (v0.3).
    """
    scores: Dict[str, float] = {}
    try:
        df = out.to_pandas()
    except Exception:
        # v0.3 Result was dict-like — fall back to iterating it directly.
        try:
            for k, v in dict(out).items():  # type: ignore[arg-type]
                if k == "description":
                    continue
                val = _coerce_score(v)
                if val is not None:
                    scores[k] = val
        except Exception:
            return {}
        return scores

    meta_cols = {
        "user_input",
        "reference",
        "response",
        "retrieved_contexts",
        "reference_contexts",
        "reference_answer",
    }
    for col in df.columns:
        if col in meta_cols:
            continue
        vals = [_coerce_score(v) for v in df[col].tolist()]
        vals = [v for v in vals if v is not None]
        if vals:
            scores[str(col)] = sum(vals) / len(vals)
    return scores


def run() -> Dict[str, Any]:
    """Run the full RAG eval suite. Returns a results dict (never raises on a single metric)."""
    rows = _load_golden()
    log.info("RAG eval over %d golden items", len(rows))

    # 1) Drive the pipeline once per question.
    samples: List[Dict[str, Any]] = []
    retrieval_scores: List[Dict[str, float]] = []
    for r in rows:
        try:
            res = _run_pipeline(r["question"])
            answer = res.get("answer", "")
            contexts = res.get("contexts", [])
        except Exception as e:
            log.warning("pipeline failed on %s: %s", r["id"], e)
            answer, contexts = "", []
        s = dict(r)
        s["_answer"] = answer
        s["_contexts"] = contexts
        samples.append(s)
        retrieval_scores.append(_retrieval_paths(r["question"], r.get("relevant_paths", [])))

    hit_rate_vals = [x["hit_rate"] for x in retrieval_scores if x["hit_rate"] is not None]
    mrr_vals = [x["mrr"] for x in retrieval_scores if x["mrr"] is not None]
    ctx_prec_vals = [x["context_precision"] for x in retrieval_scores if x["context_precision"] is not None]
    ctx_rec_vals = [x["context_recall"] for x in retrieval_scores if x["context_recall"] is not None]
    hit_rate = sum(hit_rate_vals) / max(1, len(hit_rate_vals))
    mrr = sum(mrr_vals) / max(1, len(mrr_vals))
    context_precision = sum(ctx_prec_vals) / max(1, len(ctx_prec_vals))
    context_recall = sum(ctx_rec_vals) / max(1, len(ctx_rec_vals))

    results: Dict[str, Any] = {
        "n": len(rows),
        "retrieval": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "context_precision": context_precision,
            "context_recall": context_recall,
        },
        "ragas_non_llm": {},
        "ragas_llm": {},
        "skipped": [],
    }

    # 2) Ragas — non-LLM + LLM metrics, each isolated.
    try:
        from ragas import evaluate

        dataset = _build_ragas_dataset(samples)
        judge = _judge_llm()
        emb = _embeddings_wrapper()

        non_llm_metrics = []
        for name in [
            "NonLLMContextRecall",
            "NonLLMContextPrecisionWithReference",
            "BleuScore",
            "RougeScore",
            "ExactMatch",
            "StringPresence",
            "SemanticSimilarity",
        ]:
            m = _metric(name)
            if m is not None:
                non_llm_metrics.append(m)
            else:
                results["skipped"].append(name)

        llm_metrics = []
        for name in [
            "Faithfulness",
            "ResponseRelevancy",
            "LLMContextRecall",
            "LLMContextPrecisionWithReference",
            "AnswerCorrectness",
            "AspectCritic",
        ]:
            if name == "AspectCritic":
                # AspectCritic requires name + definition at construction (v0.4).
                m = _metric(name, name="helpfulness", definition=_ASPECT_DEFINITION)
            else:
                m = _metric(name)
            if m is not None:
                llm_metrics.append(m)
            else:
                results["skipped"].append(name)

        if non_llm_metrics:
            try:
                out = evaluate(dataset=dataset, metrics=non_llm_metrics, embeddings=emb)
                results["ragas_non_llm"] = _extract_ragas_scores(out)
            except Exception as e:
                results["skipped"].append(f"non_llm_run: {e}")

        if llm_metrics:
            try:
                out = evaluate(dataset=dataset, metrics=llm_metrics, llm=judge, embeddings=emb)
                results["ragas_llm"] = _extract_ragas_scores(out)
            except Exception as e:
                results["skipped"].append(f"llm_run: {e}")
    except Exception as e:
        results["skipped"].append(f"ragas_import: {e}")

    # 3) Threshold pass/fail.
    # The Ragas LLM/non-LLM context metrics and our path-overlap retrieval context metrics both
    # emit columns named ``context_recall`` / ``context_precision``. Merging them under the same
    # key would let whichever family is written last silently clobber the other, so the gate
    # would test only one family and mask the other (e.g. a run with Ragas LLMContextRecall=0.3
    # but retrieval recall=0.8 would pass the >=0.6 gate on the retrieval value). Namespace the
    # retrieval-derived context metrics as ``retrieval_context_*`` so both families are gated
    # independently under their own thresholds.
    flat = {**results["ragas_non_llm"], **results["ragas_llm"],
            "hit_rate": hit_rate, "mrr": mrr,
            "retrieval_context_precision": context_precision,
            "retrieval_context_recall": context_recall}
    results["thresholds"] = _THRESHOLDS
    results["pass"] = {
        k: (flat[k] >= thr) for k, thr in _THRESHOLDS.items() if k in flat
    }
    results["overall_pass"] = all(results["pass"].values()) if results["pass"] else False
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=2, default=str))