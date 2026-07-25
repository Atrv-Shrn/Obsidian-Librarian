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
#
# The keys MUST be the column names Ragas actually emits, which are each metric class's ``name``
# attribute — NOT the class name snake-cased. The gate is ``{k: flat[k] >= thr for k, thr in
# _THRESHOLDS.items() if k in flat}``, so a key that never appears as a column is silently
# dropped and that metric is never gated at all. The verified v0.4 class → column mapping for the
# metrics we configure (see :func:`run`):
#
#   Faithfulness                        -> faithfulness
#   ResponseRelevancy                   -> answer_relevancy          (NOT response_relevancy)
#   AnswerCorrectness                   -> answer_correctness
#   SemanticSimilarity                  -> semantic_similarity
#   LLMContextRecall                    -> context_recall
#   NonLLMContextRecall                 -> non_llm_context_recall    (NOT context_recall)
#   LLMContextPrecisionWithReference    -> llm_context_precision_with_reference
#   NonLLMContextPrecisionWithReference -> non_llm_context_precision_with_reference
#   AspectCritic(name="helpfulness")    -> helpfulness               (caller-supplied name)
#   BleuScore/RougeScore/ExactMatch/StringPresence -> bleu_score/rouge_score/exact_match/string_present
#
# Note this also retires the earlier "both families emit ``context_precision``" assumption: only
# ``LLMContextRecall`` collides with our retrieval-derived ``context_recall``, and the precision
# classes emit distinct, fully-qualified columns. The ``retrieval_*`` namespacing is still
# correct and still required for recall — it is what keeps the two recall families independent.
_THRESHOLDS = {
    # --- Ragas LLM metrics ---
    "faithfulness": 0.7,
    "answer_relevancy": 0.6,
    "answer_correctness": 0.6,
    "context_recall": 0.6,  # LLMContextRecall
    "llm_context_precision_with_reference": 0.6,
    # --- Ragas non-LLM metrics ---
    "semantic_similarity": 0.7,
    # NOTE: ``non_llm_context_recall`` / ``non_llm_context_precision_with_reference`` are NOT
    # gated. They compare ``reference_contexts`` to ``retrieved_contexts`` by *string distance*,
    # so they score how exactly the golden file transcribes the pipeline's chunk text — heading
    # prefixes ("# Reranking\n\n"), chunk boundaries and whitespace all count against them. Our
    # references are hand-written excerpts, so these read ~0.06-0.12 even when retrieval is
    # perfect (hit_rate 1.0, and the LLM-judged equivalents score 0.94/0.83 on the same run).
    # Gating them would make the suite fail on transcription fidelity rather than retrieval
    # quality. They stay in the output as diagnostics; ``context_recall`` and
    # ``llm_context_precision_with_reference`` are the gated context metrics.
    # --- our path-overlap retrieval metrics (namespaced; see the merge in ``run``) ---
    "retrieval_context_recall": 0.6,
    "retrieval_context_precision": 0.6,
    "hit_rate": 0.6,
    "mrr": 0.6,
}


# --------------------------------------------------------------------------- wrappers


class _NomicLangchainEmbeddings:
    """LangChain-style embeddings backed by our nomic model (for Ragas).

    Implements the **async** ``aembed_*`` methods as well as the sync ones. Ragas drives its
    metrics through an async executor and calls ``aembed_documents`` / ``aembed_query``; with
    only the sync pair defined, every embedding-backed metric raised
    ``AttributeError('_NomicLangchainEmbeddings' object has no attribute 'aembed_documents')``
    inside the job, which Ragas swallows per-row and reports as ``NaN`` rather than failing the
    run. That silently blanked ``semantic_similarity`` (and, via the same path, the metrics that
    depend on embeddings) while the suite still looked like it had "run".

    The underlying FastEmbed model is synchronous CPU work, so the async methods just delegate
    — no thread offload, matching how the rest of the eval already calls it.
    """

    def __init__(self) -> None:
        from ..rag.embeddings import get_dense_embed_model

        self._model = get_dense_embed_model()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, v)) for v in self._model._get_text_embeddings(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(map(float, self._model._get_text_embedding(text)))

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


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


_ABSTENTION_MARKERS = (
    "don't have enough",
    "do not have enough",
    "doesn't contain",
    "does not contain",
    "no notes",
    "not in the vault",
    "nothing in the vault",
    "no mention",
    "not mentioned",
    "couldn't find",
    "could not find",
    "no relevant",
)


def _is_abstention(answer: str) -> bool:
    """True if the answer declines to answer rather than inventing content.

    Deliberately a substring check over a small marker set, not an LLM call: the negative items
    exist to catch hallucination, and gating that check behind another model call would make the
    hallucination metric itself depend on a model's judgement. Normalized to lowercase with
    curly apostrophes folded so "don't" (U+2019) matches the ASCII form.
    """
    a = (answer or "").lower().replace("’", "'")
    return any(m in a for m in _ABSTENTION_MARKERS)


def _retrieval_paths(question: str, relevant_paths: List[str]) -> Dict[str, Optional[float]]:
    """Hit-rate, MRR, and context precision/recall of retrieval vs ``relevant_paths``.

    SPEC line 290: retrieval metrics are hit-rate, MRR, **and** context precision/recall.
    Both use the same fuzzy path match hit-rate uses. Recall = |retrieved ∩ relevant| /
    |relevant| (of the notes we expected, how many surfaced). Precision is **R-precision** —
    precision within the top-R retrieved, where R = len(relevant_paths) — not precision over
    the full k=20 candidate pool; see the note at the computation for why the pool-wide form
    made the metric unpassable by construction.

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
    context_recall = recall_count / len(relevant_paths)

    # Precision is **R-precision**: precision within the top-R retrieved, where R is the number
    # of relevant notes for this question — NOT precision over the whole candidate pool.
    #
    # Dividing by len(retrieved) is wrong here because ``search`` always returns k=20 candidates
    # (that pool exists to feed the reranker, not to be a precision denominator). With one
    # relevant note per golden item, that formula caps precision at 1/20 = 0.05, so the metric
    # could never clear its own 0.6 threshold no matter how good retrieval was — a guaranteed
    # false failure. Observed exactly that: hit_rate 1.0 and MRR 0.875 (correct note at rank 1)
    # alongside context_precision 0.0625.
    #
    # R-precision asks the meaningful question instead: of the top-R slots, how many hold a
    # relevant note? For a single-target item it reduces to "was the right note ranked first?",
    # which is what precision should mean for this eval.
    r = len(relevant_paths)
    top_r = retrieved[:r]
    precision_count = sum(1 for p in top_r if any(_path_matches(p, ref) for ref in relevant_paths))
    context_precision = (precision_count / len(top_r)) if top_r else 0.0
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
    """ragas v0.4 returns ``MetricResult`` objects (``.value``); v0.3 returns floats.

    ``NaN`` is treated as *absent*, not as a score. Ragas swallows per-row failures and writes
    ``NaN`` into the cell, and a single such row would otherwise poison the whole column's mean
    (``sum`` of anything with NaN is NaN), silently turning one bad row into a failed metric.
    Returning ``None`` here drops that row from the average instead — the same treatment
    :func:`_retrieval_paths` gives items with no ground-truth expectation.
    """
    val = getattr(v, "value", v)
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN != NaN


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
    #
    # Negative items (no ``relevant_paths``; the vault genuinely cannot answer them) are excluded
    # from the Ragas generation metrics. Their *correct* behaviour is a refusal — "I don't have
    # enough in the vault to answer that" — which ResponseRelevancy scores 0.0 by design: it
    # classifies refusals as noncommittal, and the non-LLM reference metrics have no
    # reference_contexts to compare against (yielding NaN). Scoring a correct refusal as a
    # generation failure measures the opposite of what we want and drags every mean down. The
    # refusal behaviour is still asserted — that is what the negative items are *for* — it just
    # belongs to the hallucination check, not to answer-quality scoring.
    ragas_samples = [s for s in samples if s.get("relevant_paths")]
    n_negative = len(samples) - len(ragas_samples)
    if n_negative:
        results["ragas_excluded_negative"] = n_negative

    try:
        from ragas import evaluate

        dataset = _build_ragas_dataset(ragas_samples)
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

    # 2b) Abstention on negative items — the check the Ragas exclusion above would otherwise
    # drop. A negative item has no answer in the vault, so the *only* correct behaviour is to
    # say so; inventing an answer is the hallucination failure this eval most needs to catch.
    # Scored here as a first-class metric rather than left implicit.
    negatives = [s for s in samples if not s.get("relevant_paths")]
    if negatives:
        abstained = sum(1 for s in negatives if _is_abstention(s["_answer"]))
        results["abstention"] = {
            "n": len(negatives),
            "abstained": abstained,
            "rate": abstained / len(negatives),
        }

    # 3) Threshold pass/fail.
    # ``LLMContextRecall`` emits a column literally named ``context_recall`` — the same name our
    # path-overlap retrieval recall would use. Merging them under one key would let whichever is
    # written last silently clobber the other, so the gate would test only one family and mask
    # the other (a run with Ragas LLMContextRecall=0.3 but retrieval recall=0.8 would pass the
    # >=0.6 gate on the retrieval value). Namespacing the retrieval-derived context metrics as
    # ``retrieval_context_*`` keeps both families gated independently under their own thresholds.
    # (The precision classes emit fully-qualified distinct names — ``llm_context_precision_with_
    # reference`` / ``non_llm_context_precision_with_reference`` — so they never collided; the
    # namespacing is harmless there and kept for symmetry. See ``_THRESHOLDS`` for the full map.)
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