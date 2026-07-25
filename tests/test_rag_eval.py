"""Tests for the pure helpers in the RAG eval harness.

Covers score coercion, ragas result extraction (pandas path + dict fallback + total failure),
the threshold table, and retrieval hit-rate/MRR with the query engine stubbed.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from obsidian_librarian.evals import rag_eval
from obsidian_librarian.evals.rag_eval import (
    _THRESHOLDS,
    _coerce_score,
    _extract_ragas_scores,
    _retrieval_paths,
)


# --------------------------------------------------------------------------- _coerce_score


def test_coerce_score_float():
    assert _coerce_score(0.75) == 0.75


def test_coerce_score_metric_result_value():
    # ragas v0.4 MetricResult exposes score via .value
    assert _coerce_score(SimpleNamespace(value=0.9)) == 0.9


def test_coerce_score_non_numeric_returns_none():
    assert _coerce_score("not a number") is None
    assert _coerce_score(None) is None


def test_coerce_score_treats_nan_as_absent():
    """A single NaN row must not poison the whole column's mean.

    Ragas swallows per-row failures and writes NaN into the cell. Since ``sum`` of anything
    containing NaN is NaN, one bad row silently turned an otherwise-passing metric into a
    failure — observed as answer_relevancy/context_precision reporting NaN across a run where
    most rows scored fine. NaN must drop out of the average, not propagate through it.
    """
    assert _coerce_score(float("nan")) is None
    assert _coerce_score(SimpleNamespace(value=float("nan"))) is None
    # Real scores still pass through untouched.
    assert _coerce_score(0.0) == 0.0
    assert _coerce_score(0.85) == 0.85


def test_extract_scores_skips_nan_rows_in_the_mean():
    out = SimpleNamespace(
        to_pandas=lambda: _FakeDF({"faithfulness": [0.8, float("nan"), 1.0]})
    )
    # Mean over the two real values (0.9), not NaN.
    assert _extract_ragas_scores(out) == {"faithfulness": 0.9}


# --------------------------------------------------------------------------- abstention


def test_is_abstention_detects_refusals_not_answers():
    """Negative golden items are excluded from Ragas generation metrics (a correct refusal
    scores 0.0 on ResponseRelevancy by design), so abstention is asserted directly instead.
    This is the hallucination check — it must not be fooled by a confident wrong answer."""
    assert rag_eval._is_abstention(
        "I don't have enough in the vault to answer that. The excerpts do not mention Mamba.")
    assert rag_eval._is_abstention("Your vault doesn't contain notes on state-space models.")
    assert rag_eval._is_abstention("I couldn't find any relevant notes.")
    # Curly apostrophe must fold to ASCII.
    assert rag_eval._is_abstention("I don’t have enough in the vault to answer that.")
    # A real answer is NOT an abstention.
    assert not rag_eval._is_abstention(
        "Mamba is a state-space model that replaces attention with a selective scan.")
    assert not rag_eval._is_abstention("")  # empty is not a refusal either


# --------------------------------------------------------------------------- _extract_ragas_scores


class _FakeDF:
    """Pandas-ish: ``.columns`` + ``__getitem__(col).tolist()``."""

    def __init__(self, columns: dict):
        self.columns = list(columns.keys())
        self._data = columns

    def __getitem__(self, col):
        return SimpleNamespace(tolist=lambda c=col: self._data[c])


def test_extract_scores_pandas_path_averages_metric_columns():
    out = SimpleNamespace(
        to_pandas=lambda: _FakeDF(
            {
                "user_input": ["q"],            # meta → skipped
                "response": ["r"],              # meta → skipped
                "faithfulness": [SimpleNamespace(value=0.8), SimpleNamespace(value=0.6)],
                "answer_correctness": [0.5, 0.7],
            }
        )
    )
    scores = _extract_ragas_scores(out)
    assert scores == {"faithfulness": 0.7, "answer_correctness": 0.6}


def test_extract_scores_dict_fallback_when_no_to_pandas():
    # v0.3 dict-like result; "description" is skipped.
    out = {"description": "run", "faithfulness": 0.7, "response_relevancy": SimpleNamespace(value=0.5)}
    scores = _extract_ragas_scores(out)
    assert scores == {"faithfulness": 0.7, "response_relevancy": 0.5}


def test_extract_scores_total_failure_returns_empty():
    class _Bad:
        def to_pandas(self):
            raise RuntimeError("boom")

        def __iter__(self):
            raise RuntimeError("also boom")

    # dict(_Bad()) raises → returns {}
    assert _extract_ragas_scores(_Bad()) == {}


# --------------------------------------------------------------------------- thresholds


def test_thresholds_shape():
    assert _THRESHOLDS["faithfulness"] == 0.7
    assert _THRESHOLDS["answer_relevancy"] == 0.6
    assert _THRESHOLDS["hit_rate"] == 0.6
    assert _THRESHOLDS["mrr"] == 0.6
    # No threshold accidentally below 0 or above 1.
    for k, v in _THRESHOLDS.items():
        assert 0.0 < v <= 1.0, k


def test_threshold_keys_match_real_ragas_column_names():
    """Every Ragas threshold key must be a column Ragas actually emits.

    The gate is ``{k: ... for k, thr in _THRESHOLDS.items() if k in flat}``, so a key that never
    appears as a column is silently skipped and that metric goes **ungated** — a
    below-threshold score would not flag the run. Ragas column names are each metric class's
    ``name`` attribute, which is NOT the class name snake-cased: ``ResponseRelevancy`` emits
    ``answer_relevancy``, ``NonLLMContextRecall`` emits ``non_llm_context_recall``, and the
    precision classes emit fully-qualified ``*_with_reference`` names. This pins the mapping so
    a future rename or metric swap fails loudly here instead of silently disabling a gate.
    """
    # Verified against ragas 0.4.x for exactly the metric set ``run()`` configures.
    ragas_columns = {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "semantic_similarity",
        "context_recall",
        "non_llm_context_recall",
        "llm_context_precision_with_reference",
        "non_llm_context_precision_with_reference",
        "bleu_score",
        "rouge_score",
        "exact_match",
        "string_present",
        "helpfulness",
    }
    # Keys we compute ourselves rather than getting from Ragas.
    ours = {"hit_rate", "mrr", "retrieval_context_recall", "retrieval_context_precision"}

    unknown = set(_THRESHOLDS) - ragas_columns - ours
    assert not unknown, f"threshold keys that match no real Ragas column (never gated): {unknown}"

    # The specific regressions that motivated this test.
    assert "response_relevancy" not in _THRESHOLDS  # ResponseRelevancy emits answer_relevancy
    assert "context_precision" not in _THRESHOLDS  # no metric emits a bare context_precision
    assert "answer_relevancy" in _THRESHOLDS

    # The string-distance context metrics are deliberately NOT gated: they score how exactly
    # the golden file transcribes the pipeline's chunk text (heading prefixes, chunk
    # boundaries, whitespace), not retrieval quality, and read ~0.06-0.12 on a run where
    # retrieval is perfect. The LLM-judged equivalents below are the real gates.
    assert "non_llm_context_recall" not in _THRESHOLDS
    assert "non_llm_context_precision_with_reference" not in _THRESHOLDS
    assert "context_recall" in _THRESHOLDS
    assert "llm_context_precision_with_reference" in _THRESHOLDS


# --------------------------------------------------------------------------- _retrieval_paths


def _install_query_engine(monkeypatch, hits):
    fake = types.ModuleType("obsidian_librarian.rag.query_engine")
    fake.search = lambda question, k=20: hits
    monkeypatch.setitem(sys.modules, "obsidian_librarian.rag.query_engine", fake)


def test_retrieval_paths_no_ref_returns_none_both():
    assert _retrieval_paths("q", []) == {
        "hit_rate": None,
        "mrr": None,
        "context_precision": None,
        "context_recall": None,
    }


def test_retrieval_paths_hit_at_rank1(monkeypatch):
    _install_query_engine(monkeypatch, [{"path": "notes/a.md"}, {"path": "notes/b.md"}])
    out = _retrieval_paths("q", ["notes/a.md"])
    assert out["hit_rate"] == 1.0
    assert out["mrr"] == 1.0
    # 1 of 1 relevant surfaced → recall 1.0. Precision is R-precision with R=1: the top-1
    # slot holds the relevant note → 1.0. (Pool-wide would have said 1/2 = 0.5, which
    # penalizes retrieval for returning extra reranker candidates it is supposed to return.)
    assert out["context_recall"] == 1.0
    assert out["context_precision"] == 1.0


def test_retrieval_paths_suffix_match(monkeypatch):
    # ref "a.md" matches retrieved "notes/a.md" via endswith.
    _install_query_engine(monkeypatch, [{"path": "other.md"}, {"path": "notes/a.md"}])
    out = _retrieval_paths("q", ["a.md"])
    assert out["hit_rate"] == 1.0
    assert out["mrr"] == 0.5  # rank 2
    assert out["context_recall"] == 1.0
    # R-precision with R=1 looks only at the top-1 slot, which holds the irrelevant
    # "other.md" here — so precision is 0 even though the target was found at rank 2.
    assert out["context_precision"] == 0.0


def test_retrieval_paths_precision_is_r_precision_not_pool_wide(monkeypatch):
    """Precision must be computed over the top-R hits, not the whole k=20 candidate pool.

    ``search`` always returns 20 candidates to feed the reranker. Dividing the relevant count
    by 20 caps precision at 1/20 = 0.05 for a single-target golden item, so the metric could
    never clear its own 0.6 threshold however good retrieval was — a false failure by
    construction. R-precision (precision within the top-R, R = number of relevant notes)
    reduces to "was the right note ranked first?" for single-target items.
    """
    # Correct note at rank 1, followed by 19 irrelevant ones.
    hits = [{"path": "notes/target.md"}] + [{"path": f"notes/other{i}.md"} for i in range(19)]
    _install_query_engine(monkeypatch, hits)
    out = _retrieval_paths("q", ["notes/target.md"])
    assert out["hit_rate"] == 1.0
    assert out["mrr"] == 1.0
    assert out["context_recall"] == 1.0
    # Pool-wide would give 1/20 = 0.05 and fail the 0.6 gate; R-precision gives 1/1.
    assert out["context_precision"] == 1.0

    # And a genuine miss at rank 1 must still score 0 precision even though it's retrieved.
    hits2 = [{"path": "notes/wrong.md"}, {"path": "notes/target.md"}]
    _install_query_engine(monkeypatch, hits2)
    out2 = _retrieval_paths("q", ["notes/target.md"])
    assert out2["context_precision"] == 0.0  # top-1 slot held the wrong note
    assert out2["context_recall"] == 1.0     # but it was still found
    assert out2["mrr"] == 0.5


def test_retrieval_paths_no_hit(monkeypatch):
    _install_query_engine(monkeypatch, [{"path": "unrelated.md"}])
    out = _retrieval_paths("q", ["notes/missing.md"])
    assert out["hit_rate"] == 0.0
    assert out["mrr"] == 0.0
    assert out["context_recall"] == 0.0
    assert out["context_precision"] == 0.0


# ------------------------------------------------------------- embeddings wrapper (async)


def test_nomic_langchain_embeddings_exposes_async_methods():
    """Ragas drives metrics through an async executor and calls ``aembed_documents`` /
    ``aembed_query``. With only the sync pair defined, every embedding-backed metric raised
    ``AttributeError`` *inside the job*, which Ragas swallows per-row and reports as ``NaN``
    — so semantic_similarity and answer_relevancy silently blanked while the run still looked
    like it had succeeded. Pin the async surface so that can't regress unnoticed.
    """
    import asyncio, inspect

    cls = rag_eval._NomicLangchainEmbeddings
    for name in ("embed_documents", "embed_query", "aembed_documents", "aembed_query"):
        assert hasattr(cls, name), f"missing {name}"
    assert inspect.iscoroutinefunction(cls.aembed_documents)
    assert inspect.iscoroutinefunction(cls.aembed_query)

    # The async methods must return the same vectors as the sync ones, not stubs.
    inst = cls.__new__(cls)  # bypass __init__ (it loads the real model)
    inst._model = SimpleNamespace(
        _get_text_embeddings=lambda ts: [[0.1, 0.2] for _ in ts],
        _get_text_embedding=lambda t: [0.3, 0.4],
    )
    assert inst.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert inst.embed_query("q") == [0.3, 0.4]
    assert asyncio.run(inst.aembed_documents(["a", "b"])) == [[0.1, 0.2], [0.1, 0.2]]
    assert asyncio.run(inst.aembed_query("q")) == [0.3, 0.4]


# --------------------------------------------------------------------------- _path_matches


def test_path_matches_component_aligned_no_false_suffix():
    """A bare ``str.endswith`` would match ``"notes.md"`` against ``"es.md"`` (a character-tail),
    inflating every retrieval metric. Component alignment must not: ``es.md`` and ``notes.md``
    are unrelated notes that share only a string suffix, not a path-component suffix."""
    assert rag_eval._path_matches("notes.md", "es.md") is False
    assert rag_eval._path_matches("es.md", "notes.md") is False
    assert rag_eval._path_matches("datetime.md", "time.md") is False
    # A real path-component suffix still matches (dir-prefixed vs bare), both directions.
    assert rag_eval._path_matches("notes/bayes.md", "bayes.md") is True
    assert rag_eval._path_matches("bayes.md", "notes/bayes.md") is True
    # Exact match.
    assert rag_eval._path_matches("a/b.md", "a/b.md") is True
    # Same final component but different dir does NOT match.
    assert rag_eval._path_matches("x/b.md", "y/b.md") is False


# --------------------------------------------------------------------------- run() threshold gate


def test_run_threshold_gate_gates_ragas_and_retrieval_context_independently(monkeypatch):
    """The Ragas LLM/non-LLM context metrics and our path-overlap retrieval context metrics both
    emit a column named ``context_recall`` / ``context_precision``. The gate must score them
    independently — if the retrieval value clobbered the Ragas one in the merge, a failing Ragas
    context metric (0.3 < 0.6) would be masked by a strong retrieval overlap (1.0) and the run
    would wrongly pass. We force Ragas ``context_recall``=0.3 and retrieval recall=1.0 and assert
    the two families are gated under separate keys (no clobber)."""
    _install_query_engine(monkeypatch, [{"path": "notes/a.md"}])

    monkeypatch.setattr(
        rag_eval, "_load_golden",
        lambda: [{"id": "g1", "question": "q", "ground_truth": "a", "relevant_paths": ["notes/a.md"]}],
    )
    monkeypatch.setattr(rag_eval, "_run_pipeline", lambda question: {"answer": "ans", "contexts": ["ctx"]})
    # Stub the heavy ragas helpers so no model is loaded / no network is hit.
    monkeypatch.setattr(rag_eval, "_judge_llm", lambda: object())
    monkeypatch.setattr(rag_eval, "_embeddings_wrapper", lambda: object())
    monkeypatch.setattr(
        rag_eval, "_extract_ragas_scores",
        lambda out: {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9,
            "answer_correctness": 0.9,
            "semantic_similarity": 0.9,
            # LLMContextRecall's real column name — the one that collides with our retrieval recall.
            "context_recall": 0.3,  # Ragas context metric — below threshold
            "llm_context_precision_with_reference": 0.3,
        },
    )

    out = rag_eval.run()

    # Retrieval path-overlap: a.md retrieved vs [a.md] relevant → recall 1.0, precision 1.0.
    assert out["retrieval"]["context_recall"] == 1.0
    assert out["retrieval"]["context_precision"] == 1.0
    # Ragas context metrics gated under their own columns, NOT overwritten by the retrieval 1.0.
    assert out["pass"]["context_recall"] is False
    assert out["pass"]["llm_context_precision_with_reference"] is False
    # Retrieval context metrics gated under their own namespaced keys.
    assert out["pass"]["retrieval_context_recall"] is True
    assert out["pass"]["retrieval_context_precision"] is True
    # Retrieval hit-rate/MRR gated and passing.
    assert out["pass"]["hit_rate"] is True
    assert out["pass"]["mrr"] is True
    # The failing Ragas context metric must sink overall_pass (no masking).
    assert out["overall_pass"] is False