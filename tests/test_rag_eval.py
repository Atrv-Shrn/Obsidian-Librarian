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
    assert _THRESHOLDS["response_relevancy"] == 0.6
    assert _THRESHOLDS["hit_rate"] == 0.6
    assert _THRESHOLDS["mrr"] == 0.6
    # No threshold accidentally below 0 or above 1.
    for k, v in _THRESHOLDS.items():
        assert 0.0 < v <= 1.0, k


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
    # 1 of 1 relevant surfaced → recall 1.0; 1 of 2 retrieved is relevant → precision 0.5.
    assert out["context_recall"] == 1.0
    assert out["context_precision"] == 0.5


def test_retrieval_paths_suffix_match(monkeypatch):
    # ref "a.md" matches retrieved "notes/a.md" via endswith.
    _install_query_engine(monkeypatch, [{"path": "other.md"}, {"path": "notes/a.md"}])
    out = _retrieval_paths("q", ["a.md"])
    assert out["hit_rate"] == 1.0
    assert out["mrr"] == 0.5  # rank 2
    assert out["context_recall"] == 1.0
    assert out["context_precision"] == 0.5


def test_retrieval_paths_no_hit(monkeypatch):
    _install_query_engine(monkeypatch, [{"path": "unrelated.md"}])
    out = _retrieval_paths("q", ["notes/missing.md"])
    assert out["hit_rate"] == 0.0
    assert out["mrr"] == 0.0
    assert out["context_recall"] == 0.0
    assert out["context_precision"] == 0.0


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
            "response_relevancy": 0.9,
            "answer_correctness": 0.9,
            "semantic_similarity": 0.9,
            "context_recall": 0.3,  # Ragas context metric — below threshold
            "context_precision": 0.3,
        },
    )

    out = rag_eval.run()

    # Retrieval path-overlap: a.md retrieved vs [a.md] relevant → recall 1.0, precision 1.0.
    assert out["retrieval"]["context_recall"] == 1.0
    assert out["retrieval"]["context_precision"] == 1.0
    # Ragas context metrics gated under ``context_*`` and NOT overwritten by the retrieval 1.0.
    assert out["pass"]["context_recall"] is False
    assert out["pass"]["context_precision"] is False
    # Retrieval context metrics gated under their own namespaced keys.
    assert out["pass"]["retrieval_context_recall"] is True
    assert out["pass"]["retrieval_context_precision"] is True
    # Retrieval hit-rate/MRR gated and passing.
    assert out["pass"]["hit_rate"] is True
    assert out["pass"]["mrr"] is True
    # The failing Ragas context metric must sink overall_pass (no masking).
    assert out["overall_pass"] is False