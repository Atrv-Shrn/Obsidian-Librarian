"""Tests for :class:`NomicEmbedding` — the nomic task-prefix wiring.

nomic-embed-text is trained for asymmetric retrieval: documents get ``search_document: `` and
queries get ``search_query: ``. The LlamaIndex ``BaseEmbedding`` methods must apply these
prefixes before hitting Ollama. We monkeypatch the raw Ollama call (``_embed_one`` /
``_embed_many``) so no network is touched and assert the prefixes land on every code path.

``BaseEmbedding`` is stubbed when LlamaIndex isn't installed, but ``NomicEmbedding`` subclasses
the stub fine (the stub metaclass is subclassable + the class body defines real methods +
defaults), so the prefix behavior is exercised identically in stub and real-dep envs.
"""

from __future__ import annotations

from obsidian_librarian.rag.embeddings import NomicEmbedding


def test_nomic_applies_query_and_doc_prefixes(monkeypatch):
    seen: list = []

    def fake_embed_one(self, text):
        seen.append(text)
        return [0.1, 0.2]

    def fake_embed_many(self, texts):
        seen.extend(texts)
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(NomicEmbedding, "_embed_one", fake_embed_one)
    monkeypatch.setattr(NomicEmbedding, "_embed_many", fake_embed_many)

    emb = NomicEmbedding()
    emb._get_query_embedding("what is x?")
    emb._get_text_embedding("x is a thing")
    emb._get_text_embeddings(["a", "b"])

    assert seen == [
        "search_query: what is x?",
        "search_document: x is a thing",
        "search_document: a",
        "search_document: b",
    ]


def test_nomic_query_prefix_distinct_from_doc_prefix(monkeypatch):
    # Regression guard: the query path must use the query prefix, not the doc prefix — a
    # swapped prefix would embed retrieval queries as documents and tank recall.
    seen: list = []

    def capture_one(self, text):
        seen.append(text)
        return [0.0]

    def fake_many(self, texts):
        return [[0.0] for _ in texts]

    monkeypatch.setattr(NomicEmbedding, "_embed_one", capture_one)
    monkeypatch.setattr(NomicEmbedding, "_embed_many", fake_many)

    emb = NomicEmbedding()
    emb._get_query_embedding("find this")
    assert seen == ["search_query: find this"]
    assert "search_document:" not in seen[0]

# --------------------------------------------------------------------------- timeout wiring
#
# Regression guard for the first-real-run failure: the cold model load on container start
# (Ollama maps ~274 MB and spins a CPU runner) blew past the old hardcoded 60 s ceiling, so the
# very first sync logged `sync error: <note>: timed out` and the note was only picked up on the
# next scheduled tick.


def test_embed_timeout_comes_from_settings(monkeypatch):
    from obsidian_librarian.config import get_settings, reset_settings_cache
    from obsidian_librarian.rag import embeddings as emb_mod

    monkeypatch.setenv("EMBED_REQUEST_TIMEOUT", "123.5")
    reset_settings_cache()
    emb_mod.get_dense_embed_model.cache_clear()
    try:
        assert get_settings().embed_request_timeout == 123.5
        assert emb_mod.get_dense_embed_model().timeout == 123.5
    finally:
        reset_settings_cache()
        emb_mod.get_dense_embed_model.cache_clear()


def test_embed_default_timeout_survives_a_cold_model_load():
    # The default must leave real headroom for a first-call cold start, not the old 60 s.
    from obsidian_librarian.config import Settings

    assert Settings().embed_request_timeout >= 300.0
    assert NomicEmbedding().timeout >= 300.0


def test_get_dense_embed_model_does_not_pass_unknown_fields(monkeypatch):
    # ``embed_dim`` used to be passed to NomicEmbedding, which declares no such field; pydantic's
    # extra="ignore" silently dropped it. Assert on the constructor kwargs rather than hasattr:
    # BaseEmbedding is stubbed in this suite and a stub answers *any* attribute, so hasattr
    # would pass no matter what. Capture what get_dense_embed_model actually passes.
    from obsidian_librarian.rag import embeddings as emb_mod

    seen: dict = {}

    class _Capture(emb_mod.NomicEmbedding):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(emb_mod, "NomicEmbedding", _Capture)
    emb_mod.get_dense_embed_model.cache_clear()
    try:
        emb_mod.get_dense_embed_model()
        assert "embed_dim" not in seen
        # The kwargs it *should* pass, including the timeout that fixes the cold-start failure.
        assert set(seen) == {"base_url", "model", "doc_prefix", "query_prefix", "timeout"}
    finally:
        emb_mod.get_dense_embed_model.cache_clear()
