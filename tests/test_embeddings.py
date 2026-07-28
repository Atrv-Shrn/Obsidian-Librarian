"""Tests for :class:`NomicEmbedding` — the nomic task-prefix wiring.

nomic-embed-text is trained for asymmetric retrieval: documents get ``search_document: `` and
queries get ``search_query: ``. The LlamaIndex ``BaseEmbedding`` methods must apply these
prefixes before handing text to FastEmbed. We monkeypatch the raw embed calls (``_embed_one`` /
``_embed_many``) so no model is loaded and assert the prefixes land on every code path.

This matters more now that the backend is FastEmbed rather than Ollama: FastEmbed does **not**
apply nomic's task prefixes itself (its ``embed()`` and ``query_embed()`` return identical
vectors for the same text), so these prefixes are entirely our responsibility. Dropping them
would silently degrade retrieval instead of failing.

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

def test_query_embeddings_batch_paths_use_the_query_prefix(monkeypatch):
    # The batch query paths (sync + async) must use the QUERY prefix. These route through
    # ``_embed_many``, which is also the document batch path — an easy place to leak the doc
    # prefix onto queries and tank recall without any visible error.
    import asyncio

    seen: list = []

    def fake_many(self, texts):
        seen.extend(texts)
        return [[0.0] for _ in texts]

    monkeypatch.setattr(NomicEmbedding, "_embed_many", fake_many)

    emb = NomicEmbedding()
    emb._get_query_embeddings(["a", "b"])
    asyncio.run(emb._aget_query_embeddings(["c"]))

    assert seen == ["search_query: a", "search_query: b", "search_query: c"]


# --------------------------------------------------------------------------- FastEmbed wiring


def test_dense_model_defaults_to_fastembed_nomic():
    """The dense model must be a FastEmbed nomic id, not an Ollama tag.

    Ollama was removed from the container; a bare ``nomic-embed-text`` (the Ollama tag) is not
    a FastEmbed model id and would fail at first embed. FastEmbed ids are HF-style ``org/name``.
    """
    from obsidian_librarian.config import Settings

    model = Settings().embed_model
    assert model.startswith("nomic-ai/"), model
    assert "nomic-embed-text-v1.5" in model
    # 768 dims either way — the collection width is unchanged by the swap.
    assert Settings().embed_dim == 768


def test_get_dense_embed_model_passes_exactly_the_supported_fields(monkeypatch):
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
        # No ``base_url``/``timeout`` any more — there is no embedding server to reach.
        assert set(seen) == {"model", "cache_dir", "doc_prefix", "query_prefix"}
    finally:
        emb_mod.get_dense_embed_model.cache_clear()


def test_dense_model_cache_dir_is_the_persistent_fastembed_path(monkeypatch):
    """The dense model must share the persistent FastEmbed cache with sparse + reranker.

    Without an explicit ``cache_dir`` FastEmbed falls back to a temp dir, so the ~133 MB nomic
    ONNX download would repeat on every container recreation — exactly the cost this swap was
    meant to remove.
    """
    from obsidian_librarian.config import get_settings
    from obsidian_librarian.rag import embeddings as emb_mod

    emb_mod.get_dense_embed_model.cache_clear()
    try:
        assert emb_mod.get_dense_embed_model().cache_dir == str(get_settings().fastembed_cache_path)
    finally:
        emb_mod.get_dense_embed_model.cache_clear()
