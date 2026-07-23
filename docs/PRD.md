---
title: Obsidian-Librarian — PRD / Living State
tags: [prd, project-state]
type: prd
created: 2026-07-21
updated: 2026-07-23
related:
  - "[[SPEC]]"
---

# Obsidian-Librarian — PRD / Living State

> Source of truth for decisions, milestones, current state, and what's next.
> Full architecture lives in `docs/SPEC.md` (finalized). This file tracks *state*.

## What it is

A personal AI librarian for Obsidian. Two halves in **one Docker container**:

- **Part A — RAG pipeline** (read-only): LlamaIndex parse→split→embed→retrieve→rerank→generate.
  Qdrant (dense+sparse, RRF) + Redis (raw docstore + dedup) + SQLite (watermarks) +
  local Ollama `nomic-embed-text`. Evaluated by Ragas (non-LLM + LLM w/ `glm-5.2:cloud`
  judge) + LlamaIndex retrieval metrics + golden set. **Never writes, never uses Langfuse.**
- **Part B — Agent**: LangGraph + LangChain, `deepseek-v4-pro:cloud` via Ollama Cloud.
  Reads via our RAG-MCP; **writes via the Obsidian Local REST API plugin MCP**
  (propose-then-confirm HITL over chat). Obsidian skill file makes it a specialist.
  Evaluated by golden set + Langfuse. Exposed via FastAPI OpenAI-compatible `:8000/v1`.

## Locked decisions (with reasoning)

- **Pipeline read-only, agent owns all writes** — pipeline only feeds info; agent controls vault.
- **Single container** — runs anywhere; supervisord = PID1 over ollama/qdrant/redis/rag-mcp/agent-api.
- **Three models, three roles**: `nomic-embed-text` (local) embeds; `deepseek-v4-pro:cloud` generates;
  `glm-5.2:cloud` judges (≠ generator → no self-preference bias).
- **Baseline = hybrid (dense+sparse in Qdrant, server-side RRF) + cross-encoder rerank + generate** — not dense-only.
- **Writes via Obsidian plugin MCP** (boots with Obsidian, surgical PATCH by heading/block/frontmatter,
  link integrity) + an **Obsidian skill file** in the system prompt. No filesystem writer code.
- **Two seams**: MCP (agent↔tools) and FastAPI (user↔agent, OpenAI-compatible for Copilot/Open WebUI).
- **HITL = propose-then-confirm over stateless chat** (endpoint is stateless; interrupt() won't work).
  Agent streams plan + ends turn; user replies "yes" next turn; history resent → execute. `WRITE_CONFIRM=false` escapes.
- **One-shot A→Z** — no walking skeleton; build sequence is dependency order only.
- **Models download on first run** into `/data` (not pre-baked); truly-offline not required.

## Milestones

- [x] **M0** — Scaffold: repo, `pyproject.toml`, `config.py`, Dockerfile + supervisord + docker-compose,
      `.env.example`, `.gitignore`, `sample_vault/` seed, package layout, `docs/PRD.md`.
- [x] **M1** — Ingestion: reader, embeddings (nomic + BM25), ingest, watermarks, deletes.
- [x] **M2** — Query engine: hybrid retrieve + rerank + synthesize; `cli query`.
- [x] **M3** — RAG-MCP read server (FastMCP :8765).
- [x] **M4** — Agent: graph, llm, skill file, memory, observability; `cli chat`.
- [x] **M5** — Obsidian plugin writes + propose-then-confirm HITL (best-effort TLS-insecure MCP).
- [x] **M6** — FastAPI OpenAI endpoint (`/v1/chat/completions` SSE + `/v1/models`).
- [x] **M7** — APScheduler auto-sync + finalize single-container packaging (supervisord + vault-sync program).
- [x] **M8** — Evals: RAG (Ragas non-LLM + LLM glm judge + LlamaIndex) · Agent (golden + Langfuse).

## Current state

- On branch `prototype` (work branch; `main` untouched).
- **M0–M8 all implemented.** Full module tree under `src/obsidian_librarian/`:
  `config.py`, `cli.py`; `rag/{embeddings,reader,ingest,retrieve,query_engine}.py`,
  `rag/sync/{watermarks,scheduler}.py`; `mcp/rag_server.py`;
  `agent/{llm,prompts,memory,observability,graph}.py`, `agent/skills/obsidian.md`;
  `api/openai_compat.py`; `evals/{rag_eval,agent_eval}.py`, `evals/{golden_set,agent_tasks}.jsonl`.
- Packaging: `pyproject.toml`, `Dockerfile`, `supervisord.conf` (ollama/ollama-bootstrap/qdrant/
  redis/rag-mcp/agent-api/vault-sync), `docker-compose.yml`, `entrypoint.sh`, `Makefile`, `.env.example`.
- `python -m compileall` passes (all modules syntactically valid).
- **Verify pass complete.** Adversarial cross-module review ran; all confirmed findings applied:
  - `rag/embeddings.py`: fastembed reranker import path (`fastembed.rerank.cross_encoder.TextCrossEncoder`);
    query embeddings use the query prefix.
  - `rag/retrieve.py`: rerank uses `.rerank(query, docs)` plain floats in input order (no `.rank`/`top_k`);
    path-prefix filter uses `MatchText` + a PREFIX-tokenizer text payload index on `path` (Qdrant has no
    native `PrefixMatch`; `MatchValue` would be exact-equality only).
  - `rag/ingest.py`: `ensure_collection` best-effort creates the `path` text payload index; `sync_vault`
    + `reindex` now maintain the content-hash dedup set (`add_hash`/`remove_hash`) and count `skipped_dup`;
    dead `_ingest_one` removed.
  - `agent/graph.py`: `_reconcile_new_messages` diffs resent history vs checkpoint state (prevents
    `add_messages` duplicate ballooning while keeping the checkpointer for pending-write carry);
    `_invoke_config` returns the same handler instance so `ainvoke` can capture `_langfuse_trace_id`.
  - `agent/observability.py`: Langfuse v3-first (creds on the `Langfuse(...)` client singleton) with a
    v2 fallback; never raises.
  - `api/openai_compat.py`: `ChatMessage` carries `tool_calls`/`tool_call_id` and converts them to
    LangChain shape (preserves tool-call continuity on resent history); SSE stream buffers the final
    LLM run's prose, discards on tool events, and contains mid-stream errors as an SSE error chunk.
  - `evals/rag_eval.py`: ragas v0.4 result extraction via `to_pandas()` + `MetricResult.value` coercion
    (v0.3 dict fallback); empty `relevant_paths` → `None` metrics skipped from averages; `AspectCritic`
    built with `name`+`definition`.
  - `evals/agent_eval.py`: `tool_calls` scoped to turn-2 AI messages only (was re-counting turn-1 read
    tools, tainting `decline_honored`/`executed_write`); `_langfuse_trace_id` surfaced into the result
    and passed to the scorer (scores no longer orphaned).
  - `pyproject.toml`: pinned `ragas>=0.4,<0.5` and `langfuse>=3,<4`; removed unused `sse-starlette`.
- **Repo hygiene + test layer added (post-M8):**
  - `LICENSE` (MIT, Copyright (c) 2026 Atrv-Shrn); `pyproject.toml` `license = { text = "MIT" }`.
  - `README.md` given a Features / Requirements / Make-targets / Tests / License section (basics only,
    per request — full design stays in SPEC.md).
  - `.env.example` audited against `config.py`: an "Advanced / optional" block documents the 12
    non-required knobs (EMBED_DIM, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT, QDRANT_API_KEY,
    REDIS_DOCSTORE_NAMESPACE, RERANK_MODEL, SPARSE_MODEL, EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX,
    RAG_MCP_PATH, PENDING_WRITE_MARKER, API_DEFAULT_THREAD_ID) with comments flagging the
    don't-change-unless-you-know-why ones (embed prefixes, dim, marker).
  - `tests/` full unit-test layer (122 tests, all passing): `conftest.py` installs a `sys.meta_path`
    finder that stubs only the *missing* heavy packages (qdrant_client, redis, llama_index,
    langchain_ollama, langchain_mcp_adapters, ragas, apscheduler, fastmcp, httpx, datasets, plus
    optional langgraph.prebuilt / langgraph.checkpoint.sqlite) so tests import the REAL project
    modules and exercise the dependency-free helpers without refactoring source. The stub is both
    callable and subclassable (via a `_StubMeta` metaclass) so `class NomicEmbedding(BaseEmbedding):`
    imports cleanly. An autouse `_isolate_env` fixture points VAULT/DATA/SQLITE at per-test tmp dirs
    + resets the settings cache (production defaults `/vault`,`/data` are unwritable on Windows).
    Covers: config, watermarks, ingest, reader, graph (_reconcile_new_messages + _invoke_config),
    openai_compat, agent_eval, rag_eval, observability, and a parametrized import-smoke over every
    project module.
  - **Bug found + fixed by the test layer:** `api/openai_compat.py` `_to_lc_messages` built
    `AIMessage(content=c, tool_calls=tool_calls or None)`; an assistant message with no tool calls
    made `[] or None` → `None`, which `AIMessage` rejects (pydantic wants a list) → would crash any
    chat request containing a tool-call-less assistant turn. Fixed to pass the list directly.
  - `pyproject.toml` `[tool.pytest.ini_options]` gained `pythonpath = ["src"]` so the suite runs
    without `pip install -e .` (and `testpaths`, `asyncio_mode = "auto"`).
- **Full adversarial audit pass (2026-07-22/23, branch `prototype`):** re-verify-then-fix run over
  the whole project; 35 confirmed findings ranked by severity, all applied. Runtime-correctness
  fixes (CRITICAL/HIGH):
  - `agent/memory.py`: checkpointer is the **async** `AsyncSqliteSaver` (was the sync saver, which
    silently no-ops under `await ainvoke` → pending writes never carried across turns). `get_checkpointer`
    is now `async` and awaited in `graph.build_agent`.
  - `rag/ingest.py`: on modify/reindex, stale Qdrant points for shrunk notes lingered (ids are
    `uuid5(path::chunk_index)`; shrinking N→M leaves ::M..N-1 un-overwritten). `index_documents` now
    `delete_by_path(rel)` before re-upsert so only the current chunk set is retrievable.
  - `agent/graph.py`: `_reconcile_new_messages` does a multiset subtraction of resent history vs
    checkpoint state (was re-feeding prior answers as new messages). Real checkpoint-read failures
    propagate, not swallow.
  - `api/openai_compat.py`: `thread_id` is now the full SHA-1 of the first user message (was a
    200-char-truncated opener → long shared prefixes collided cross-conversation, leaking pending
    writes); `X-Conversation-Id` header takes precedence when supplied. Errors redacted (SSE error
    chunk + 500, no traceback leak). CORS default tightened to localhost-only (was `*`).
  - `rag/reader.py`: `load_note` now resolves + checks `is_relative_to(vault)` before reading —
    rejects `../` traversal and absolute paths (out-of-vault read).
  - `agent/prompts.py`: HITL marker is templated from `pending_write_marker` (was hardcoded
    `[PENDING_WRITE]`, ignoring the setting). `mcp/rag_server.py` `reindex` invalidates the
    `_vault_docs` lru_cache. `evals/agent_eval.py`: `_check_read` negative-signal set no longer
    includes the bare word "vault" (trivially passed); NFKC + explicit curly-quote normalization so
    "don't" (U+2019) is caught; turn-2 write detection uses `_is_write_tool` (was counting read tools
    as writes, tainting `decline_honored`/`executed_write`).
  - LOW/hygiene: `ingest.indexed_at` is now TZ-aware UTC ISO-8601 (was naive `time.strftime`);
    `watermarks.connection()` context manager added + used in production paths (no more leaked SQLite
    handles; raw `connect()` kept for tests); scheduler's tautological
    `next_run_time=None if not s.sync_on_start else None` replaced with an explicit deferred first
    tick (avoids a double on-start run alongside `main()`'s explicit `_job()`).
  - Packaging/container: FastEmbed `cache_dir` → persistent `/data/caches/fastembed` (via
    `config.fastembed_cache_path`); `HF_HOME` → `/data/caches/hf`. Non-root `librarian` user (uid 1000)
    + `gosu` drop in `entrypoint.sh`; `supervisord` `user=librarian`; entrypoint chowns bind mounts.
    `docker-compose.yml` forwards `OLLAMA_BASE_URL`/`OLLAMA_LOCAL_BASE_URL`/`LANGFUSE_HOST`/
    `API_CORS_ORIGINS`. `pyproject.toml`: dropped unused `tenacity` + 3 unused `llama-index-*` pkgs,
    bumped `langchain-mcp-adapters>=0.2`, added `evals/*.jsonl` package-data, removed stale
    `asyncio_mode = "auto"`. `.env.example`: embed-prefix values quoted (dotenv strips trailing
    whitespace from unquoted values).
- **Second adversarial verify pass (2026-07-23, branch `prototype`):** a dedicated
  `verify-spec-conformance` workflow re-audited the post-fix tree; 6 of 8 raised findings confirmed
  (3 MEDIUM, 3 LOW) and applied:
  - **M — `mcp/rag_server.py` `get_backlinks` title match (Fix 1):** now resolves the target note's
    frontmatter title from the vault snapshot and matches wikilink targets against it, mirroring
    `reader.compute_backlinks` (which indexes by title AND stem). Stem-only matching missed
    title-targeted links, so `get_backlinks` returned a set that contradicted the `backlinks` payload
    stored in Qdrant.
  - **M — `api/openai_compat.py` stream-error containment (Fix 2):** `_to_lc_messages(req.messages)`
    moved inside `_stream_agent`'s `try:`. It's an async generator, so that line runs on Starlette's
    first `__anext__` — after `StreamingResponse` has sent the 200 + SSE headers. A conversion error
    there used to escape the generator → client got a 200 with a truncated stream and no
    `data: error` / `data: [DONE]`. Now contained as an SSE error chunk.
  - **M — `evals/rag_eval.py` context-metric clobber (Fix 3):** the Ragas LLM/non-LLM context metrics
    and our path-overlap retrieval context metrics both emit `context_recall` / `context_precision`.
    The merge wrote them under one key, so whichever family landed last silently clobbered the other
    and the gate tested only one family (a failing Ragas `LLMContextRecall=0.3` masked by a strong
    retrieval overlap `1.0` would pass). Retrieval-derived context metrics renamed to
    `retrieval_context_*` and gated under their own thresholds; both families scored independently.
  - **M — `rag/ingest.py` raw docstore stores verbatim markdown (Fix 4):** SPEC says the Redis raw
    store holds the *full raw markdown* (with the leading `---` frontmatter) for citations, but
    `index_documents` stored `src.text` — the frontmatter-stripped body (load_note/load_vault strip
    the YAML block). Now re-reads the verbatim file from disk; falls back to the stripped body only
    if the file can't be re-read (deleted mid-sync).
  - **L — `api/openai_compat.py` CORS reads via `Settings` (Fix 5):** `_configure_cors` read
    `os.environ.get("API_CORS_ORIGINS")` directly, violating `config.py`'s "nothing reads env vars
    directly" contract and silently ignoring a value set only in `.env` (pydantic-settings
    populates model fields but does NOT export them back to `os.environ`). Now reads
    `Settings().api_cors_origins` via a transient `Settings()` (not the cached `get_settings()`, to
    avoid import-time `ensure_dirs()`/mkdir of `/data`). `import os` removed.
  - **L — `evals/rag_eval.py` `_path_matches` component-aligned (Fix 6):** was a bare `str.endswith`,
    which over-matches on character tails (`"notes.md".endswith("es.md")` is True for two unrelated
    notes), inflating hit-rate/MRR/context-precision/recall. Now compares path components
    (`PurePosixPath` parts) so a real dir-prefixed-vs-bare suffix still matches both ways but a
    shared character tail does not.
  - **Bug found + fixed by the Fix-3 guard test (bonus, beyond the 6):** writing the run()-threshold
    guard forced the ragas code path to actually execute, which surfaced that `_metric(name, **kw)`
    collided with the `AspectCritic` call site `_metric("AspectCritic", name="helpfulness",
    definition=...)` — a `TypeError: got multiple values for argument 'name'` raised at *call*
    time, before `_metric`'s internal try/except, propagating to `run()`'s outer `except` and
    aborting the **entire** Ragas block (the LLM loop mid-iteration, before the non-LLM `evaluate`
    ran) → both `ragas_non_llm` and `ragas_llm` always empty in production. Renamed the param to
    `metric_name`. This is a real production correctness fix, not a test artifact.
  - Test-adequacy guards added for each fix (2 in `test_rag_server.py` for `get_backlinks`
    title/stem matching; 2 in `test_rag_eval.py` for `_path_matches` component alignment and the
    `run()` threshold gate proving the two context-metric families are gated independently). Suite
  148 → **165 tests, all passing.**
  - Test layer grew 122 → **148 tests, all passing**. New: `test_prompts` (HITL on/off + marker
    substitution + skill load), `test_query_engine` (`_context_block` + `format_sources` dedup +
    empty-retrieval branch), `test_retrieve` (rerank ordering + top-k + ties + empty), reader
    traversal/absolute-path containment, `agent_eval` curly-apostrophe + `_is_write_tool` + bare-vault
    regression, watermarks `connection()` closes handle. `conftest` autouse fixture now clears the
    project's `@lru_cache` singletons per test (embedder/reranker/LLM/prompt/vault-doc cache).
    Fixed two tautological/wrong-reason asserts in `test_ingest` (`.replace(..., 0)` no-op + trivial
    coverage checks → real max-chars + full-coverage invariants).
- **Design-sensitive choices APPLIED but flagged for user sign-off** (not yet confirmed by user;
  revertible):
  - **Conversation-id source (#4):** `thread_id` = full SHA-1 of first user message, with
    `X-Conversation-Id` header override. Alternative: derive from an explicit client-sent id only.
  - **Non-root container + RO vault (#7):** supervisord/children run as `librarian` (uid 1000);
    entrypoint chowns `/vault` + `/data` at boot. **NOT runtime-tested here** (no Docker run in this
    pass) — needs a `docker compose up` smoke before merge. A read-only vault mount was considered
    and **not** applied (the agent writes via the Obsidian plugin MCP over HTTP, not the filesystem,
    so RO would be safe — but the entrypoint currently chowns `/vault` RW; switch to `read_only: true`
    only if confirmed).
  - **CORS origins (#11):** default localhost-only via a regex; `API_CORS_ORIGINS=*` opts back into
    the old permissive behavior. The endpoint is unauthenticated + write-capable, so `*` is unsafe
    without network-level isolation.

- **Skim audit (2026-07-23, branch `prototype`) — findings RAISED, not yet fixed.** Suite still
  165/165 green; `ruff` reports 17 lint errors. Open items, by severity:
  - **H — `evals/agent_eval.py` `_score_langfuse` uses a v2 API.** Calls `Langfuse().score(...)`;
    langfuse v3 (the pin: `>=3,<4`, 3.0.0 installed) exposes only `create_score` /
    `score_current_span` / `score_current_trace`. The call raises, the `except` swallows it → agent
    scores **never reach Langfuse**, defeating the trace-id plumbing added in the last pass. It also
    builds a bare `Langfuse()` (no creds; `config.py` never exports to `os.environ`) instead of the
    singleton `observability._init_langfuse_client()` registers — and `get_langfuse_handler` is
    imported there but unused.
  - **H — `docker-compose.yml` / `.env.example` `VAULT_PATH` collision.** Compose interpolates
    `${VAULT_PATH:-./sample_vault}` as the **host** mount source; `.env.example` sets
    `VAULT_PATH=/vault` (the **container** path). Copy `.env.example`→`.env` then `make docker-up`
    and it bind-mounts a non-existent host `/vault`. Needs distinct names (e.g. `HOST_VAULT_PATH`).
  - **H — `entrypoint.sh` chowns the user's real vault.** `chown -R librarian /vault` rewrites
    ownership of every file in the bind-mounted Obsidian vault on the host (Linux). The pipeline only
    *reads* the vault and the agent writes over HTTP via the plugin — so the chown buys nothing and
    can lock the user out of their own notes. Drop `/vault` from the chown (keep `/data`).
  - **M — `evals/rag_eval.py` threshold keys likely don't match ragas column names.**
    `_THRESHOLDS` gates `response_relevancy` and `context_precision`, but ragas names those columns
    `answer_relevancy` (ResponseRelevancy) and `llm_context_precision_with_reference` /
    `non_llm_context_recall`. `results["pass"]` only includes keys present in `flat`, so a
    non-matching key **fails open** — those gates silently never fire. Verify against live ragas.
  - **M — Qdrant `--config-path /dev/null` is unverified.** An empty YAML may fail the config
    deserializer → supervisord's qdrant program crash-loops and the stack never serves. Needs the
    deferred `docker compose up` smoke (same smoke that covers the non-root change).
  - **L — `rag/embeddings.py` passes `embed_dim=` to `NomicEmbedding`,** which declares no such
    field. Harmless only because pydantic v2 defaults to `extra="ignore"`; dead either way.
  - **L — `agent_tasks.jsonl` carries fields nothing reads:** `expects_marker`,
    `must_propose_before_write`, `expected_path_contains`. The last one is the schema promise for
    SPEC's "write tasks assert the resulting file" — still unimplemented (see deferred item below).
  - **L — hygiene:** `.env.example` documents 42/44 settings, missing `API_CORS_ORIGINS`
    (security-relevant, forwarded by compose) and `FASTEMBED_CACHE_PATH`; README's Tests section
    names `langchain-openai` (project uses `langchain-ollama`); Dockerfile header comment omits the
    `vault-sync` + `ollama-bootstrap` programs; 3 of the ruff errors are dead locals
    (`graph.py:90`, `agent_eval.py:184`, `agent_eval.py:215`); stale pre-fix `build/lib/` tree and a
    duplicated 28 KB `.claude/` audit workflow sit untracked in the worktree (`.claude/` is not in
    `.gitignore`); **the entire project is still uncommitted** — one commit (`first commit`) with
    everything else untracked on `prototype`.
  - **Doc drift:** SPEC says node ids are `sha1(rel_path::chunk_index)`; the implementation uses
    `uuid5` (correct — Qdrant requires UUID/int ids), so SPEC is stale. This PRD said `reindex`
    invalidates an `_vault_docs` **lru_cache**; it is now a TTL cache.

- **M9 — FIRST REAL CONTAINER RUN + image optimization (2026-07-23, branch `prototype`). Stack is
  UP and the full ingest path is verified end-to-end.** `docker compose up -d` → all 7 supervisord
  programs healthy, `/health` + `/v1/models` answer from the host, Qdrant collection `obsidian_librarian`
  built with the right shape (dense 768 cosine + named sparse), all 5 sample notes → **17 chunks**
  indexed, watermarks written, raw markdown in Redis, `nomic-embed-text` (274 MB) pulled by
  `ollama-bootstrap`. Bugs the boot found that three static audits did not:
  - **`Dockerfile` missing `zstd`** — the Ollama install script now ships a zstd-compressed payload
    and aborts without it; `python:3.11-slim` has no zstd. Build-blocking. Fixed.
  - **`supervisord.conf` qdrant passed `--storage-snapshot-path`, which is not a qdrant flag**
    (only `--storage-snapshot` exists, and it *restores* a snapshot). Qdrant exited instantly and sat
    in `BACKOFF` **while every other program reported RUNNING** — the vector store was dead and
    nothing surfaced it. Flag removed; storage/snapshots now come from `directory=/data/qdrant`
    (qdrant resolves `./storage` + `./snapshots` relative to cwd → both land on the volume, verified).
    NOTE: the earlier audit blamed `--config-path /dev/null` for this — **that was wrong**;
    `/dev/null` was tested directly and works fine (qdrant starts on built-in defaults).
  - **`VAULT_PATH` collision** (raised in the skim audit) — fixed and verified: compose now reads
    `HOST_VAULT_PATH`; a `.env` copied from `.env.example` resolves the mount to `./sample_vault`
    instead of the literal `/vault`.
  - **Image optimization — 5.75 GB → 1.09 GB disk (−81%), 2.1 GB → 256 MB content (−88%),
    image unpack 1050 s → 49 s (−95%), export stage 1316 s → 172 s.** Measured causes and fixes:
    - Ollama's installer ships GPU runners we can never use (`cuda_v12` 1.2 GB + `cuda_v13` 831 MB +
      `vulkan` 47 MB = 2.07 GB). Deleted **in the same RUN layer** (a later `rm` would not reclaim).
      Verified ollama still serves with `library=cpu`. `/usr/local/lib/ollama`: 2.1 GB → 28 MB.
    - `ragas` + `datasets` moved to a **`[evals]` optional extra** (~500 MB: pyarrow 156 M, pandas
      79 M, sknetwork 35 M, langchain-community 25 M, openai 20 M, nltk 15 M). Safe because every
      ragas import in `rag_eval.py` is lazy and inside `try/except` — the slim image imports fine and
      reports the ragas families under `skipped`. `make install-evals` / `make docker-build-evals`
      (`--build-arg INSTALL_EXTRAS=[evals]`) restore them.
    - `uv` replaces pip for the install (parallel resolve+download; the bottleneck is a ~230 kB/s
      link), installed and removed inside the one layer. BuildKit cache mounts for `uv` + `pip` make
      an interrupted build resume instead of re-downloading. `__pycache__` stripped from site-packages.
    - **`.dockerignore` added** — excludes `.git`, `build/`, `__pycache__`, `.pytest_cache`, `docs/`,
      `tests/`, `.claude/`, and critically **`.env`**, so a secrets file can never be baked in.
      Keeps `README.md` (pyproject declares it as the package readme; the build fails without it).
  - Suite still **165/165 green** after the dependency split.
  - **Open, found during the run (not yet fixed):** (a) first sync logged
    `sync error: 2026-07-10.md: timed out` — the cold-start CPU embed exceeded `NomicEmbedding`'s
    60 s timeout; the per-file try/except caught it and the next scheduled tick indexed it, so it
    self-healed, but the timeout is tight for a cold model load. (b) That failure left the file's
    content hash already in the Redis dedup set, so the successful retry counted it as
    `dup=1` — a failed index should not register its hash. Accounting-only, not data loss.
    (c) `/health` reports `status: ok` from settings alone; it never probes Qdrant/Redis, so it
    returned `ok` the whole time qdrant was in BACKOFF. A real dependency probe is worth adding.

## What's next

- **Triage the remaining skim-audit findings** (Langfuse `create_score`, ragas threshold-key
  mismatch, ruff dead code, `.env.example` gaps) — none block the running stack.
- **Fill `.env` with `OLLAMA_API_KEY`** to exercise generation (`query` / `chat`); everything up to
  retrieval already works without it.
- **Commit the tree** — still one commit (`first commit`) with everything else untracked.
- Optional startup win, needs sign-off (changes a locked SPEC decision): FastEmbed already ships
  `nomic-embed-text-v1.5` as ONNX, so dropping local Ollama would remove its 28 MB runtime, the
  `ollama` + `ollama-bootstrap` programs, and the 274 MB first-boot model pull.
- First real run requires the live stack (Ollama + Qdrant + Redis + OLLAMA_API_KEY): `make sync`
  then `make eval` (RAG) / `make eval-agent` (needs Obsidian plugin up + optional Langfuse).
- Obsidian plugin MCP write-execution assertions are deferred while Obsidian isn't running
  (marker/HITL behavior is still fully checked by `agent_eval`).
- **Deferred hardening (not blocking, not a crash):** HITL is currently prompt-only — there is no
  graph-level tool-execution guard that blocks a write tool from firing without a prior
  `[PENDING_WRITE]` marker. The propose-then-confirm contract is enforced by the system prompt +
  eval assertions, not by the graph. Promoting this to a graph-level guard is a design change left
  for a future pass.

## Deferred (out of scope until trigger fires)

Filesystem write fallback (headless writes) · endpoint auth / non-localhost binding · GPU embeddings ·
Qdrant as scaled service · self-hosted Langfuse · watchdog instant sync · instant post-write reindex ·
sub-agents. See SPEC.md "Defer until a trigger fires".