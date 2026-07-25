# Obsidian-Librarian

A personal AI librarian for your [Obsidian](https://obsidian.md) vault — a **read-only
RAG pipeline** (LlamaIndex + Qdrant + Redis + in-process FastEmbed embeddings) that turns your
notes into grounded answers, and a **LangGraph agent** (deepseek via Ollama Cloud) that
reasons over those tools, talks to you, and controls the vault — create, edit, delete
notes — through the Obsidian Local REST API plugin. **Ships as one Docker container.**

* Spec: [`docs/SPEC.md`](docs/SPEC.md) · Living state: [`docs/PRD.md`](docs/PRD.md)
* The pipeline **never writes**; the agent does **all** writing.
* Two seams: **MCP** (agent ↔ RAG-MCP read + Obsidian plugin write) and **FastAPI**
  (you ↔ agent, OpenAI-compatible `:8000/v1`).

## Installation

### 1. Prerequisites

- **Docker Desktop** (or Docker Engine) — running. Everything else ships in the image.
- An **Ollama Cloud API key** — needed for generation and the eval judge. Indexing and
  retrieval work without one; asking questions does not.
- *(writes only)* **Obsidian** running with the
  [Local REST API plugin](https://coddingtonbear.github.io/obsidian-local-rest-api/) enabled,
  and its bearer token.

### 2. Get the code

```bash
git clone https://github.com/Atrv-Shrn/Obsidian-Librarian.git
cd Obsidian-Librarian
```

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`. For a first run you only need these:

```bash
OLLAMA_API_KEY=sk-...                      # required to ask questions
HOST_VAULT_PATH=/absolute/path/to/Vault    # your vault ON THE HOST
OBSIDIAN_API_KEY=...                       # only if you want the agent to write
```

> [!important] `HOST_VAULT_PATH` vs `VAULT_PATH` — they are not the same thing
> `HOST_VAULT_PATH` is the folder **on your machine** that gets bind-mounted into the
> container. `VAULT_PATH` (`/vault`) is the path **inside** the container — leave it alone.
> Setting `VAULT_PATH` to a host path mounts an empty directory: the index comes up with zero
> notes and every answer is *"I don't have enough in the vault."*
>
> Leave `HOST_VAULT_PATH` unset to run against the bundled `sample_vault/` — recommended for
> your first run.

### 4. Build and start

```bash
docker compose build      # first build is slow: it downloads Qdrant and ~250 MB of wheels
docker compose up -d
```

On **first** start the container downloads the FastEmbed ONNX models (nomic dense ~133 MB,
BM25 sparse, cross-encoder rerank) into `/data`, then indexes your vault. Expect a few minutes
before it answers. Watch it happen:

```bash
docker compose logs -f
```

### 5. Verify

```bash
curl http://localhost:8000/health
```

A healthy stack returns `200` and:

```json
{"status":"ok","model":"deepseek-v4-pro:cloud","obsidian_writes":false,
 "write_confirm":true,"checks":{"qdrant":{"ok":true},"redis":{"ok":true}}}
```

If a dependency is down you get **503** and `"status":"degraded"` with the culprit named under
`checks`. `docker ps` also shows `(healthy)` / `(unhealthy)`. To see the individual services:

```bash
docker exec obsidian-librarian supervisorctl status
```

All five of `qdrant`, `redis`, `rag-mcp`, `agent-api`, `vault-sync` should be `RUNNING`.

### 6. Use it

```bash
# One-shot grounded answer from the RAG pipeline (no agent)
docker exec -it obsidian-librarian librarian query "What is Bayes' Theorem?"

# Interactive agent session (reads + propose-then-confirm writes)
docker exec -it obsidian-librarian librarian chat

# Force a re-index
docker exec -it obsidian-librarian librarian sync
```

Or point a chat client at `http://localhost:8000/v1` — Obsidian Copilot, Open WebUI, or
anything that speaks the OpenAI Chat Completions API. Use `http://`, **not** `https://`: the
server speaks plain HTTP, and a TLS handshake against it fails as an opaque "Connection error"
in the client while the server logs `Invalid HTTP request received.`. No API key is required.

> [!important] The endpoint is unauthenticated — keep it on loopback
> `/v1/chat/completions` takes no credentials and can drive vault **writes**. `docker-compose.yml`
> therefore publishes it as `127.0.0.1:8000:8000`, so only this machine can reach it. Changing
> that to `8000:8000` exposes read/write access to your vault to every device on your network —
> and to the internet if the port is forwarded or the container runs on a VPS.
>
> `API_CORS_ORIGINS` is **not** a substitute: CORS is a browser mechanism, and `curl` or any
> script ignores it. If you need access from another device, put real authentication in front
> of the endpoint first.

> [!warning] Before you point this at your real vault
> The agent's **propose-then-confirm is enforced by the system prompt, not by the graph.** There
> is no hard guard preventing a write tool from firing without your approval — if the model
> ignores the instruction, the write is real and immediate. Test writes against `sample_vault/`
> or a copy of your vault first. Deletes honor Obsidian's own trash setting.

### Troubleshooting

| Symptom | Cause |
| --- | --- |
| Every answer is *"I don't have enough in the vault"* | `HOST_VAULT_PATH` unset or pointing somewhere empty — check `docker compose config` and confirm the bind source is what you expect. |
| `/health` returns 503 | Read `checks` in the body, then `docker exec obsidian-librarian supervisorctl status` to find the dead program and `tail /data/<program>.err`. |
| Container is up but the API refuses connections | It's still starting — the API restarts once while dependencies settle. Wait for `docker ps` to report `(healthy)`. |
| `sync error: <note>: timed out` | The first embed after a cold start pays the one-time FastEmbed model download. It self-heals on the next sync tick. |
| Writes never happen | Obsidian must be open with the Local REST API plugin running. Check `"obsidian_writes"` in `/health`. |
| `make: command not found` (Windows) | The Makefile is a convenience for Linux/macOS. Use the `docker compose` / `docker exec` commands above. |

## Features

- **Read-only RAG pipeline** — header-aware chunking, in-process `nomic-embed-text-v1.5`
  dense + BM25 sparse embeddings (both FastEmbed ONNX, no model server), Qdrant hybrid retrieval with server-side RRF, cross-encoder rerank,
  grounded generation. Redis keeps raw note text + a content-hash dedup set; SQLite holds
  sync watermarks. The pipeline **never writes to the vault**.
- **LangGraph agent** — `deepseek-v4-pro:cloud` via Ollama Cloud, reads through the RAG-MCP,
  writes through the Obsidian Local REST API plugin MCP with **propose-then-confirm** HITL
  (`[PENDING_WRITE]` marker → you reply "yes" → write executes).
- **One container** — supervisord runs Qdrant, Redis, the RAG-MCP server, the agent API,
  and a scheduled vault sync together. Embeddings run in-process; no local model server.
- **OpenAI-compatible API** at `:8000/v1` for any chat client.
- **Evals** — Ragas (non-LLM + LLM with a `glm-5.2:cloud` judge ≠ generator) + LlamaIndex
  retrieval metrics for the pipeline; golden tasks + Langfuse for the agent.

## Requirements

- Docker (the container ships everything except your vault and API keys).
- An **Ollama Cloud** API key (`OLLAMA_API_KEY`) — routes both generation and judge models.
- For writes: the [Obsidian Local REST API plugin](https://coddingtonbear.github.io/obsidian-local-rest-api/)
  running in your vault, plus its `OBSIDIAN_API_KEY`.
- Optional: Langfuse keys (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`) for agent tracing.

Copy [`.env.example`](.env.example) to `.env` and fill in the keys. Only the `replace-me`
API-key values are truly required to run; everything else has a working default.

## Make targets

After `make install` (or inside the container), the common entry points:

| Target | What it does |
| --- | --- |
| `make sync` | Scan the vault and apply index CRUD (add/modify/delete, dedup). |
| `make query Q="..."` | One-shot grounded answer from the RAG pipeline. |
| `make chat` | Interactive agent session (reads + propose-then-confirm writes). |
| `make rag-serve` | Start the RAG-MCP read server (`:8765/mcp`). |
| `make api-serve` | Start the OpenAI-compatible FastAPI server (`:8000/v1`). |
| `make eval` | Run the RAG eval suite (Ragas + retrieval metrics). Needs `make install-evals` first. |
| `make eval-agent` | Run the agent golden-task eval (+ Langfuse if configured). |
| `make docker-up` / `make docker-down` | Build and run / stop the full single container. |

See the [`Makefile`](Makefile) for the full list.

## Optional extras

Ragas and its dependency tree (pyarrow, pandas, scikit-network, nltk — ~500 MB) are **not**
runtime dependencies; the serving container never imports them. Install them only to run the
RAG evals:

```bash
pip install -e ".[evals]"          # or: make install-evals
make docker-build-evals            # same, for the container image
```

Without the extra, `rag_eval` still runs and reports the retrieval metrics, listing the ragas
metric families under `skipped`.

## Tests

Unit tests for the dependency-free helpers live under `tests/`. The heavy stack
(Qdrant, Redis, LlamaIndex, langchain-ollama, ragas, …) is auto-stubbed by
`tests/conftest.py`, so the suite runs without the container:

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [`LICENSE`](LICENSE).