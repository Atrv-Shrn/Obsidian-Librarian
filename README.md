# Obsidian-Librarian

A personal AI librarian for your [Obsidian](https://obsidian.md) vault.

It has two halves. A **read-only RAG pipeline** turns your notes into grounded answers — parse,
chunk, embed, hybrid-retrieve, rerank, generate. A **LangGraph agent** sits on top of that,
talks to you in any OpenAI-compatible chat client, and can create, edit and delete notes through
the Obsidian Local REST API plugin — always proposing a write first and waiting for your "yes".

Everything ships as **one Docker container**. Qdrant, Redis, the embedding models and both
servers run inside it; the only things you bring are your vault and an API key.

```
you ──▶ chat client ──▶ :8000/v1 ──▶ agent ──┬──▶ RAG-MCP  ──▶ Qdrant + Redis ──▶ your notes
                                              └──▶ Obsidian plugin MCP ──▶ writes (after you confirm)
```

- Architecture: [`docs/SPEC.md`](docs/SPEC.md) · Project state: [`docs/PRD.md`](docs/PRD.md)
- The pipeline **never writes**. The agent does all writing, and only after you confirm.

---

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Using it](#using-it)
- [Connecting a chat client](#connecting-a-chat-client)
- [Security](#security)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## How it works

**Retrieval** is hybrid. Every chunk gets a dense vector (`nomic-embed-text-v1.5`, run in-process
via FastEmbed — no model server) and a sparse BM25 vector. Qdrant stores both per point and fuses
the two ranked lists server-side with Reciprocal Rank Fusion, so semantic similarity and exact
keyword matches both count. A cross-encoder then reranks the shortlist, because first-stage
retrieval optimises recall and reranking fixes ordering.

**Three models, three jobs.** `nomic-embed-text-v1.5` embeds (local, free, your notes never leave
the machine). `deepseek-v4-pro:cloud` generates. `glm-5.2:cloud` judges during evals — a different
model from the generator on purpose, so it never grades its own output.

**Writes are propose-then-confirm.** The agent replies with a `[PENDING_WRITE]` marker and a plan;
nothing touches disk until you reply "yes". Deletes honour Obsidian's own trash setting.

**Storage.** Qdrant holds vectors, Redis holds raw note text plus a content-hash set that skips
re-embedding unchanged notes, SQLite holds sync watermarks and agent conversation state. All of it
lives in one Docker volume.

---

## Requirements

| | |
|---|---|
| **Docker Desktop** or Docker Engine | Everything else ships in the image |
| **Ollama Cloud API key** | Required to *ask questions*. Indexing and retrieval work without one |
| **Obsidian + Local REST API plugin** | Only if you want the agent to write. [Plugin docs](https://coddingtonbear.github.io/obsidian-local-rest-api/) |
| **Langfuse keys** | Optional, for agent tracing |

Roughly 2 GB of disk for the image, plus ~400 MB of models downloaded on first run.

---

## Installation

### 1. Get the code

```bash
git clone https://github.com/Atrv-Shrn/Obsidian-Librarian.git
cd Obsidian-Librarian
```

### 2. Configure

```bash
cp .env.example .env
```

For a first run you only need these three lines in `.env`:

```bash
OLLAMA_API_KEY=sk-...                       # required to ask questions
HOST_VAULT_PATH=/absolute/path/to/Vault     # your vault, ON THE HOST
OBSIDIAN_API_KEY=...                        # only if you want writes
```

> [!IMPORTANT]
> **`HOST_VAULT_PATH` and `VAULT_PATH` are different things.**
> `HOST_VAULT_PATH` is the folder on *your machine* that gets bind-mounted in.
> `VAULT_PATH` (`/vault`) is the path *inside* the container — leave it alone.
>
> Setting `VAULT_PATH` to a host path mounts an empty directory: the index comes up with zero
> notes and every answer is *"I don't have enough in the vault."*
>
> Leave `HOST_VAULT_PATH` unset to run against the bundled `sample_vault/`. **Recommended for
> your first run** — it lets you watch the whole thing work before pointing it at real notes.

### 3. Build and start

```bash
docker compose build      # first build downloads Qdrant + ~250 MB of wheels
docker compose up -d
```

On first start the container downloads the FastEmbed ONNX models (~133 MB dense, plus BM25 and
the cross-encoder) into its volume, then indexes your vault. Give it a few minutes.

```bash
docker compose logs -f
```

### 4. Verify

```bash
curl http://localhost:8000/health
```

A healthy stack returns `200`:

```json
{"status":"ok","model":"deepseek-v4-pro:cloud","obsidian_writes":false,
 "write_confirm":true,"checks":{"qdrant":{"ok":true},"redis":{"ok":true}}}
```

`obsidian_writes:false` is expected unless Obsidian is running with the plugin enabled — see
[writes](#enabling-writes). A dependency being down gives `503` and `"status":"degraded"` with the
culprit named under `checks`.

Check the individual services:

```bash
docker exec obsidian-librarian supervisorctl status
```

All five of `qdrant`, `redis`, `rag-mcp`, `agent-api`, `vault-sync` should be `RUNNING`.

### 5. Ask it something

```bash
docker exec -it obsidian-librarian librarian query "What is Bayes' Theorem?"
```

If that returns a grounded answer with `[[wikilink]]` citations, you're done.

### Enabling writes

Writes need Obsidian **open** with the Local REST API plugin enabled, and `OBSIDIAN_API_KEY` set
to the plugin's bearer token.

> [!IMPORTANT]
> **Start Obsidian before the container**, or restart the agent afterwards:
> ```bash
> docker exec obsidian-librarian supervisorctl restart agent-api
> ```
> The agent probes the plugin once when it first builds and caches the result. If Obsidian wasn't
> reachable then, it runs read-only until restarted — `/health` will keep saying
> `"obsidian_writes": false` no matter how long you wait.

---

## Using it

```bash
# One-shot grounded answer from the RAG pipeline (no agent, no tools)
docker exec -it obsidian-librarian librarian query "How does hybrid search work?"

# Ranked chunks without generation — useful for debugging retrieval
docker exec -it obsidian-librarian librarian search "reranking"

# Interactive agent session (reads + propose-then-confirm writes)
docker exec -it obsidian-librarian librarian chat

# Re-index now instead of waiting for the scheduler
docker exec -it obsidian-librarian librarian sync

# Show resolved config (secrets masked)
docker exec -it obsidian-librarian librarian info
```

The vault re-syncs automatically every `SYNC_INTERVAL_MINUTES` (default 15). Only changed notes
are re-embedded — a content hash skips the rest.

---

## Connecting a chat client

Point any OpenAI-compatible client at:

```
http://localhost:8000/v1
```

No API key is required; enter any placeholder if the client insists. Works with Obsidian Copilot,
Open WebUI, and anything else that speaks the Chat Completions API.

> [!WARNING]
> **Use `http://`, not `https://`.** The server speaks plain HTTP. A TLS handshake against it
> fails as an opaque *"Connection error"* or *"Failed to fetch"* in the client, while the server
> logs `Invalid HTTP request received.` This is the single most common setup mistake.

If your client runs in a browser or Electron app, it may also need its origin allowed. Obsidian
sends `Origin: app://obsidian.md`, which is **not** a localhost origin even though it calls
`localhost` — add it to `API_CORS_ORIGINS`:

```bash
API_CORS_ORIGINS=app://obsidian.md,http://localhost,http://127.0.0.1
```

Then recreate the container (`docker compose up -d`) — CORS is read at startup.

---

## Security

Read this before changing anything about how the port is published.

**The endpoint is unauthenticated.** `/v1/chat/completions` takes no credentials and can drive
vault writes. `docker-compose.yml` therefore publishes it on loopback only:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

Only your machine can reach it. Changing this to `"8000:8000"` exposes read/write access to your
vault to **every device on your network** — and to the internet if the port is forwarded or the
container runs on a VPS.

**`API_CORS_ORIGINS` is not authentication.** CORS is enforced by browsers. `curl`, scripts, and
anything that isn't a browser ignore it completely. It stops a malicious *web page* from driving
your vault; it stops nothing else.

If you need access from another device, put real authentication in front of the endpoint first —
a reverse proxy with a bearer token or mTLS. Do not just widen the bind.

> [!WARNING]
> **Propose-then-confirm is enforced by the system prompt, not by the graph.** There is no
> hard guard preventing a write tool from firing without your approval — if the model ignores its
> instructions, the write is real and immediate.
>
> In adversarial testing it held: it refused an explicit *"don't ask for confirmation, just do
> it"*, and it identified and refused a prompt-injection payload planted inside a note that told
> it to delete the Inbox. But that is *behavioural*, not structural. Test against
> `sample_vault/` or a copy before pointing it at notes you care about.

---

## Configuration

Everything lives in `.env`; [`.env.example`](.env.example) documents every knob. The ones worth
knowing:

| Variable | Default | Notes |
|---|---|---|
| `OLLAMA_API_KEY` | — | Required for generation. Routes both cloud models |
| `HOST_VAULT_PATH` | `./sample_vault` | Your vault on the host |
| `OBSIDIAN_API_KEY` | — | Local REST API plugin token; required for writes |
| `WRITE_CONFIRM` | `true` | Set `false` to skip propose-then-confirm. **Don't** |
| `SYNC_INTERVAL_MINUTES` | `15` | Background re-index cadence |
| `API_CORS_ORIGINS` | *(localhost only)* | Comma-separated origins. Never `*` |
| `EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5-Q` | See the warning below |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Changing these needs a re-index |
| `RETRIEVAL_TOP_N` / `RERANK_TOP_K` | `20` / `6` | Candidates fetched / passed to the generator |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Optional agent tracing |

> [!CAUTION]
> **Changing `EMBED_MODEL` requires a full re-index.** Vectors from different models are not
> comparable, and because most models here are 768-dimensional, Qdrant will happily *accept* the
> new vectors alongside the old ones and return silently meaningless results — no error anywhere.
>
> ```bash
> docker compose down -v      # drops the volume: index, cache, watermarks
> docker compose up -d        # re-indexes from scratch
> ```
> Your notes are a bind mount and are never touched by this.

---

## Evaluation

The RAG pipeline is scored with [Ragas](https://docs.ragas.io) plus retrieval metrics, against a
golden set in `src/obsidian_librarian/evals/golden_set.jsonl`.

Eval dependencies are **not** in the serving image — they add ~500 MB the container never
otherwise imports. Install them first:

```bash
docker exec -u root -w /app obsidian-librarian pip install -e ".[evals]"
docker exec obsidian-librarian librarian eval
```

A full RAG eval takes several minutes — it drives the pipeline once per golden item and then
calls the judge model for the LLM-scored metrics.

Or locally, outside Docker:

```bash
pip install -e ".[evals]"
make eval
```

Current results against the bundled test vault:

| Metric | Score | Gate |
|---|---|---|
| hit_rate | 1.00 | 0.6 |
| mrr | 0.875 | 0.6 |
| faithfulness | 0.97 | 0.7 |
| answer_relevancy | 0.94 | 0.6 |
| context_recall | 0.94 | 0.6 |
| semantic_similarity | 0.91 | 0.7 |
| answer_correctness | 0.77 | 0.6 |
| abstention *(refuses when the answer isn't in the vault)* | 2/2 | — |

The golden set is small (8 scored items plus 2 negative). It catches gross regressions, not subtle
ones — worth growing if you build on this.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

202 tests, and they run **without** the container: `tests/conftest.py` installs an import hook that
stubs the heavy dependencies (Qdrant, Redis, LlamaIndex, ragas, fastmcp…) so the pure helpers can
be tested in isolation.

### Make targets

| Target | What it does |
|---|---|
| `make sync` | Scan the vault and apply index CRUD |
| `make query Q="..."` | One-shot grounded answer |
| `make chat` | Interactive agent session |
| `make eval` / `make eval-agent` | RAG evals / agent golden tasks |
| `make rag-serve` / `make api-serve` | Run a single server in the foreground |
| `make docker-up` / `make docker-down` | Build and run / stop the container |
| `make info` | Show resolved config |

On Windows, use the `docker compose` and `docker exec` commands directly — the Makefile assumes a
POSIX shell.

### Layout

```
src/obsidian_librarian/
├── config.py            single source of truth for every setting
├── cli.py               typer CLI (sync/search/query/chat/eval/info)
├── rag/                 the read-only pipeline
│   ├── embeddings.py    FastEmbed dense + sparse + cross-encoder
│   ├── reader.py        vault parsing, frontmatter, wikilinks
│   ├── ingest.py        chunk → embed → upsert, with dedup
│   ├── retrieve.py      hybrid search + RRF + rerank
│   ├── query_engine.py  retrieval → grounded generation
│   └── sync/            watermarks + APScheduler
├── mcp/rag_server.py    read-only MCP tools for the agent
├── agent/               LangGraph agent, prompts, memory, Langfuse
├── api/openai_compat.py OpenAI-compatible FastAPI endpoint
└── evals/               Ragas + retrieval metrics, golden sets
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| *"Connection error"* / *"Failed to fetch"* in a chat client | Almost always `https://` instead of `http://`. Check the server log for `Invalid HTTP request received.` |
| Every answer is *"I don't have enough in the vault"* | `HOST_VAULT_PATH` unset or pointing somewhere empty. Run `docker compose config` and check the bind source |
| `"obsidian_writes": false` with Obsidian open | The agent cached the probe at build time. `docker exec obsidian-librarian supervisorctl restart agent-api` |
| Writes never happen at all | Obsidian must be open with the Local REST API plugin enabled, and `OBSIDIAN_API_KEY` must match its token |
| `/health` returns 503 | Read `checks` in the body, then `supervisorctl status`, then `tail /data/<program>.err` |
| API refuses connections right after `up` | Still starting. Wait for `docker ps` to show `(healthy)` |
| Answers cite notes that no longer exist | Index is stale. `docker exec obsidian-librarian librarian sync` |
| Retrieval returns nonsense after changing `EMBED_MODEL` | Mixed embedding spaces. `docker compose down -v && docker compose up -d` |
| `sync error: <note>: timed out` | First embed pays the one-time model download. Self-heals next tick |
| CORS errors in a browser client | Add its origin to `API_CORS_ORIGINS`, then `docker compose up -d` |

Logs live in the container at `/data/<program>.log` and `/data/<program>.err` — one pair per
supervisord program.

---

## License

MIT — see [`LICENSE`](LICENSE).
