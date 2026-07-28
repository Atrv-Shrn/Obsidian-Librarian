# syntax=docker/dockerfile:1
# Obsidian-Librarian — single container.
# supervisord (PID 1) supervises: qdrant, redis, rag-mcp, agent-api, vault-sync.
# Embedding/rerank models download into /data on first run.
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VAULT_PATH=/vault \
    DATA_PATH=/data \
    QDRANT_URL=http://127.0.0.1:6333 \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    SQLITE_PATH=/data/librarian.db

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates wget tar xz-utils \
        redis-server supervisor sqlite3 procps gosu \
    && rm -rf /var/lib/apt/lists/*

# --- Qdrant (bundled binary) ---
ARG QDRANT_VERSION=v1.11.3
RUN curl -fsSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C /usr/local/bin qdrant && chmod +x /usr/local/bin/qdrant

# NOTE: no Ollama here. It was installed solely to serve the dense embedding model, and
# FastEmbed ships nomic-embed-text-v1.5 as ONNX (run in-process alongside the BM25 sparse
# encoder and the reranker). Removing it drops the install layer (~28 MB after stripping the
# 2.07 GB of unusable CUDA/Vulkan runners), the `zstd` build dep the installer needed, the
# resident server process, and the boot-blocking `ollama pull nomic-embed-text` (~274 MB on
# first run). Generation and the Ragas judge both use Ollama *Cloud* over HTTPS.

# --- Non-root runtime user ---
# Everything (supervisord + ollama/qdrant/redis/python) runs as `librarian`. The entrypoint
# starts as root only to chown the bind-mounted volumes, then drops privileges via gosu.
RUN useradd -r -u 1000 -d /home/librarian -m -s /usr/sbin/nologin librarian

# --- Python app ---
WORKDIR /app

# Eval-only dependencies (ragas + datasets) are an OPTIONAL extra, not a runtime dep: they
# drag in pyarrow (156 MB), pandas (79 MB), scikit-network (35 MB), langchain-community,
# openai and nltk — ~500 MB the serving container never imports. `rag_eval.py` imports ragas
# lazily inside try/except, so the slim image still imports cleanly and simply reports the
# ragas metrics as skipped. Build with `--build-arg INSTALL_EXTRAS=[evals]` for an eval image.
ARG INSTALL_EXTRAS=""

COPY pyproject.toml README.md ./
COPY src ./src
# `uv` resolves and downloads in parallel — a large win over pip on a slow link, which is the
# bottleneck here. Both cache mounts live in BuildKit, not the image, so they cost nothing in
# final size while letting an interrupted build resume from already-downloaded wheels instead
# of re-fetching from scratch. PIP_NO_CACHE_DIR=1 (set above) still keeps the image layer clean.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=0 pip install uv \
    && uv pip install --system ".${INSTALL_EXTRAS}" \
    && pip uninstall -y uv \
    && find /usr/local/lib/python3.11/site-packages -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chown -R librarian:librarian /app

# --- Supervisord + entrypoint ---
# Our supervisord.conf is a *complete* top-level config (it declares [supervisord],
# [unix_http_server], [rpcinterface:supervisor], [supervisorctl] + all [program:*] blocks).
# Install it as the MAIN config, not a conf.d drop-in: the Debian supervisor package's
# default /etc/supervisor/supervisord.conf already declares those top-level sections and
# `[include]s conf.d/*.conf`, so dropping ours there would re-parse [supervisord] and raise
# configparser.DuplicateSectionError at boot (PID 1 exits, the whole stack never starts).
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
VOLUME ["/vault", "/data"]

# /health probes Qdrant + Redis and returns 503 when either is down, so `docker ps` shows
# (unhealthy) instead of a green container serving against a dead vector store — the exact
# failure mode seen on the first run, where qdrant sat in supervisord BACKOFF unnoticed.
# start-period is generous: the first boot downloads the FastEmbed ONNX models (nomic dense
# ~133 MB + BM25 + cross-encoder) into /data before the stack settles.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]