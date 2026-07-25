#!/usr/bin/env bash
# Entrypoint for the single Obsidian-Librarian container.
# Prepares /data subdirs + model caches, fixes volume ownership, then drops to the
# non-root `librarian` user and hands off to supervisord (PID 1).
set -euo pipefail

DATA="${DATA_PATH:-/data}"
VAULT="${VAULT_PATH:-/vault}"
APP_USER="${APP_USER:-librarian}"

mkdir -p "$DATA"/{qdrant,redis,logs,caches} "$VAULT"
# Qdrant wants its working dir to exist.
mkdir -p "$DATA/qdrant/snapshots"

# No OLLAMA_MODELS dir: Ollama was removed from the container (it only ever served the dense
# embedding model, which now runs in-process via FastEmbed). Generation + judge use Ollama Cloud.

# Keep FastEmbed / HuggingFace ONNX model caches (nomic dense + BM25 sparse + cross-encoder
# rerank) on the persistent volume so they're downloaded once, not on every container
# recreation. The app passes /data/caches/fastembed as `cache_dir` to the FastEmbed
# constructors (see config.py); HF_HOME redirects any huggingface_hub traffic for the same
# reason — FastEmbed fetches the nomic ONNX weights through huggingface_hub.
export HF_HOME="${HF_HOME:-$DATA/caches/hf}"
mkdir -p "$DATA/caches/fastembed" "$HF_HOME"

# Bind mounts arrive root-owned; chown them to the runtime user so the non-root supervisord
# and its children (qdrant/redis/python) can read the vault and write state.
chown -R "$APP_USER":"$APP_USER" "$DATA" "$VAULT" 2>/dev/null || true

exec gosu "$APP_USER" "$@"