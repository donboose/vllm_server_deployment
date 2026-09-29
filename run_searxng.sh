#!/usr/bin/env bash

# SearXNG metasearch — the free, keyless backend for dsh's web_search tool.
#
# Run on the shared server (or any machine with docker + outbound internet).
# Idempotent: re-running (re)creates the container; nothing is lost because
# all state lives in the project-local data dir.
#
#   ./run_searxng.sh                    # host port 8888
#   SEARXNG_PORT=9000 ./run_searxng.sh  # custom port
#
# Everything stays inside the project (shared server):
#   .searxng/settings.yml  — instance config (written ONCE; edit to trim engines)
#   .searxng/.secret       — secret_key (generated once, owner-only)
#
# Env overrides:
#   SEARXNG_PORT    host port (default 8888)
#   SEARXNG_IMAGE   image (default searxng/searxng)
#   SEARXNG_NAME    container name (default searxng)

set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

PORT="${SEARXNG_PORT:-8888}"
IMAGE="${SEARXNG_IMAGE:-searxng/searxng}"
NAME="${SEARXNG_NAME:-searxng}"
DATA="$PROJECT_ROOT/.searxng"

command -v docker >/dev/null 2>&1 || { echo "docker not found on PATH" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not found on PATH" >&2; exit 1; }

# Dedicated runtime dir, world-writable so the container's user can write
# while the user keeps ownership of settings.yml / .secret (no chown ping-
# pong: FORCE_OWNERSHIP=false below stops the entrypoint from chowning it).
mkdir -p "$DATA"
chmod 777 "$DATA"

# --- secret_key (generated once, stored in the project, owner-only) ---
SECRET_FILE="$DATA/.secret"
if [ ! -f "$SECRET_FILE" ]; then
  openssl rand -hex 32 > "$SECRET_FILE"
  chmod 600 "$SECRET_FILE"
  echo "generated secret_key -> $SECRET_FILE"
fi

# --- settings.yml (written once; user-editable, e.g. to trim engines) ---
# No port/bind_address needed: the image listens on :::8080 internally.
# limiter: false => no Redis/valkey sidecar, no bot-detection of our own clients.
if [ ! -f "$DATA/settings.yml" ]; then
  cat > "$DATA/settings.yml" <<EOF
use_default_settings: true

general:
  instance_name: "SearXNG internal"

server:
  secret_key: "$(cat "$SECRET_FILE")"
  limiter: false

search:
  formats:
    - html
    - json
EOF
  echo "wrote $DATA/settings.yml"
else
  echo "keeping existing $DATA/settings.yml (edit it to trim engines)"
fi

# --- (re)create the container ---
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  -e FORCE_OWNERSHIP=false \
  -p "$PORT:8080" \
  -v "$DATA:/etc/searxng" \
  "$IMAGE" >/dev/null
echo "container '$NAME' running on host port $PORT"

# --- self-test the JSON API (output goes to a temp file, not the volume) ---
SELFTEST="$(mktemp)"
trap 'rm -f "$SELFTEST"' EXIT
ok=""
for _ in $(seq 1 15); do
  if curl -fs -m 5 "http://127.0.0.1:$PORT/search?q=searxng&format=json" > "$SELFTEST" 2>/dev/null; then
    ok=1; break
  fi
  sleep 1
done
if [ -n "$ok" ]; then
  echo "self-test OK:"
  head -c 300 "$SELFTEST"; echo
  IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [ -n "$IP" ] && echo "clients: use SEARXNG_URL=http://$IP:$PORT"
else
  echo "self-test FAILED — check: docker logs $NAME" >&2
  exit 1
fi
