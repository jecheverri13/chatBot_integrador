#!/bin/sh
set -e

PORT="${PORT:-8000}"
CHROMA_DIR="${CHROMA_PERSIST_DIR:-storage/chroma}"

if [ "${RUN_INGEST_ON_START:-false}" = "true" ] || [ ! -f "${CHROMA_DIR}/chroma.sqlite3" ]; then
  echo "[entrypoint] Índice Chroma ausente o RUN_INGEST_ON_START=true; ejecutando ingesta…"
  python scripts/ingest.py
fi

echo "[entrypoint] Iniciando uvicorn en 0.0.0.0:${PORT}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}"
