# Imagen de producción: FastAPI + RAG (Chroma + MiniLM + Gemini)
# Compatible con Render (PORT dinámico) y docker compose local.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=8000 \
    EMBEDDING_DEVICE=cpu \
    CHROMA_PERSIST_DIR=storage/chroma

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Capa cacheable: dependencias antes del código
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY Documentos/ ./Documentos/

RUN sed -i 's/\r$//' scripts/docker-entrypoint.sh \
    && chmod +x scripts/docker-entrypoint.sh

# Índice Chroma y caché HF embebidos en la imagen (Render usa disco efímero).
# GEMINI_API_KEY dummy solo para importar Settings durante la ingesta de build.
RUN mkdir -p storage/chroma evaluations .cache/huggingface \
    && GEMINI_API_KEY=build-placeholder EMBEDDING_DEVICE=cpu python scripts/ingest.py \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD sh -c 'curl -f "http://127.0.0.1:${PORT:-8000}/health" || exit 1'

ENTRYPOINT ["/bin/sh", "scripts/docker-entrypoint.sh"]
