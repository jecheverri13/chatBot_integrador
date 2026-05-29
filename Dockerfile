# Imagen de producción: FastAPI + RAG (Chroma + MiniLM + Gemini)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

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

RUN mkdir -p storage/chroma evaluations .cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
