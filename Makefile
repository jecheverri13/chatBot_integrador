.PHONY: setup index query reset-db test docker-build docker-up docker-down docker-ingest docker-logs

PY ?= python
COMPOSE ?= docker compose

setup:
	$(PY) -m venv .venv
	.venv/bin/python -m pip install -U pip
	.venv/bin/python -m pip install -r requirements.txt
	@echo "Activa el venv: source .venv/bin/activate  (Linux/macOS) o .venv\\Scripts\\Activate.ps1 (Windows)"

index:
	$(PY) scripts/ingest.py

query:
	@test -n "$(Q)" || (echo 'Uso: make query Q="tu pregunta"' && exit 1)
	$(PY) -c "from app.rag.service import RagService; print(RagService().answer('''$(Q)'''))"

reset-db:
	$(PY) -c "import shutil; from pathlib import Path; from app.config import settings; p=settings.chroma_persist_dir; shutil.rmtree(p, ignore_errors=True); print('Chroma eliminado:', p)"

test:
	$(PY) -m pytest tests -v

docker-build:
	$(COMPOSE) build

docker-up:
	$(COMPOSE) up -d

docker-down:
	$(COMPOSE) down

docker-ingest:
	$(COMPOSE) run --rm api python scripts/ingest.py

docker-logs:
	$(COMPOSE) logs -f api
