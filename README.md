# Backend Chatbot para Productores Lácteos (FastAPI + LangChain + Chroma + MiniLM + Gemini)

Backend API en **FastAPI** con **LangChain** (ingesta Markdown, chunking híbrido, ChromaDB con coseno, cadena RAG) y **Google Gemini** solo para el **chat**.

- Restricción de dominio: solo responde sobre contexto lácteo/normativo (`app/guardrails.py`).
- RAG: archivos `.md` en `Documentos/` (recursivo) → limpieza MD → `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` (700/140) → **sentence-transformers/all-MiniLM-L6-v2** (vectores normalizados) → **Chroma** en `storage/chroma/`.
- Embeddings locales: no requieren cuota de embeddings en Gemini; tras cambiar documentos o modelo de embeddings, vuelve a ejecutar la ingesta.

## Estructura del proyecto

```text
ChatBot/
├── app/
│   ├── config.py
│   ├── guardrails.py
│   ├── main.py
│   ├── schemas.py
│   └── rag/
│       ├── service.py        # RagService (ingesta + query)
│       ├── document_loader.py
│       ├── preprocess.py     # Limpieza Markdown
│       ├── chunking.py       # Chunking híbrido por secciones
│       ├── vectorstore.py    # Chroma coseno
│       ├── embeddings.py     # MiniLM
│       ├── retriever.py
│       ├── prompts.py
│       └── rag_chain.py
├── tests/
├── scripts/
│   └── ingest.py
├── Documentos/               # Fuente: *.md
├── storage/chroma/           # generado (ver .gitignore)
├── Dockerfile
├── docker-compose.yml
├── requirements-prod.txt
├── Makefile
├── .env.example
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- API key de Gemini: [Google AI Studio](https://aistudio.google.com/apikey) (solo generación).

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Variables principales: ver [.env.example](.env.example). `CHROMA_PERSIST_DIR` puede ser ruta absoluta o relativa al proyecto.

## Ingesta

Coloca archivos `.md` en `Documentos/` (subcarpetas incluidas si `DOCUMENTS_RECURSIVE=true`).

```powershell
python scripts/ingest.py
```

O con la API en marcha:

```powershell
uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/ingest
```

La ingesta **recrea** la colección Chroma en `storage/chroma/` (o `CHROMA_PERSIST_DIR`).

## Ejecución

```powershell
uvicorn app.main:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado |
| POST | `/ingest` | Reconstruye Chroma desde los `.md` |
| POST | `/chat` | Pregunta (guardrail + RAG); respuesta incluye `sources` (con `section`) y `scores` |

## Retriever

Por defecto: `RETRIEVER_K=5` chunks al LLM, umbral `RETRIEVER_SCORE_THRESHOLD=0.35`, sin `dynamic_k` (siempre k=5).

- `RETRIEVER_FETCH_MIN=40`: candidatos evaluados antes del filtro (mejor cobertura entre archivos).
- `RETRIEVER_SUMMARY_BOOST=0.08`: prioriza guías/FAQ (`*_rag.md`) frente al Decreto 616 masivo.

Para MMR (menos chunks redundantes de la misma sección):

```env
RETRIEVER_USE_MMR=true
RETRIEVER_MMR_FETCH_K=20
RETRIEVER_MMR_LAMBDA=0.7
```

### Interpretación de scores

Los `scores` en `/chat` son relevancia coseno (0–1) entre la pregunta y cada chunk:

| Rango | Significado típico |
|-------|-------------------|
| > 0.80 | Query casi idéntica al texto del chunk |
| 0.45 – 0.65 | Paráfrasis válida (preguntas naturales en español) |
| < 0.35 | Filtrado por `RETRIEVER_SCORE_THRESHOLD` |

No esperes 0.85 en todas las preguntas: *“almacenar leche cruda”* vs *“vender en 8 horas”* suele puntuar ~0.5–0.6 aun siendo la respuesta correcta.

### Diagnóstico de retrieval

```powershell
python scripts/diagnose_retrieval.py
python scripts/diagnose_retrieval.py -q "¿Cuánto tiempo puedo almacenar leche cruda antes de venderla?"
```

Muestra chunks por archivo en el índice y el top-15 de candidatos con scores.

## Despliegue con Docker

Requisitos: [Docker Desktop](https://www.docker.com/products/docker-desktop/) o Docker Engine + Compose v2.

### 1. Configurar entorno

```powershell
copy .env.example .env
# Editar .env y definir GEMINI_API_KEY
```

En Docker se recomienda `EMBEDDING_DEVICE=cpu` (ya es el default en `docker-compose.yml`).

### 2. Construir imagen

```powershell
docker compose build
```

La imagen usa [`requirements-prod.txt`](requirements-prod.txt) (sin pytest/ragas). Tamaño esperado: ~1–3 GB por PyTorch + sentence-transformers.

### 3. Indexar documentos (primera vez o tras cambiar `.md`)

```powershell
docker compose run --rm api python scripts/ingest.py
```

### 4. Levantar API

```powershell
docker compose up -d
docker compose logs -f api
```

- API: http://127.0.0.1:8000  
- Swagger: http://127.0.0.1:8000/docs  

### 5. Verificación

```powershell
curl http://127.0.0.1:8000/health
```

### Volúmenes

| Volumen | Propósito |
|---------|-----------|
| `./storage/chroma` | Índice Chroma persistente |
| `./Documentos` (ro) | Fuentes Markdown del host |
| `hf_cache` | Caché del modelo MiniLM (Hugging Face) |

### Comandos útiles

```powershell
docker compose down
docker compose run --rm api python scripts/diagnose_retrieval.py
docker compose exec api python scripts/ingest.py
```

El contenedor corre como usuario **non-root** (`appuser`, UID 1000). No se copia `.env` dentro de la imagen.

## Despliegue en Render

Render ejecuta el `Dockerfile` como **Web Service** (no usa `docker-compose.yml`). Requisitos clave:

- La API escucha en la variable **`PORT`** que Render asigna automáticamente (el entrypoint `scripts/docker-entrypoint.sh` la usa).
- El índice Chroma y el modelo MiniLM se generan **durante el build** de la imagen (disco efímero en Render).
- Define **`GEMINI_API_KEY`** en el panel de Render (Environment → Add Environment Variable).

### Opción A: Blueprint (`render.yaml`)

1. Sube el repo a GitHub/GitLab.
2. En [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Conecta el repositorio; Render detectará [`render.yaml`](render.yaml).
4. Al crear el servicio, introduce `GEMINI_API_KEY` cuando lo solicite.

### Opción B: Web Service manual

1. **New** → **Web Service** → conecta el repositorio.
2. **Runtime:** Docker.
3. **Dockerfile path:** `./Dockerfile`.
4. **Health Check Path:** `/health`.
5. Variables de entorno mínimas:

| Variable | Valor |
|----------|-------|
| `GEMINI_API_KEY` | Tu API key de Google AI Studio |
| `EMBEDDING_DEVICE` | `cpu` |
| `MODEL_CHAT` | `gemini-2.0-flash` (opcional) |

### Plan y recursos

La imagen incluye PyTorch + sentence-transformers (~1–3 GB). Se recomienda plan **Starter** o superior (≥512 MB RAM; **Standard** si el arranque falla por memoria).

### Reindexar tras cambiar documentos

Tras modificar archivos en `Documentos/`, haz un **nuevo deploy** (rebuild de la imagen). Alternativa sin rebuild: `RUN_INGEST_ON_START=true` en Render (más lento en cada arranque).

### Verificación

```text
GET https://<tu-servicio>.onrender.com/health
GET https://<tu-servicio>.onrender.com/docs
```

## Tests

```powershell
pytest tests/ -q
```

## Migración desde FAISS o índice antiguo

Borra `storage/chroma/` si cambiaste métrica o modelo de embeddings, y ejecuta de nuevo la ingesta.

## Checklist

- Embeddings: `normalize_embeddings=True`, dimensión 384 (MiniLM).
- Chroma: colección con `hnsw:space=cosine`.
- Ingesta: limpieza MD + metadata de sección.
- Chunking: híbrido por encabezados; chunks ≤ 256 tokens (proxy en tests).
- Retriever conectado a cadena RAG; citas por sección en prompts.
