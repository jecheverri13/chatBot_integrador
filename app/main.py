from __future__ import annotations

import logging
import logging.handlers
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import cache, sessions
from app.guardrails import is_in_scope, out_of_scope_message
from app.schemas import ChatRequest, ChatResponse, SourceRef

if TYPE_CHECKING:
    from app.rag.service import RagService


# ─── Logging con rotación ─────────────────────────────────────────────────────

def _setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    if root.handlers:
        return
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if os.getenv("LOG_TO_FILE", "true").lower() not in ("0", "false", "no"):
        log_file = Path("app.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    logging.getLogger("app").setLevel(logging.INFO)


# ─── Rate Limiting ────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    _setup_logging()
    log = logging.getLogger("app.main")
    log.info("Aplicación iniciada (v2.0.0)")
    yield


app = FastAPI(
    title="Backend Chatbot Productores Lácteos",
    version="2.0.0",
    description="API con restricciones de dominio y RAG sobre documentos normativos.",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_rag_service: RagService | None = None
logger = logging.getLogger(__name__)


def _get_rag_service() -> RagService:
    """Carga diferida: uvicorn abre el puerto antes de importar PyTorch/Chroma."""
    global _rag_service
    if _rag_service is None:
        from app.rag import RagService

        _rag_service = RagService()
    return _rag_service


def _as_page(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cache_size": cache.size(), "active_sessions": sessions.active_sessions()}


@app.post("/ingest")
@limiter.limit("2/hour")
def ingest_documents(request: Request) -> dict:
    t0 = time.perf_counter()
    logger.info("[/ingest] Inicio de indexación…")
    try:
        result = _get_rag_service().build_index()
        cache.invalidate_all()
        logger.info("[/ingest] OK en %.3fs | resultado=%s", time.perf_counter() - t0, result)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/ingest] Fallo a los %.3fs", time.perf_counter() - t0)
        raise HTTPException(
            status_code=500,
            detail="Error durante la indexación de documentos.",
        ) from exc


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")
def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    t_req = time.perf_counter()
    q_preview = payload.question[:200] + ("…" if len(payload.question) > 200 else "")
    logger.info(
        "[/chat] Petición recibida | len_pregunta=%d | preview=%r",
        len(payload.question),
        q_preview,
    )

    if not is_in_scope(payload.question):
        logger.info(
            "[/chat] Guardrail: fuera de alcance (sin RAG/Gemini) | %.3fs",
            time.perf_counter() - t_req,
        )
        return ChatResponse(
            answer=out_of_scope_message(),
            blocked=True,
            reason="out_of_scope",
        )

    # Cache hit: evita llamada a Gemini para preguntas repetidas
    cached = cache.get(payload.question)
    if cached:
        logger.info("[/chat] Cache hit | %.3fs", time.perf_counter() - t_req)
        sid, _ = sessions.get_or_create(payload.session_id)
        sessions.append_turn(sid, payload.question, cached["answer"])
        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceRef(**s) for s in (cached.get("sources") or [])],
            scores=cached.get("scores") or None,
            session_id=sid,
        )

    sid, history_turns = sessions.get_or_create(payload.session_id)
    history = sessions.format_history(history_turns)

    logger.info("[/chat] Guardrail: en alcance; iniciando RAG… | session=%s", sid)
    try:
        t_rag = time.perf_counter()
        out = _get_rag_service().query(payload.question, history=history)
        logger.info(
            "[/chat] RAG completado en %.3fs (total: %.3fs)",
            time.perf_counter() - t_rag,
            time.perf_counter() - t_req,
        )
        raw_sources = out.get("sources") or []
        sources = [
            SourceRef(
                source=str(s.get("source", "")),
                section=s.get("section") if s.get("section") else None,
                page=_as_page(s.get("page")),
                preview=str(s.get("preview", "")),
            )
            for s in raw_sources
        ]
        scores = out.get("scores") or None
        answer = str(out.get("answer", ""))

        cache.set(payload.question, {"answer": answer, "sources": raw_sources, "scores": scores})
        sessions.append_turn(sid, payload.question, answer)

        logger.info(
            "[/chat] Respuesta serializada | len=%d | fuentes=%d | total=%.3fs",
            len(answer),
            len(sources),
            time.perf_counter() - t_req,
        )
        return ChatResponse(
            answer=answer,
            sources=sources or None,
            scores=scores,
            session_id=sid,
        )
    except FileNotFoundError as exc:
        logger.warning("[/chat] Índice ausente a los %.3fs", time.perf_counter() - t_req)
        raise HTTPException(
            status_code=400,
            detail="El índice de documentos no está disponible. Ejecuta POST /ingest primero.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("[/chat] Error tras %.3fs", time.perf_counter() - t_req)
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar la consulta.",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
