"""Orquestación: ingesta a Chroma + consulta RAG con Gemini."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    ResourceExhausted = ()  # type: ignore[misc, assignment]

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.rag.chunking import split_and_enrich
from app.rag.embeddings import build_local_embeddings
from app.rag.document_loader import load_all_documents_from_dir
from app.rag.rag_chain import query as rag_query
from app.rag.retriever import resolve_retriever_k
from app.rag.vectorstore import COLLECTION_METADATA, reset_chroma_store

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _chroma_sqlite_path() -> Path:
    return settings.chroma_persist_dir / "chroma.sqlite3"


def chroma_index_ready() -> bool:
    return _chroma_sqlite_path().is_file()


class RagService:
    def __init__(self) -> None:
        self._embeddings = None
        self._llm: ChatGoogleGenerativeAI | None = None

    def _embeddings_client(self):
        if self._embeddings is None:
            t0 = time.perf_counter()
            logger.info(
                "[RAG] Primera carga del modelo de embeddings (SentenceTransformer). "
                "Suele mostrar 'Loading weights' y puede tardar 10–60s en CPU la primera vez."
            )
            self._embeddings = build_local_embeddings()
            logger.info(
                "[RAG] Modelo de embeddings listo en %.3fs",
                time.perf_counter() - t0,
            )
        return self._embeddings

    def _llm_client(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            t0 = time.perf_counter()
            logger.info(
                "[RAG] Inicializando ChatGoogleGenerativeAI | model=%s",
                settings.model_chat,
            )
            self._llm = ChatGoogleGenerativeAI(
                model=settings.model_chat,
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm_temperature,
                max_retries=settings.gemini_sdk_max_retries,
            )
            logger.info("[RAG] Cliente Gemini listo en %.3fs", time.perf_counter() - t0)
        return self._llm

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        if ResourceExhausted and isinstance(exc, ResourceExhausted):
            return True
        msg = str(exc).lower()
        if "429" in msg or "resource exhausted" in msg or "rate limit" in msg:
            return True
        return False

    def _with_retry(self, fn: Callable[[], T]) -> T:
        delay = settings.gemini_retry_initial_delay_sec
        last: Exception | None = None
        for attempt in range(settings.gemini_max_retries):
            t_call = time.perf_counter()
            try:
                out = fn()
                elapsed = time.perf_counter() - t_call
                logger.info(
                    "[RAG] Invocación Gemini OK | intento=%d/%d | %.3fs",
                    attempt + 1,
                    settings.gemini_max_retries,
                    elapsed,
                )
                return out
            except Exception as exc:
                last = exc
                if not self._is_rate_limit_error(exc):
                    logger.warning(
                        "[RAG] Invocación Gemini falló (no reintentable) | intento=%d | %.3fs | %s",
                        attempt + 1,
                        time.perf_counter() - t_call,
                        exc,
                    )
                    raise
                if attempt >= settings.gemini_max_retries - 1:
                    logger.error(
                        "[RAG] Gemini rate limit: agotados %d intentos | último error: %s",
                        settings.gemini_max_retries,
                        exc,
                    )
                    break
                sleep_s = min(
                    delay * (2**attempt) + random.uniform(0, 1.0),
                    120.0,
                )
                logger.warning(
                    "[RAG] Gemini 429/rate limit | intento %d/%d | espera %.1fs | error=%s",
                    attempt + 1,
                    settings.gemini_max_retries,
                    sleep_s,
                    exc,
                )
                time.sleep(sleep_s)
        assert last is not None
        raise last

    def _load_vectorstore(self) -> Chroma:
        if not chroma_index_ready():
            raise FileNotFoundError(
                "No existe el índice vectorial en Chroma. Ejecuta el proceso de ingesta primero."
            )
        t0 = time.perf_counter()
        logger.info(
            "[RAG] Abriendo Chroma | dir=%s | collection=%s",
            settings.chroma_persist_dir,
            settings.chroma_collection_name,
        )
        emb = self._embeddings_client()
        logger.info(
            "[RAG] Chroma: embedding_function asignada en %.3fs (objeto listo; búsqueda embeddeará la query)",
            time.perf_counter() - t0,
        )
        t_chroma = time.perf_counter()
        store = Chroma(
            persist_directory=str(settings.chroma_persist_dir.resolve()),
            embedding_function=emb,
            collection_name=settings.chroma_collection_name,
        )
        logger.info(
            "[RAG] Instancia Chroma creada en %.3fs",
            time.perf_counter() - t_chroma,
        )
        return store

    def build_index(self) -> dict[str, Any]:
        docs_dir = settings.documents_dir
        if not docs_dir.exists():
            raise FileNotFoundError(f"No existe el directorio de documentos: {docs_dir}")

        all_docs = load_all_documents_from_dir(docs_dir)
        if not all_docs:
            raise ValueError("No se pudo extraer texto de los documentos.")

        chunks = split_and_enrich(all_docs)
        if not chunks:
            raise ValueError("No quedaron chunks válidos tras el filtrado.")

        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        reset_chroma_store(settings.chroma_persist_dir, settings.chroma_collection_name)

        logger.info("Indexando %d chunks en Chroma…", len(chunks))
        Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings_client(),
            persist_directory=str(settings.chroma_persist_dir.resolve()),
            collection_name=settings.chroma_collection_name,
            collection_metadata=COLLECTION_METADATA,
        )

        sources = {Path(str(d.metadata.get("source", ""))).name for d in all_docs if d.metadata.get("source")}
        return {
            "documents_processed": len(sources),
            "chunks_indexed": len(chunks),
            "vector_store_dir": str(settings.chroma_persist_dir.resolve()),
            "collection": settings.chroma_collection_name,
        }

    def query(self, question: str, history: str = "") -> dict[str, Any]:
        t_all = time.perf_counter()
        k = resolve_retriever_k(question, max_k=settings.retriever_k)
        logger.info(
            "[RAG] query() inicio | k=%d threshold=%.2f | len(pregunta)=%d",
            k,
            settings.retriever_score_threshold,
            len(question),
        )
        t_vs = time.perf_counter()
        store = self._load_vectorstore()
        logger.info("[RAG] Vector store cargado en %.3fs", time.perf_counter() - t_vs)
        t_llm = time.perf_counter()
        llm = self._llm_client()
        logger.info("[RAG] LLM resuelto en %.3fs", time.perf_counter() - t_llm)
        t_rag = time.perf_counter()
        out = rag_query(
            store,
            llm,
            question,
            k=k,
            score_threshold=settings.retriever_score_threshold,
            with_retry=self._with_retry,
            history=history,
            use_mmr=settings.retriever_use_mmr,
            mmr_fetch_k=settings.retriever_mmr_fetch_k,
            mmr_lambda=settings.retriever_mmr_lambda,
        )
        logger.info(
            "[RAG] rag_query (retrieval+Gemini) en %.3fs | query() total=%.3fs",
            time.perf_counter() - t_rag,
            time.perf_counter() - t_all,
        )
        return out

    def answer(self, question: str, history: str = "") -> str:
        return str(self.query(question, history=history).get("answer", "")).strip() or "No se pudo generar respuesta."
