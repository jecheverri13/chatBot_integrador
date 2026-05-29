"""Configuración de retrievers Chroma (umbral de similitud y MMR)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

SUMMARY_KIND = "summary"


def _fetch_count(k: int) -> int:
    return max(k * settings.retriever_fetch_multiplier, settings.retriever_fetch_min)


def _apply_summary_boost(
    pairs: list[tuple],
    k: int,
) -> list[tuple]:
    """Prioriza chunks summary con score base razonable; garantiza al menos uno en top-k."""
    if not pairs or settings.retriever_summary_boost <= 0:
        return pairs

    boosted: list[tuple] = []
    for doc, score in pairs:
        s = float(score)
        if doc.metadata.get("doc_kind") == SUMMARY_KIND and s >= settings.retriever_summary_min_score:
            s = min(1.0, s + settings.retriever_summary_boost)
        boosted.append((doc, s))
    boosted.sort(key=lambda x: x[1], reverse=True)

    top = boosted[:k]
    has_summary = any(d.metadata.get("doc_kind") == SUMMARY_KIND for d, _ in top)
    if has_summary:
        return top

    best_summary: tuple | None = None
    for doc, score in boosted:
        if doc.metadata.get("doc_kind") == SUMMARY_KIND and float(score) >= settings.retriever_score_threshold:
            best_summary = (doc, float(score))
            break
    if best_summary is None or len(top) == 0:
        return top

    doc, score = best_summary
    if any(d.metadata.get("source") == doc.metadata.get("source") for d, _ in top):
        return top
    return [best_summary] + top[: k - 1]


def build_threshold_retriever(
    vectorstore: Chroma,
    k: int,
    score_threshold: float,
):
    """
    Búsqueda por similitud con umbral (coseno / relevance en [0,1]).
    Usar cuando quieras descartar recuperaciones de baja confianza.
    """
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def build_mmr_retriever(
    vectorstore: Chroma,
    k: int,
    fetch_k: int,
    lambda_mult: float,
):
    """
    MMR reduce redundancia entre chunks recuperados.
    Usar cuando el contexto repita las mismas ideas; menos control explícito de umbral.
    """
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def retrieve_with_scores_filtered(
    vectorstore: Chroma,
    question: str,
    k: int,
    score_threshold: float,
) -> tuple[list, list[float]]:
    """
    Recupera documentos con scores de relevancia y aplica el mismo criterio de umbral.
    Devuelve listas alineadas (doc, score) tras filtrar por score_threshold.
    """
    fetch = _fetch_count(k)
    logger.info(
        "[retrieval] Inicio | k_final=%d | fetch=%d | score_threshold=%.3f | len(query)=%d",
        k,
        fetch,
        score_threshold,
        len(question),
    )
    t_embed_search = time.perf_counter()
    pairs = vectorstore.similarity_search_with_relevance_scores(question, k=fetch)
    t_search = time.perf_counter() - t_embed_search
    logger.info(
        "[retrieval] similarity_search_with_relevance_scores terminó | candidatos=%d | %.3fs",
        len(pairs),
        t_search,
    )

    ranked = _apply_summary_boost([(d, float(s)) for d, s in pairs], k=fetch)
    filtered: list[tuple] = [(d, s) for d, s in ranked if s >= score_threshold][:k]
    above = sum(1 for _, s in ranked if s >= score_threshold)
    logger.info(
        "[retrieval] Tras umbral: conservados=%d (superaron threshold: %d de %d candidatos)",
        len(filtered),
        above,
        len(ranked),
    )
    if not filtered:
        logger.info("[retrieval] Sin hits | ningún chunk superó score_threshold=%s", score_threshold)
        return [], []
    docs = [p[0] for p in filtered]
    scores = [p[1] for p in filtered]
    log_retrieval_hits(question, docs, scores)
    return docs, scores


def retrieve_with_mmr(
    vectorstore: Chroma,
    question: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> tuple[list, list[float]]:
    """Recuperación MMR; scores vacíos (LangChain no expone relevancia directa en MMR)."""
    logger.info(
        "[retrieval] MMR | k=%d fetch_k=%d lambda=%.2f",
        k,
        fetch_k,
        lambda_mult,
    )
    retriever = build_mmr_retriever(vectorstore, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
    t0 = time.perf_counter()
    docs = retriever.invoke(question)
    logger.info("[retrieval] MMR terminó en %.3fs | hits=%d", time.perf_counter() - t0, len(docs))
    log_retrieval_hits(question, docs, [])
    return docs, []


def retrieve_documents(
    vectorstore: Chroma,
    question: str,
    k: int,
    score_threshold: float,
    *,
    use_mmr: bool = False,
    mmr_fetch_k: int = 20,
    mmr_lambda: float = 0.7,
) -> tuple[list, list[float]]:
    """Punto único de recuperación: umbral por defecto, MMR si use_mmr=True."""
    if use_mmr:
        return retrieve_with_mmr(
            vectorstore,
            question,
            k=k,
            fetch_k=mmr_fetch_k,
            lambda_mult=mmr_lambda,
        )
    return retrieve_with_scores_filtered(
        vectorstore,
        question,
        k=k,
        score_threshold=score_threshold,
    )


def resolve_retriever_k(question: str, max_k: int) -> int:
    """k fijo por configuración; dynamic_k solo si está habilitado explícitamente."""
    if settings.retriever_use_dynamic_k:
        return dynamic_k(question, max_k=max_k)
    return max_k


def dynamic_k(question: str, max_k: int = 5) -> int:
    """Ajusta k según longitud de la pregunta (legacy; desactivado por defecto)."""
    length = len(question)
    if length < 50:
        return min(2, max_k)
    if length < 150:
        return min(3, max_k)
    return min(4, max_k)


def log_retrieval_hits(question: str, docs: list, scores: list[float]) -> None:
    """Log INFO por cada chunk recuperado (diagnóstico de retrieval)."""
    total_chars = sum(len(d.page_content) for d in docs)
    q_prev = (question[:100] + "…") if len(question) > 100 else question
    logger.info(
        "[retrieval] Resumen hits | num_docs=%d | chars_contexto_total=%d | query_preview=%r",
        len(docs),
        total_chars,
        q_prev,
    )
    for i, doc in enumerate(docs):
        sc = scores[i] if i < len(scores) else None
        src = doc.metadata.get("source", "")
        section = doc.metadata.get("section", doc.metadata.get("heading", ""))
        kind = doc.metadata.get("doc_kind", "")
        src_name = Path(str(src)).name if src else ""
        preview = (doc.page_content[:120] + "…") if len(doc.page_content) > 120 else doc.page_content
        score_str = f"{sc:.4f}" if sc is not None else "n/a"
        logger.info(
            "[retrieval] hit #%d score=%s chars=%d source=%s kind=%s section=%s | %s",
            i + 1,
            score_str,
            len(doc.page_content),
            src_name,
            kind,
            section,
            preview,
        )
