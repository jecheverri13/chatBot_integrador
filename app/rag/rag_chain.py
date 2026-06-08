"""Cadena RAG (LCEL) con Gemini y post-comprobación de alucinaciones genéricas."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.rag.prompts import RAG_HUMAN_PROMPT, RAG_SYSTEM_PROMPT
from app.rag.retriever import retrieve_documents

logger = logging.getLogger(__name__)

T = TypeVar("T")

_HALLUCINATION_PATTERNS = [
    re.compile(r"como modelo de lenguaje", re.I),
    re.compile(r"como inteligencia artificial", re.I),
    re.compile(r"no tengo acceso", re.I),
    re.compile(r"basándome en mi conocimiento general", re.I),
    re.compile(r"based on my general knowledge", re.I),
]

_SECTION_CITE_RE = re.compile(
    r"\s*\((?:Sección|sección|Fuente|fuente)\s*:[^)]*\)",
    re.I,
)


def has_hallucination_markers(text: str) -> bool:
    return any(p.search(text) for p in _HALLUCINATION_PATTERNS)


def _context_body(doc: Any) -> str:
    return doc.metadata.get("parent_content") or doc.page_content


def format_docs_for_prompt(docs: list[Any]) -> str:
    seen_parents: set[str] = set()
    parts: list[str] = []
    for doc in docs:
        parent_id = doc.metadata.get("parent_id") or doc.metadata.get("section", "")
        if parent_id and parent_id in seen_parents:
            continue
        if parent_id:
            seen_parents.add(parent_id)

        raw_src = doc.metadata.get("source", "")
        src = Path(str(raw_src)).name if raw_src else doc.metadata.get("filename", "")
        section = doc.metadata.get("section") or doc.metadata.get("heading", "")
        page = doc.metadata.get("page")
        loc = f"sección: {section}" if section else ""
        if page is not None and str(page).strip():
            loc = f"{loc}, página {page}".strip(", ")
        loc_suffix = f" ({loc})" if loc else ""
        parts.append(f"Fuente: {src}{loc_suffix}\n{_context_body(doc)}")
    return "\n\n".join(parts)


def post_process_answer(text: str) -> str:
    for pat in _HALLUCINATION_PATTERNS:
        if pat.search(text):
            logger.warning("Patrón de alucinación genérico detectado en la respuesta")
            return (
                "No encontré información suficiente en los documentos para responder esta pregunta."
            )
    cleaned = _SECTION_CITE_RE.sub("", text)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    return cleaned.strip()


def build_prompt_chain(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_HUMAN_PROMPT),
        ]
    )
    return prompt | llm | StrOutputParser()


def query(
    vectorstore: Chroma,
    llm: ChatGoogleGenerativeAI,
    question: str,
    k: int,
    score_threshold: float,
    with_retry: Callable[[Callable[[], T]], T],
    history: str = "",
    *,
    use_mmr: bool = False,
    mmr_fetch_k: int = 15,
    mmr_lambda: float = 0.7,
) -> dict[str, Any]:
    try:
        logger.info(
            "[rag_chain] Inicio query | k=%d score_threshold=%.3f",
            k,
            score_threshold,
        )
        t_ret = time.perf_counter()
        docs, scores = retrieve_documents(
            vectorstore,
            question,
            k=k,
            score_threshold=score_threshold,
            use_mmr=use_mmr,
            mmr_fetch_k=mmr_fetch_k,
            mmr_lambda=mmr_lambda,
        )
        logger.info(
            "[rag_chain] Retrieval listo en %.3fs | num_docs=%d",
            time.perf_counter() - t_ret,
            len(docs),
        )
        if not docs:
            logger.info("[rag_chain] Sin documentos; respuesta fija sin llamar a Gemini")
            return {
                "answer": (
                    "No encontré evidencia suficiente en los documentos cargados para responder "
                    "con precisión. Reformula la pregunta o consulta un tema normativo del sector lácteo."
                ),
                "sources": [],
                "scores": [],
                "contexts": [],
            }

        t_ctx = time.perf_counter()
        context = format_docs_for_prompt(docs)
        logger.info(
            "[rag_chain] Contexto formateado en %.3fs | len_context_chars=%d",
            time.perf_counter() - t_ctx,
            len(context),
        )
        history_block = (history.strip() + "\n\n") if history.strip() else ""
        t_chain = time.perf_counter()
        chain = build_prompt_chain(llm)
        logger.info("[rag_chain] Cadena LCEL construida en %.3fs", time.perf_counter() - t_chain)
        logger.info(
            "[rag_chain] Invocando Gemini (con reintentos si 429) | len_question=%d | con_historial=%s",
            len(question),
            bool(history_block),
        )
        t_llm = time.perf_counter()
        raw = with_retry(
            lambda: chain.invoke({"context": context, "question": question, "history_block": history_block})
        )
        logger.info(
            "[rag_chain] Gemini respondió en %.3fs | len_respuesta_raw=%d",
            time.perf_counter() - t_llm,
            len(raw or ""),
        )
        answer = (raw or "").strip() or "No se pudo generar respuesta."
        answer = post_process_answer(answer)
        contexts = [_context_body(d) for d in docs]
        sources = [
            {
                "source": Path(str(d.metadata.get("source", ""))).name
                or d.metadata.get("filename", ""),
                "section": d.metadata.get("section") or d.metadata.get("heading"),
                "page": d.metadata.get("page"),
                "preview": (d.page_content[:200] + "…") if len(d.page_content) > 200 else d.page_content,
            }
            for d in docs
        ]
        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "contexts": contexts,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error en cadena RAG: %s", exc)
        raise RuntimeError(f"Fallo al ejecutar RAG: {exc}") from exc


def build_rag_chain(_retriever: Any, llm: ChatGoogleGenerativeAI):
    """Compatibilidad con blueprint: retorna la subcadena prompt|llm|parser."""
    return build_prompt_chain(llm)
