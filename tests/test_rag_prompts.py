"""Tests de prompts RAG y post-procesado de respuestas."""

from __future__ import annotations

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.rag.prompts import RAG_HUMAN_PROMPT, RAG_SYSTEM_PROMPT
from app.rag.rag_chain import post_process_answer

_VERBOSE_ANSWER = (
    "Puedes almacenar leche cruda antes de venderla por los siguientes periodos:\n\n"
    "*   **Máximo 8 horas** si no está enfriada "
    "(Sección: Guía para Productores Lácteos - Colombia > 6. FAQ para Productores Lácteos > "
    "¿Cuánto tiempo puedo almacenar leche cruda antes de venderla?, "
    "Sección: Preguntas y respuestas útiles para chatbot > "
    "¿En cuánto tiempo debe venderse la leche cruda en zonas excepcionadas?, "
    "Sección: Guía para Productores Lácteos - Colombia > 4. Comercialización de Leche Cruda y Enfriada).\n"
    "*   **Máximo 24 horas** si está enfriada a 4°C ± 2°C "
    "(Sección: Guía para Productores Lácteos - Colombia > 6. FAQ para Productores Lácteos > "
    "¿Cuánto tiempo puedo almacenar leche cruda antes de venderla?)."
)


def test_prompts_do_not_require_inline_citations() -> None:
    assert "Cita la sección" not in RAG_SYSTEM_PROMPT
    assert "(Sección:" not in RAG_HUMAN_PROMPT
    assert "No cites fuentes" in RAG_HUMAN_PROMPT
    assert "NO incluyas referencias" in RAG_SYSTEM_PROMPT


def test_post_process_strips_section_citations() -> None:
    cleaned = post_process_answer(_VERBOSE_ANSWER)
    assert "(Sección:" not in cleaned
    assert "(sección:" not in cleaned.lower()


def test_post_process_preserves_factual_content() -> None:
    cleaned = post_process_answer(_VERBOSE_ANSWER)
    assert "8 horas" in cleaned
    assert "24 horas" in cleaned
    assert "4°C" in cleaned or "4" in cleaned
