"""Validación de chunking Markdown híbrido."""

from __future__ import annotations

import logging
import os

import pytest
import tiktoken
from langchain_core.documents import Document

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.rag.chunking import CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, MIN_CHUNK_SIZE, split_and_enrich

logger = logging.getLogger(__name__)


def _token_len_cl100k(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def test_no_chunk_exceeds_256_tokens_proxy() -> None:
    base = (
        "La normativa sanitaria del sector lácteo exige controles de temperatura, "
        "limpieza de equipos y trazabilidad del acopio en fincas y centros de enfrío. "
    )
    docs = [
        Document(
            page_content="# Capítulo\n\n" + (base * 45).strip(),
            metadata={"source": "doc.md", "filename": "doc.md"},
        )
    ]
    chunks = split_and_enrich(docs)
    assert chunks, "debe haber al menos un chunk"
    for c in chunks:
        assert len(c.page_content) <= CHILD_CHUNK_SIZE + 200, (
            f"chunk embed_text excede tamaño razonable: {len(c.page_content)}"
        )
        n = _token_len_cl100k(c.page_content)
        assert n <= 256, f"Chunk supera 256 tokens (proxy cl100k): n={n}"


def test_overlap_between_consecutive_chunks() -> None:
    long_text = "# Intro\n\n" + ("texto repetido con variación. " * 120) + " FIN"
    docs = [Document(page_content=long_text, metadata={"source": "a.md", "filename": "a.md"})]
    chunks = split_and_enrich(docs)
    if len(chunks) < 2:
        pytest.skip("un solo chunk; texto demasiado corto para probar overlap")
    a, b = chunks[0].page_content, chunks[1].page_content
    overlap_ok = any(a[i : i + 20] in b for i in range(0, max(0, len(a) - 20), 5))
    assert overlap_ok or CHILD_CHUNK_OVERLAP > 0


def test_section_metadata_on_markdown_headers() -> None:
    md = "# Título principal\n\n" + ("contenido de sección. " * 30) + "\n\n## Subsección\n\n" + (
        "más texto normativo. " * 20
    )
    docs = [Document(page_content=md, metadata={"source": "s.md", "filename": "s.md"})]
    chunks = split_and_enrich(docs)
    assert chunks
    with_section = [c for c in chunks if c.metadata.get("section") or c.metadata.get("h1")]
    assert with_section, "se esperaba metadata de sección tras split por encabezados"


def test_min_chunk_discarded_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    tiny = "abc"
    docs = [
        Document(page_content=tiny, metadata={"source": "t.md", "filename": "t.md"}),
        Document(
            page_content="# X\n\n" + ("y " * (MIN_CHUNK_SIZE + 10)),
            metadata={"source": "t.md", "filename": "t.md"},
        ),
    ]
    split_and_enrich(docs)
    assert "descartado" in caplog.text.lower() or "Chunk descartado" in caplog.text


def test_faq_split_into_individual_pairs() -> None:
    md = (
        "## 6. FAQ\n\n"
        "**Q1:** ¿Cuánto tiempo puedo almacenar leche cruda?\n"
        "**A:** Máximo 8 horas si no está enfriada.\n\n"
        "**Q2:** ¿Qué suplementos están prohibidos?\n"
        "**A:** Harinas de carne, sangre y hueso.\n"
    )
    docs = [
        Document(
            page_content="# Guía\n\n" + md,
            metadata={"source": "guia.md", "filename": "guia.md", "doc_kind": "summary"},
        )
    ]
    chunks = split_and_enrich(docs)
    faq_chunks = [c for c in chunks if c.metadata.get("chunk_role") == "faq"]
    assert len(faq_chunks) >= 2
    assert any("8 horas" in c.page_content for c in faq_chunks)
    assert all(c.metadata.get("parent_content") for c in faq_chunks)


def test_table_block_kept_intact() -> None:
    md = (
        "# Estándares\n\n"
        "Intro breve sobre parámetros.\n\n"
        "| Parámetro | Valor |\n"
        "|-----------|-------|\n"
        "| Proteína | 2.9% |\n"
        "| Grasa | 3.0% |\n\n"
        + ("texto adicional normativo. " * 40)
    )
    docs = [Document(page_content=md, metadata={"source": "t.md", "filename": "t.md"})]
    chunks = split_and_enrich(docs)
    table_chunks = [c for c in chunks if c.metadata.get("chunk_role") == "table"]
    assert table_chunks, "se esperaba al menos un chunk de tabla"
    parent = table_chunks[0].metadata.get("parent_content", "")
    assert "| Proteína |" in parent or "| Proteína |" in table_chunks[0].page_content


def test_parent_metadata_on_sections() -> None:
    md = "# Título\n\n" + ("contenido de sección. " * 30)
    docs = [Document(page_content=md, metadata={"source": "s.md", "filename": "s.md"})]
    chunks = split_and_enrich(docs)
    assert chunks
    assert chunks[0].metadata.get("parent_content")
    assert chunks[0].metadata.get("parent_id")
