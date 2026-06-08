"""Tests de limpieza de Markdown."""

from __future__ import annotations

import os

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.rag.preprocess import clean_markdown, preprocess_text


def test_removes_html_comments() -> None:
    raw = "<!-- Convertido desde PDF -->\n\n# Título\n\nContenido."
    cleaned = clean_markdown(raw)
    assert "<!--" not in cleaned
    assert "# Título" in cleaned
    assert "Contenido" in cleaned


def test_normalizes_markdown_links() -> None:
    raw = "Ver [decreto](https://example.com/dec) para más."
    assert clean_markdown(raw) == "Ver decreto para más."


def test_collapses_excessive_blank_lines() -> None:
    raw = "a\n\n\n\n\nb"
    assert clean_markdown(raw) == "a\n\nb"


def test_removes_citation_artifacts() -> None:
    raw = "Ver 【47†source】 y filecite turn47file47 al final."
    cleaned = clean_markdown(raw)
    assert "【" not in cleaned
    assert "filecite" not in cleaned.lower()


def test_removes_yaml_frontmatter() -> None:
    raw = "---\ntitle: test\n---\n\n# Título\n\nContenido."
    cleaned = clean_markdown(raw)
    assert "title:" not in cleaned
    assert "# Título" in cleaned
    assert "Contenido" in cleaned


def test_preprocess_returns_segment() -> None:
    segments = preprocess_text("# Sección\n\ntexto útil con suficiente longitud.")
    assert len(segments) == 1
    assert "Sección" in segments[0]
