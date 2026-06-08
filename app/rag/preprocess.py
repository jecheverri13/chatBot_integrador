"""Limpieza de Markdown antes del chunking (preserva encabezados y significado)."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Artefactos de conversión PDF / citas tipo 【6†source】, filecite, turn47file47
_CITATION_BRACKET_RE = re.compile(r"【[^】]*】")
_FILECITE_RE = re.compile(r"filecite|turn\d+file\d+", re.I)
_PRIVATE_USE_RE = re.compile(r"[\uE000-\uF8FF]+")


def clean_markdown(text: str) -> str:
    """Elimina ruido de sintaxis MD que no aporta a la recuperación semántica."""
    cleaned = _FRONTMATTER_RE.sub("", text)
    cleaned = _HTML_COMMENT_RE.sub("", cleaned)
    cleaned = _CITATION_BRACKET_RE.sub("", cleaned)
    cleaned = _FILECITE_RE.sub("", cleaned)
    cleaned = _PRIVATE_USE_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MULTI_BLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def preprocess_text(text: str) -> list[str]:
    """Preprocesa el texto y lo devuelve como una lista de segmentos listos para chunking."""
    if not text.strip():
        return []
    cleaned = clean_markdown(text)
    if not cleaned:
        logger.warning("Texto vacío tras limpieza de Markdown")
        return []
    return [cleaned]
