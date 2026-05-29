"""Carga recursiva de archivos Markdown (.md) con codificación UTF-8."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.rag.preprocess import preprocess_text

logger = logging.getLogger(__name__)

DOC_KIND_SUMMARY = "summary"
DOC_KIND_LEGAL = "legal_full"


def infer_doc_kind(filename: str) -> str:
    """Clasifica documentos resumidos/FAQ frente a texto legal completo."""
    lower = filename.lower()
    if lower.endswith("_rag.md") or lower.startswith("guia_") or lower.startswith("normativas_"):
        return DOC_KIND_SUMMARY
    if lower.startswith("decreto_") or "decreto" in lower:
        return DOC_KIND_LEGAL
    return DOC_KIND_SUMMARY


def load_document(doc_path: Path) -> list[Document]:
    """Lee un .md, preprocesa y devuelve Documentos LangChain con metadata base."""
    path = doc_path.resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"El archivo no es UTF-8 válido: {path}") from exc

    segments = preprocess_text(text)
    doc_kind = infer_doc_kind(path.name)
    out: list[Document] = []
    for i, content in enumerate(segments):
        out.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "doc_kind": doc_kind,
                    "chunk_index": i,
                },
            )
        )
    if not out:
        logger.warning("Sin texto tras preprocesar: %s", path)
    return out


def load_all_documents_from_dir(docs_dir: Path | None = None) -> list[Document]:
    """Descubre y carga todos los .md del directorio (recursivo por defecto)."""
    base = docs_dir or settings.documents_dir
    if not base.exists():
        raise FileNotFoundError(f"No existe el directorio de documentos: {base}")

    pattern = "**/*.md" if settings.documents_recursive else "*.md"
    paths = sorted(base.glob(pattern))
    if not paths:
        logger.warning("No se encontraron archivos %s en %s", pattern, base)

    all_docs: list[Document] = []
    for doc_path in paths:
        if not doc_path.is_file():
            continue
        try:
            all_docs.extend(load_document(doc_path))
        except ValueError as exc:
            logger.error("Error cargando %s: %s", doc_path, exc)
            raise
    logger.info("Cargados %d documento(s) desde %d archivo(s) .md", len(all_docs), len(paths))
    return all_docs
