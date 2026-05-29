"""Cobertura de ingesta: todos los .md del directorio Documentos."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.config import settings
from app.rag.chunking import split_and_enrich
from app.rag.document_loader import load_all_documents_from_dir


def test_load_all_md_files_in_documentos() -> None:
    docs = load_all_documents_from_dir(settings.documents_dir)
    filenames = {Path(str(d.metadata.get("source", ""))).name for d in docs}
    assert len(filenames) >= 4, f"Se esperaban al menos 4 .md, obtuvo: {filenames}"
    assert "guia_productor_lacteo_rag.md" in filenames
    assert "Decreto_616_de_2006.md" in filenames


def test_chunks_have_doc_kind_and_context_prefix() -> None:
    docs = load_all_documents_from_dir(settings.documents_dir)
    chunks = split_and_enrich(docs)
    assert chunks
    kinds = {c.metadata.get("doc_kind") for c in chunks}
    assert "summary" in kinds
    assert "legal_full" in kinds
    assert chunks[0].page_content.startswith("Documento:")


@patch("app.rag.service.Chroma.from_documents")
@patch("app.rag.service.reset_chroma_store")
def test_build_index_reports_four_documents(
    _reset: MagicMock,
    from_docs: MagicMock,
) -> None:
    from app.rag.service import RagService

    from_docs.return_value = MagicMock()
    service = RagService()
    service._embeddings_client = MagicMock()  # noqa: SLF001

    result = service.build_index()
    assert result["documents_processed"] >= 4
    assert result["chunks_indexed"] > 0
    indexed = from_docs.call_args.kwargs.get("documents") or from_docs.call_args[0][0]
    indexed_filenames = {
        d.metadata.get("filename") for d in indexed if d.metadata.get("filename")
    }
    assert len(indexed_filenames) >= 4
