"""Tests de calidad de retrieval (golden questions)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.rag.chunking import split_and_enrich
from app.rag.document_loader import DOC_KIND_SUMMARY, infer_doc_kind, load_document
from app.rag.embeddings import build_local_embeddings
from app.rag.preprocess import clean_markdown
from app.rag.retriever import retrieve_with_scores_filtered
from app.rag.vectorstore import COLLECTION_METADATA


@pytest.fixture
def golden_chunks() -> list[Document]:
    guia = load_document(
        Path(__file__).resolve().parent.parent / "Documentos" / "guia_productor_lacteo_rag.md"
    )
    d2838 = load_document(
        Path(__file__).resolve().parent.parent / "Documentos" / "decreto_2838_2006_rag.md"
    )
    return split_and_enrich(guia + d2838)


@pytest.fixture
def golden_store(tmp_path: Path, golden_chunks: list[Document]):
    emb = build_local_embeddings()
    persist = tmp_path / "golden_chroma"
    store = Chroma.from_documents(
        documents=golden_chunks,
        embedding=emb,
        persist_directory=str(persist),
        collection_name="golden_test",
        collection_metadata=COLLECTION_METADATA,
    )
    return store


GOLDEN_CASES = [
    (
        "¿Cuánto tiempo puedo almacenar leche cruda antes de venderla?",
        ("guia_productor", "decreto_2838", "normativas"),
        ("8 horas", "24 horas"),
    ),
    (
        "¿Qué suplementos alimenticios están prohibidos para el ganado lechero?",
        ("guia_productor",),
        ("carne", "sangre", "hueso"),
    ),
    (
        "¿A qué temperatura y tiempo se pasteuriza la leche en proceso discontinuo?",
        ("guia_productor", "normativas"),
        ("61", "63", "30"),
    ),
]


@pytest.mark.parametrize("question,expected_files,expected_terms", GOLDEN_CASES)
def test_golden_retrieval_top3(
    golden_store: Chroma,
    question: str,
    expected_files: tuple[str, ...],
    expected_terms: tuple[str, ...],
) -> None:
    docs, scores = retrieve_with_scores_filtered(
        golden_store,
        question,
        k=5,
        score_threshold=0.30,
    )
    assert docs, f"Sin hits para: {question!r}"
    top3 = docs[:3]
    filenames = [str(d.metadata.get("filename", "")) for d in top3]
    assert any(any(exp in fn for fn in filenames) for exp in expected_files), (
        f"Ningún archivo esperado en top-3. Got: {filenames}"
    )
    combined = " ".join(d.page_content.lower() for d in top3)
    assert any(term.lower() in combined for term in expected_terms), (
        f"Términos esperados no encontrados en top-3: {expected_terms}"
    )
    assert scores[0] >= 0.35, f"Score top-1 muy bajo: {scores[0]}"


def test_infer_doc_kind() -> None:
    assert infer_doc_kind("guia_productor_lacteo_rag.md") == DOC_KIND_SUMMARY
    assert infer_doc_kind("Decreto_616_de_2006.md") == "legal_full"


def test_clean_removes_filecite_artifacts() -> None:
    raw = "**Fuente:** filecite turn47file47 【6†source】\n\nTexto útil."
    cleaned = clean_markdown(raw)
    assert "filecite" not in cleaned.lower()
    assert "【" not in cleaned
    assert "Texto útil" in cleaned
