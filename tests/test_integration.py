"""Integración Chroma + HuggingFaceEmbeddings (blueprint Fase 1)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Tests deben poder importar app sin .env en CI: variables mínimas
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-imports-only")

from app.rag.embeddings import build_local_embeddings, embedding_dimension_smoke
from app.rag.vectorstore import COLLECTION_METADATA


@pytest.fixture
def tmp_chroma_dir(tmp_path: Path) -> Path:
    d = tmp_path / "chroma_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_embedding_dimension_is_384() -> None:
    emb = build_local_embeddings()
    dim = embedding_dimension_smoke(emb)
    assert dim == 384, f"Se esperaban 384 dimensiones, obtuvo {dim}"


def test_chroma_integration_scores_and_identical_query(tmp_chroma_dir: Path) -> None:
    emb = build_local_embeddings()
    phrase = "Normativa sanitaria para acopio de leche cruda en fincas."
    docs = [
        Document(page_content="Texto irrelevante sobre clima."),
        Document(page_content=phrase),
        Document(page_content="Otro texto sobre logística portuaria."),
    ]
    store = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(tmp_chroma_dir),
        collection_name="integration_test",
        collection_metadata=COLLECTION_METADATA,
    )

    pairs = store.similarity_search_with_relevance_scores(phrase, k=3)
    assert len(pairs) >= 1
    scores = [float(s) for _, s in pairs]
    for s in scores:
        assert 0.0 <= s <= 1.0, f"Score fuera de [0,1]: {s}"

    top_doc, top_score = pairs[0]
    assert phrase in top_doc.page_content or top_doc.page_content == phrase
    assert top_score > 0.8, f"Query idéntica debería tener score > 0.8, obtuvo {top_score}"
