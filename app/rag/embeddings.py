"""Embeddings locales all-MiniLM-L6-v2 para Chroma (coseno + vectores normalizados)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    r = requested.strip().lower()
    if r == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if r == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA solicitada pero no disponible; usando cpu")
                return "cpu"
        except ImportError:
            logger.warning("torch no disponible; usando cpu")
            return "cpu"
        return "cuda"
    return "cpu"


def build_local_embeddings() -> HuggingFaceEmbeddings:
    """
    normalize_embeddings=True es OBLIGATORIO con space=cosine en Chroma:
    sin normalización L2, la métrica coseno no coincide con la geometría esperada.
    """
    device = _resolve_device(settings.embedding_device)
    model_kwargs: dict[str, Any] = {"device": device}
    token = (settings.hf_token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        model_kwargs["token"] = token
        logger.info(
            "Embeddings HuggingFace modelo=%s device=%s (HF_TOKEN aplicado para Hub)",
            settings.embedding_model,
            device,
        )
    else:
        logger.info("Embeddings HuggingFace modelo=%s device=%s", settings.embedding_model, device)
    t_load = time.perf_counter()
    logger.info(
        "[embeddings] Construyendo SentenceTransformer (descarga/caché + 'Loading weights')…"
    )
    emb = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info(
        "[embeddings] SentenceTransformer construido en %.3fs",
        time.perf_counter() - t_load,
    )
    return emb


def embedding_dimension_smoke(embeddings: HuggingFaceEmbeddings) -> int:
    """Una oración de prueba; all-MiniLM-L6-v2 debe producir 384 dimensiones."""
    v = embeddings.embed_query("Prueba de dimensionalidad.")
    return len(v)
