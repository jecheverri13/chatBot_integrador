"""Cliente Chroma persistente y colección con similitud coseno (HNSW)."""

from __future__ import annotations

import logging
from pathlib import Path
import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

# ⚠️ ADVERTENCIA: no cambiar sin revisar embeddings y normalización L2.
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}


def get_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    path = str(persist_dir.resolve())
    logger.info("Chroma PersistentClient path=%s", path)
    return chromadb.PersistentClient(path=path)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> Collection:
    col = client.get_or_create_collection(
        name=collection_name,
        metadata=COLLECTION_METADATA,
    )
    logger.info(
        "Colección Chroma name=%s metadata=%s",
        collection_name,
        col.metadata,
    )
    return col


def delete_collection_if_exists(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> None:
    try:
        client.delete_collection(collection_name)
        logger.info("Colección eliminada: %s", collection_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("No se pudo eliminar colección %s: %s", collection_name, exc)


def reset_chroma_store(persist_dir: Path, collection_name: str) -> None:
    """Borra la colección para re-indexación limpia (persist_dir se conserva)."""
    client = get_chroma_client(persist_dir)
    delete_collection_if_exists(client, collection_name)
