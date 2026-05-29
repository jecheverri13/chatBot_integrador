"""Chunking híbrido Markdown: secciones por encabezado + recursive en bloques largos."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 140
MIN_CHUNK_SIZE = 100

HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

MD_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", ", ", " ", ""]

_FAQ_SECTION_RE = re.compile(r"faq", re.I)


def _section_path_from_metadata(meta: dict[str, Any]) -> str:
    parts = [meta.get("h1"), meta.get("h2"), meta.get("h3")]
    return " > ".join(str(p) for p in parts if p)


def _is_faq_section(meta: dict[str, Any], content: str) -> bool:
    section = _section_path_from_metadata(meta) or meta.get("section", "")
    if _FAQ_SECTION_RE.search(section):
        return True
    return bool(re.search(r"\*\*Q\d+:", content))


def _prepend_context_prefix(text: str, meta: dict[str, Any]) -> str:
    filename = meta.get("filename") or Path(str(meta.get("source", ""))).name or ""
    section = meta.get("section") or meta.get("heading") or ""
    prefix = f"Documento: {filename}"
    if section:
        prefix += f" | Sección: {section}"
    return f"{prefix}\n\n{text}"


def _split_by_headers(text: str, base_meta: dict[str, Any]) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    if not sections:
        return [Document(page_content=text, metadata=dict(base_meta))]
    out: list[Document] = []
    for sec in sections:
        meta = dict(base_meta)
        meta.update(sec.metadata)
        section_path = _section_path_from_metadata(sec.metadata)
        if section_path:
            meta["section"] = section_path
            meta["heading"] = section_path.split(" > ")[-1]
        if _is_faq_section(meta, sec.page_content):
            meta["is_faq"] = True
        out.append(Document(page_content=sec.page_content, metadata=meta))
    return out


def split_and_enrich(documents: list[Document]) -> list[Document]:
    """
    Estrategia híbrida:
    1) MarkdownHeaderTextSplitter por # / ## / ###
    2) RecursiveCharacterTextSplitter si la sección supera CHUNK_SIZE (excepto FAQ cortos)
    3) Prefijo contextual en page_content para mejorar embeddings
    """
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=MD_SEPARATORS,
        length_function=len,
    )

    section_docs: list[Document] = []
    for doc in documents:
        base_meta = dict(doc.metadata)
        section_docs.extend(_split_by_headers(doc.page_content, base_meta))

    raw_splits: list[Document] = []
    for sec in section_docs:
        is_faq = sec.metadata.get("is_faq") is True
        if is_faq or len(sec.page_content) <= CHUNK_SIZE:
            raw_splits.append(sec)
        else:
            raw_splits.extend(recursive.split_documents([sec]))

    discarded = 0
    out: list[Document] = []
    per_source: dict[str, int] = {}

    for doc in raw_splits:
        text = doc.page_content.strip()
        if len(text) < MIN_CHUNK_SIZE:
            discarded += 1
            logger.debug("Chunk descartado (len=%d < %d)", len(text), MIN_CHUNK_SIZE)
            continue

        src = str(doc.metadata.get("source", ""))
        per_source[src] = per_source.get(src, 0) + 1
        idx = per_source[src]
        enriched_text = _prepend_context_prefix(text, doc.metadata)
        char_count = len(enriched_text)
        estimated_tokens = max(1, round(char_count / 3.5))
        if estimated_tokens > 230:
            logger.warning(
                "Chunk con ~tokens estimados altos (%d) en fuente %s sección=%s",
                estimated_tokens,
                Path(src).name if src else "?",
                doc.metadata.get("section", ""),
            )

        meta: dict[str, Any] = dict(doc.metadata)
        meta["chunk_index"] = idx
        meta["char_count"] = char_count
        meta["estimated_tokens"] = estimated_tokens
        if "section" not in meta and meta.get("heading"):
            meta["section"] = meta["heading"]
        out.append(Document(page_content=enriched_text, metadata=meta))

    avg_tokens = (
        sum(d.metadata.get("estimated_tokens", 0) for d in out) / len(out) if out else 0
    )
    logger.info(
        "Chunking MD híbrido: secciones=%d chunks_finales=%d descartados=%d promedio_tokens=%.1f",
        len(section_docs),
        len(out),
        discarded,
        avg_tokens,
    )
    return out
