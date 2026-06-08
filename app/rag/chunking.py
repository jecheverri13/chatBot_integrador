"""Chunking híbrido Markdown: parent-child, FAQ por Q/A, tablas intactas."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHILD_CHUNK_SIZE = 320
CHILD_CHUNK_OVERLAP = 60
MIN_CHUNK_SIZE = 40
PARENT_MAX_CHARS = 1800

# Alias retrocompatible con tests existentes
CHUNK_SIZE = CHILD_CHUNK_SIZE
CHUNK_OVERLAP = CHILD_CHUNK_OVERLAP

HEADERS_SUMMARY = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
HEADERS_LEGAL = HEADERS_SUMMARY + [("####", "h4")]

RECURSIVE_SEPARATORS = ["\n\n", "\n- ", "\n", ". ", " "]

_FAQ_SECTION_RE = re.compile(r"faq", re.I)
_FAQ_PAIR_RE = re.compile(
    r"(?ms)^\*\*Q\d+:\*\*\s*(?P<q>.+?)\s*\n\*\*A:\*\*\s*(?P<a>.+?)(?=^\*\*Q\d+:|\Z)"
)
_TABLE_BLOCK_RE = re.compile(
    r"(?ms)(^\|.+\|\s*\n(?:^\|[-:| ]+\|\s*\n)?(?:^\|.+\|\s*\n)+)"
)
_KEYWORDS_RE = re.compile(r"\*\*Palabras clave:\*\*\s*(.+?)(?:\n\n|\Z)", re.I | re.S)


def _section_path_from_metadata(meta: dict[str, Any]) -> str:
    parts = [meta.get("h1"), meta.get("h2"), meta.get("h3"), meta.get("h4")]
    return " > ".join(str(p) for p in parts if p)


def _extract_keywords(text: str) -> str:
    match = _KEYWORDS_RE.search(text)
    return match.group(1).strip() if match else ""


def _is_faq_section(meta: dict[str, Any], content: str) -> bool:
    section = _section_path_from_metadata(meta) or meta.get("section", "")
    if _FAQ_SECTION_RE.search(section):
        return True
    return bool(_FAQ_PAIR_RE.search(content))


def _make_parent_id(meta: dict[str, Any]) -> str:
    src = str(meta.get("source", ""))
    section = meta.get("section_path") or meta.get("section", "")
    raw = f"{src}:{section}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_embed_text(body: str, meta: dict[str, Any]) -> str:
    """Texto denso para embedding — sin prefijo de archivo."""
    keywords = meta.get("keywords") or _extract_keywords(body)
    if keywords and "keywords" not in meta:
        meta["keywords"] = keywords
    heading = meta.get("heading") or meta.get("h4") or meta.get("h3") or meta.get("h2") or ""
    parts = [p for p in (heading, keywords, body.strip()) if p]
    return "\n".join(parts)


def _truncate_parent(text: str) -> str:
    return text.strip()[:PARENT_MAX_CHARS]


def _split_faq_pairs(text: str, base_meta: dict[str, Any]) -> list[Document]:
    pairs = list(_FAQ_PAIR_RE.finditer(text))
    if not pairs:
        return [Document(page_content=text, metadata=dict(base_meta))]

    parent_content = _truncate_parent(text)
    section_base = _section_path_from_metadata(base_meta)
    out: list[Document] = []

    for match in pairs:
        q = match.group("q").strip()
        a = match.group("a").strip()
        meta = dict(base_meta)
        meta["is_faq"] = True
        meta["chunk_role"] = "faq"
        meta["heading"] = q
        meta["section"] = f"{section_base} > {q}" if section_base else q
        meta["section_path"] = meta["section"]
        meta["parent_content"] = parent_content
        meta["parent_id"] = _make_parent_id(meta)
        body = f"Pregunta: {q}\nRespuesta: {a}"
        out.append(
            Document(
                page_content=body,
                metadata=meta,
            )
        )
    return out


def _split_by_headers(text: str, base_meta: dict[str, Any]) -> list[Document]:
    headers = HEADERS_LEGAL if base_meta.get("doc_kind") == "legal_full" else HEADERS_SUMMARY
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    )
    sections = header_splitter.split_text(text)
    if not sections:
        return [Document(page_content=text, metadata=dict(base_meta))]

    out: list[Document] = []
    for sec in sections:
        meta = dict(base_meta)
        meta.update(sec.metadata)
        section_path = _section_path_from_metadata(meta)
        if section_path:
            meta["section"] = section_path
            meta["section_path"] = section_path
            meta["heading"] = section_path.split(" > ")[-1]
        meta["parent_content"] = _truncate_parent(sec.page_content)
        meta["parent_id"] = _make_parent_id(meta)

        keywords = _extract_keywords(sec.page_content)
        if keywords:
            meta["keywords"] = keywords

        if _is_faq_section(meta, sec.page_content):
            out.extend(_split_faq_pairs(sec.page_content, meta))
        else:
            if base_meta.get("doc_kind") == "legal_full" and meta.get("h4"):
                meta["chunk_role"] = "article"
            out.append(Document(page_content=sec.page_content, metadata=meta))
    return out


def _split_preserving_tables(
    text: str,
    recursive: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[str]:
    """Parte el texto preservando bloques de tabla Markdown como unidades."""
    parts: list[str] = []
    last = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        before = text[last : match.start()]
        if before.strip():
            if len(before.strip()) <= chunk_size:
                parts.append(before.strip())
            else:
                parts.extend(s.strip() for s in recursive.split_text(before) if s.strip())
        parts.append(match.group(1).strip())
        last = match.end()

    tail = text[last:]
    if tail.strip():
        if len(tail.strip()) <= chunk_size:
            parts.append(tail.strip())
        else:
            parts.extend(s.strip() for s in recursive.split_text(tail) if s.strip())
    return parts


def _split_section_into_children(
    sec: Document,
    recursive: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Divide una sección larga en child chunks conservando parent_content."""
    text = sec.page_content.strip()
    parent_content = sec.metadata.get("parent_content") or _truncate_parent(text)
    is_faq = sec.metadata.get("is_faq") is True or sec.metadata.get("chunk_role") == "faq"

    if is_faq or len(text) <= CHILD_CHUNK_SIZE:
        return [sec]

    if _TABLE_BLOCK_RE.search(text):
        raw_parts = _split_preserving_tables(text, recursive, CHILD_CHUNK_SIZE)
    else:
        raw_parts = (
            [text]
            if len(text) <= CHILD_CHUNK_SIZE
            else [s.strip() for s in recursive.split_text(text) if s.strip()]
        )

    children: list[Document] = []
    for part in raw_parts:
        meta = dict(sec.metadata)
        if _TABLE_BLOCK_RE.fullmatch(part.strip()) or part.strip().startswith("|"):
            meta["chunk_role"] = "table"
        elif meta.get("chunk_role") != "article":
            meta["chunk_role"] = "fragment"
        meta["parent_content"] = parent_content
        meta["parent_id"] = _make_parent_id(meta)
        children.append(Document(page_content=part, metadata=meta))
    return children


def split_and_enrich(documents: list[Document]) -> list[Document]:
    """
    Estrategia parent-child:
    1) MarkdownHeaderTextSplitter (# / ## / ### / #### legal)
    2) FAQ split por pares Q/A
    3) RecursiveCharacterTextSplitter en secciones largas (tablas intactas)
    4) page_content = embed_text denso; parent_content en metadata para el LLM
    """
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
        length_function=len,
    )

    section_docs: list[Document] = []
    for doc in documents:
        base_meta = dict(doc.metadata)
        section_docs.extend(_split_by_headers(doc.page_content, base_meta))

    raw_splits: list[Document] = []
    for sec in section_docs:
        raw_splits.extend(_split_section_into_children(sec, recursive))

    discarded = 0
    out: list[Document] = []
    per_source: dict[str, int] = {}

    for doc in raw_splits:
        body = doc.page_content.strip()
        if len(body) < MIN_CHUNK_SIZE:
            discarded += 1
            logger.debug("Chunk descartado (len=%d < %d)", len(body), MIN_CHUNK_SIZE)
            continue

        meta: dict[str, Any] = dict(doc.metadata)
        if "parent_content" not in meta:
            meta["parent_content"] = _truncate_parent(body)
        if "parent_id" not in meta:
            meta["parent_id"] = _make_parent_id(meta)
        if "section_path" not in meta and meta.get("section"):
            meta["section_path"] = meta["section"]

        embed_text = _build_embed_text(body, meta)
        char_count = len(embed_text)
        estimated_tokens = max(1, round(char_count / 3.5))
        if estimated_tokens > 230:
            src = str(meta.get("source", ""))
            logger.warning(
                "Chunk con ~tokens estimados altos (%d) en fuente %s sección=%s",
                estimated_tokens,
                Path(src).name if src else "?",
                meta.get("section", ""),
            )

        src = str(meta.get("source", ""))
        per_source[src] = per_source.get(src, 0) + 1
        meta["chunk_index"] = per_source[src]
        meta["char_count"] = char_count
        meta["estimated_tokens"] = estimated_tokens
        if "section" not in meta and meta.get("heading"):
            meta["section"] = meta["heading"]

        out.append(Document(page_content=embed_text, metadata=meta))

    avg_tokens = (
        sum(d.metadata.get("estimated_tokens", 0) for d in out) / len(out) if out else 0
    )
    logger.info(
        "Chunking MD parent-child: secciones=%d chunks_finales=%d descartados=%d promedio_tokens=%.1f",
        len(section_docs),
        len(out),
        discarded,
        avg_tokens,
    )
    return out
