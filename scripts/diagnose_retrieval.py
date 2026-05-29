"""Diagnóstico de índice Chroma y ranking de retrieval."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_chroma import Chroma

from app.config import settings
from app.rag.embeddings import build_local_embeddings
from app.rag.service import chroma_index_ready


def _load_store() -> Chroma:
    if not chroma_index_ready():
        raise SystemExit(
            f"No existe índice en {settings.chroma_persist_dir}. Ejecuta: python scripts/ingest.py"
        )
    return Chroma(
        persist_directory=str(settings.chroma_persist_dir.resolve()),
        embedding_function=build_local_embeddings(),
        collection_name=settings.chroma_collection_name,
    )


def print_index_stats(store: Chroma) -> None:
    coll = store._collection  # noqa: SLF001 — diagnóstico local
    data = coll.get(include=["metadatas"])
    metas = data.get("metadatas") or []
    ids = data.get("ids") or []
    print(f"\n=== Índice Chroma ({settings.chroma_collection_name}) ===")
    print(f"Total chunks: {len(ids)}")
    by_file: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    for m in metas:
        if not m:
            continue
        fn = m.get("filename") or Path(str(m.get("source", ""))).name or "?"
        by_file[fn] += 1
        by_kind[str(m.get("doc_kind", "?"))] += 1
    print("\nChunks por archivo:")
    for fn, count in by_file.most_common():
        print(f"  {fn}: {count}")
    print("\nChunks por doc_kind:")
    for kind, count in by_kind.most_common():
        print(f"  {kind}: {count}")


def print_retrieval_ranking(store: Chroma, question: str, top_n: int = 15) -> None:
    fetch = max(settings.retriever_k * settings.retriever_fetch_multiplier, settings.retriever_fetch_min)
    pairs = store.similarity_search_with_relevance_scores(question, k=fetch)
    print(f"\n=== Top {top_n} candidatos (fetch={fetch}) ===")
    print(f"Pregunta: {question!r}\n")
    keywords = ("8 horas", "24 horas", "8 hora", "24 hora")
    keyword_hits: list[int] = []
    for i, (doc, score) in enumerate(pairs[:top_n], start=1):
        meta = doc.metadata or {}
        fn = meta.get("filename") or Path(str(meta.get("source", ""))).name
        section = meta.get("section", meta.get("heading", ""))
        kind = meta.get("doc_kind", "")
        preview = doc.page_content[:150].replace("\n", " ")
        if any(kw in doc.page_content.lower() for kw in keywords):
            keyword_hits.append(i)
        print(f"#{i} score={float(score):.4f} | {fn} | kind={kind}")
        print(f"    section: {section}")
        print(f"    preview: {preview}…")
    if keyword_hits:
        print(f"\nChunks con '8/24 horas' en posiciones: {keyword_hits}")
    else:
        print("\nNingún chunk en el top contiene '8 horas' o '24 horas' en el texto.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico RAG: índice y retrieval")
    parser.add_argument(
        "-q",
        "--question",
        default="¿Cuánto tiempo puedo almacenar leche cruda antes de venderla?",
        help="Pregunta de prueba",
    )
    parser.add_argument("-n", "--top", type=int, default=15, help="Candidatos a mostrar")
    parser.add_argument("--stats-only", action="store_true", help="Solo estadísticas del índice")
    args = parser.parse_args()

    store = _load_store()
    print_index_stats(store)
    if not args.stats_only:
        print_retrieval_ranking(store, args.question, top_n=args.top)


if __name__ == "__main__":
    main()
