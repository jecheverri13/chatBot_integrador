"""Caché LRU en memoria para respuestas RAG frecuentes."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

_MAX_SIZE = 500
_TTL_SECONDS = 3600  # 1 hora

_store: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
_lock = threading.Lock()


def _key(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def get(question: str) -> dict[str, Any] | None:
    k = _key(question)
    with _lock:
        if k not in _store:
            return None
        value, ts = _store[k]
        if time.time() - ts > _TTL_SECONDS:
            del _store[k]
            return None
        _store.move_to_end(k)
        return value


def set(question: str, result: dict[str, Any]) -> None:  # noqa: A001
    k = _key(question)
    with _lock:
        if k in _store:
            _store.move_to_end(k)
        _store[k] = (result, time.time())
        while len(_store) > _MAX_SIZE:
            _store.popitem(last=False)


def invalidate_all() -> None:
    with _lock:
        _store.clear()


def size() -> int:
    with _lock:
        return len(_store)
