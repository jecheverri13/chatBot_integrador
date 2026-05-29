"""Gestión de historial conversacional por sesión (memoria en proceso)."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

_MAX_TURNS = 10       # turnos máximos por sesión (1 turno = 1 usuario + 1 asistente)
_TTL_SECONDS = 1800   # 30 minutos de inactividad

_sessions: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def get_or_create(session_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Devuelve (session_id, lista_de_turnos). Crea sesión nueva si no existe."""
    sid = session_id or str(uuid.uuid4())
    with _lock:
        _cleanup_expired()
        if sid not in _sessions:
            _sessions[sid] = {"turns": [], "last_active": time.time()}
        else:
            _sessions[sid]["last_active"] = time.time()
        return sid, list(_sessions[sid]["turns"])


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Agrega un par usuario/asistente al historial y aplica el límite de turnos."""
    with _lock:
        if session_id not in _sessions:
            return
        turns = _sessions[session_id]["turns"]
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": answer})
        # Mantener solo los últimos _MAX_TURNS turnos
        if len(turns) > _MAX_TURNS * 2:
            _sessions[session_id]["turns"] = turns[-(_MAX_TURNS * 2):]
        _sessions[session_id]["last_active"] = time.time()


def format_history(turns: list[dict[str, str]], max_turns: int = 5) -> str:
    """Formatea los últimos turnos como texto compacto para el prompt."""
    recent = turns[-(max_turns * 2):]
    if not recent:
        return ""
    lines = []
    for turn in recent:
        role = "Usuario" if turn["role"] == "user" else "Asistente"
        content = turn["content"][:300]  # Truncar para no inflar el contexto
        lines.append(f"{role}: {content}")
    return "Historial de conversación:\n" + "\n".join(lines)


def _cleanup_expired() -> None:
    now = time.time()
    expired = [sid for sid, data in _sessions.items() if now - data["last_active"] > _TTL_SECONDS]
    for sid in expired:
        del _sessions[sid]


def active_sessions() -> int:
    with _lock:
        return len(_sessions)
