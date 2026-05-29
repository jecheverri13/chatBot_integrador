import re

DAIRY_CONTEXT_KEYWORDS = {
    "leche",
    "lacteo",
    "lacteos",
    "lácteo",
    "lácteos",
    "ordeño",
    "ganado",
    "bovino",
    "vaca",
    "finca",
    "higiene",
    "inocuidad",
    "pasteurizacion",
    "pasteurización",
    "decreto",
    "productor",
    "acopio",
    "sanitario",
}

# Patrones de prompt injection conocidos
_INJECTION_PATTERNS = [
    re.compile(r"ignora\s+(las\s+)?instrucciones", re.I),
    re.compile(r"olvida\s+(tus\s+)?(instrucciones|reglas)", re.I),
    re.compile(r"forget\s+(your\s+)?(instructions|rules|prompt)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"act\s+as\s+if", re.I),
    re.compile(r"pretend\s+you\s+are", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"nuevo\s+rol", re.I),
    re.compile(r"actúa\s+como\s+si", re.I),
]


def normalize_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.lower()).strip()
    return cleaned


def has_injection_pattern(question: str) -> bool:
    return any(p.search(question) for p in _INJECTION_PATTERNS)


def is_in_scope(question: str) -> bool:
    if has_injection_pattern(question):
        return False
    q = normalize_text(question)
    return any(keyword in q for keyword in DAIRY_CONTEXT_KEYWORDS)


def out_of_scope_message() -> str:
    return (
        "Solo puedo responder preguntas sobre producción láctea, "
        "normativa sanitaria y temas relacionados con los documentos cargados."
    )
