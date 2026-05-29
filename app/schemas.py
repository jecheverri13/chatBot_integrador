from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000, description="Pregunta del usuario")
    session_id: str | None = Field(None, description="ID de sesión para historial conversacional")


class SourceRef(BaseModel):
    source: str = ""
    section: str | None = None
    page: int | None = None
    preview: str = ""


class ChatResponse(BaseModel):
    answer: str
    blocked: bool = False
    reason: str | None = None
    sources: list[SourceRef] | None = None
    scores: list[float] | None = None
    session_id: str | None = None
