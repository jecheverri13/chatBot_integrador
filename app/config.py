from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    model_chat: str = "gemini-2.0-flash"

    project_root: Path = Path(__file__).resolve().parent.parent
    documents_dir: Path = project_root / "Documentos"
    documents_recursive: bool = True
    chroma_persist_dir: Path = project_root / "storage" / "chroma"
    chroma_collection_name: str = "rag_docs"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "auto"  # auto | cpu | cuda
    hf_token: str | None = None  # HF_TOKEN en .env: acceso al Hub de Hugging Face (opcional)

    retriever_k: int = 5
    retriever_score_threshold: float = 0.35
    retriever_use_dynamic_k: bool = False
    retriever_fetch_multiplier: int = 8
    retriever_fetch_min: int = 40
    retriever_summary_boost: float = 0.08
    retriever_summary_min_score: float = 0.40
    retriever_use_mmr: bool = False
    retriever_mmr_fetch_k: int = 20
    retriever_mmr_lambda: float = 0.7

    gemini_max_retries: int = 2
    gemini_retry_initial_delay_sec: float = 2.0
    gemini_sdk_max_retries: int = 1

    llm_temperature: float = 0.0

    evaluations_dir: Path = project_root / "evaluations"

    @field_validator("gemini_max_retries")
    @classmethod
    def cap_gemini_max_retries(cls, v: int) -> int:
        """Máximo 2 intentos ante 429 en la capa de aplicación (independiente del .env)."""
        if v < 1:
            return 1
        return min(v, 2)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> Settings:
        root = self.project_root
        for name in ("chroma_persist_dir", "documents_dir", "evaluations_dir"):
            p = getattr(self, name)
            if isinstance(p, Path) and not p.is_absolute():
                setattr(self, name, (root / p).resolve())
        return self


settings = Settings()
