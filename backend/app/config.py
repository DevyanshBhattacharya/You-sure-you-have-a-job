"""Application settings, loaded from .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Which backend to use: "gemini" (hosted) or "ollama" (local).
    llm_provider: str = "gemini"

    # Gemini
    gemini_api_key: str = ""
    classifier_model: str = "gemini-2.5-flash"
    qa_model: str = "gemini-2.5-pro"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1536

    # Ollama (local; no key, no quota)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    # Falls back to ollama_model when blank.
    ollama_qa_model: str = ""
    ollama_embedding_model: str = "nomic-embed-text"
    # Local inference on CPU is slow; a per-request minute is not unusual.
    ollama_timeout_seconds: float = 300.0

    # Gmail OAuth
    google_credentials_file: str = "credentials.json"
    google_token_file: str = "token.json"
    # Loopback port for the consent flow. A "Desktop app" OAuth client accepts
    # any port, but a "Web application" client only accepts the exact redirect
    # URIs registered in the Cloud console — so this must match one of them
    # (register `http://localhost:<port>/`, with the trailing slash).
    oauth_redirect_port: int = 8080

    # Storage
    database_url: str = "sqlite:///./data/agent.db"

    # Watcher
    poll_interval_seconds: int = 60
    backfill_default_days: int = 90

    # On startup, re-queue mail that was stored but never classified (e.g. the
    # process stopped mid-backfill). Costs one classifier call per email, so it
    # can be turned off if a large backlog would be an unwelcome surprise.
    process_backlog_on_start: bool = True
    backlog_batch_limit: int = 1000

    # Server
    cors_origins: str = "http://localhost:5173"

    @property
    def credentials_path(self) -> Path:
        return self._resolve(self.google_credentials_file)

    @property
    def token_path(self) -> Path:
        return self._resolve(self.google_token_file)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else BACKEND_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
