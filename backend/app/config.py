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
    # Context window, in tokens. This must be set explicitly: Ollama defaults to
    # 4096 regardless of what the model supports, and silently drops the oldest
    # tokens past that — which is the *start* of the prompt, i.e. the system
    # instruction and the question. The failure is invisible (no error, no
    # warning), and looks like the model ignoring its instructions. A full email
    # or a tool result listing every application clears 4096 easily.
    ollama_num_ctx: int = 16384
    # Context for schema-constrained extraction, which is a different workload:
    # the prompt is capped (classify.MAX_PROMPT_BODY_CHARS) at roughly 2k tokens
    # and the output is a small object, so the chat window above is pure waste
    # here — and not harmless waste. The KV cache for a window is allocated in
    # VRAM next to the weights, so an oversized one evicts layers to the CPU.
    # Measured on qwen3:4b (5.1 GB): at 16384 only 2.6 GB stayed resident and a
    # single classification ran past ten minutes; sized to the prompt, the whole
    # model stays on the GPU. Raise this only if truncation warnings appear.
    ollama_extraction_num_ctx: int = 6144
    # Reasoning during *chat* (the Q&A agent). Keep it on for models that
    # support it: setting this false does not stop a reasoning model reasoning,
    # it stops Ollama separating the reasoning into its own `thinking` field, so
    # it lands in `content` and is shown to the user as the answer. Verified on
    # Ollama 0.32.5 with qwen3:4b. Only set it false for a non-reasoning model.
    #
    # It does not apply to schema-constrained extraction, where reasoning is
    # always off — see the note in providers/ollama.py:generate_json.
    ollama_think: bool = True

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

    # How often to re-sweep for stored-but-unclassified mail. The startup sweep
    # is capped at `backlog_batch_limit`, and anything queued when the process
    # stops is lost with the in-memory queue; without a repeating sweep those
    # rows sit unprocessed until someone restarts the server. Set 0 to disable.
    backlog_sweep_seconds: int = 120

    # Restart an import that a crash or a dropped connection left unfinished.
    # Messages already stored are skipped, so this resumes rather than repeats.
    resume_backfill_on_start: bool = True

    # Run the first import automatically, so the app fills itself in rather than
    # waiting for someone to find the "Import mail" button. Only fires when no
    # import has ever completed; after that the watcher keeps things current.
    auto_backfill_on_start: bool = True

    # Server
    cors_origins: str = "http://localhost:5173"

    # Shared secret guarding every /api and /ws route. Empty means no auth,
    # which is only safe bound to loopback — this app serves the full text of a
    # mailbox and will summarise it on request. Generate one with:
    #   python -c "import secrets; print(secrets.token_urlsafe(32))"
    app_auth_token: str = ""

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
