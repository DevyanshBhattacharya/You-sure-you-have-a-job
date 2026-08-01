"""Provider registry.

`LLM_PROVIDER` in .env selects the backend; everything else in the codebase
goes through `app.agent.llm`, which delegates here.
"""

from __future__ import annotations

import threading

from app.agent.providers.base import (
    DeferWorkError,
    JSONResult,
    LLMUnavailableError,
    Provider,
    ProviderUnavailableError,
    QuotaExceededError,
    StreamEvent,
    TextChunk,
    ToolCall,
    ToolCallsChunk,
    ToolSpec,
    Turn,
    Usage,
    UsageChunk,
)
from app.config import get_settings

__all__ = [
    "DeferWorkError",
    "JSONResult",
    "LLMUnavailableError",
    "Provider",
    "ProviderUnavailableError",
    "QuotaExceededError",
    "StreamEvent",
    "TextChunk",
    "ToolCall",
    "ToolCallsChunk",
    "ToolSpec",
    "Turn",
    "Usage",
    "UsageChunk",
    "get_provider",
    "reset_provider",
]

_lock = threading.Lock()
_provider: Provider | None = None
_provider_name: str | None = None


def _build(name: str) -> Provider:
    if name == "ollama":
        from app.agent.providers.ollama import OllamaProvider

        return OllamaProvider()
    if name == "gemini":
        from app.agent.providers.gemini import GeminiProvider

        return GeminiProvider()
    raise LLMUnavailableError(
        f"Unknown LLM_PROVIDER {name!r}. Supported values: 'gemini', 'ollama'."
    )


def get_provider() -> Provider:
    """Return the configured backend, rebuilding it if the setting changed."""
    global _provider, _provider_name
    name = (get_settings().llm_provider or "gemini").strip().lower()
    with _lock:
        if _provider is None or _provider_name != name:
            _provider = _build(name)
            _provider_name = name
        return _provider


def reset_provider() -> None:
    global _provider, _provider_name
    with _lock:
        _provider = None
        _provider_name = None
