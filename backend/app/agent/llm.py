"""LLM access.

A facade over the configured provider (`LLM_PROVIDER=gemini|ollama`). Keeping
this module's surface stable means the classifier, indexer and pipeline never
learn which backend is in play.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from typing import TypeVar

from pydantic import BaseModel

from app import statestore
from app.agent.providers import get_provider, reset_provider
from app.agent.providers.base import (
    DeferWorkError,
    JSONResult,
    LLMUnavailableError,
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
from app.db import session_scope

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "DeferWorkError",
    "JSONResult",
    "LLMUnavailableError",
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
    "describe",
    "embed",
    "flush_usage",
    "generate_json",
    "pending_usage",
    "is_configured",
    "list_models",
    "provider_name",
    "record_usage",
    "reset_client",
    "stream_turn",
]


def is_configured() -> bool:
    try:
        return get_provider().is_configured()
    except LLMUnavailableError:
        return False


def provider_name() -> str:
    try:
        return get_provider().name
    except LLMUnavailableError:
        return "unconfigured"


def describe() -> dict:
    try:
        return get_provider().describe()
    except LLMUnavailableError as exc:
        return {"provider": "unconfigured", "error": str(exc)}


def reset_client() -> None:
    reset_provider()


def list_models() -> list[str]:
    provider = get_provider()
    lister = getattr(provider, "list_models", None)
    if lister is None:
        return []
    return lister()


_pending = Usage()
_pending_lock = threading.Lock()


def record_usage(usage: Usage) -> None:
    """Accumulate token usage in memory.

    Deliberately does *not* touch the database. Model calls happen inside the
    pipeline's write transaction, and opening a second SQLite connection to
    write counters there deadlocks against it ("database is locked"). Totals
    are flushed by `flush_usage()` once that transaction has closed.
    """
    if usage.calls == 0:
        return
    with _pending_lock:
        _pending.merge(usage)


def pending_usage() -> Usage:
    with _pending_lock:
        return Usage(_pending.prompt_tokens, _pending.output_tokens, _pending.calls)


def flush_usage() -> bool:
    """Write accumulated totals. Safe to call when nothing is pending."""
    with _pending_lock:
        if _pending.calls == 0:
            return False
        snapshot = Usage(_pending.prompt_tokens, _pending.output_tokens, _pending.calls)
        _pending.prompt_tokens = _pending.output_tokens = _pending.calls = 0

    try:
        with session_scope() as session:
            statestore.increment(session, statestore.LLM_CALLS, snapshot.calls)
            statestore.increment(session, statestore.TOKENS_PROMPT, snapshot.prompt_tokens)
            statestore.increment(session, statestore.TOKENS_OUTPUT, snapshot.output_tokens)
    except Exception:  # noqa: BLE001 - accounting must never break the pipeline
        log.exception("Failed to record token usage; will retry on the next flush")
        with _pending_lock:  # put it back rather than losing the count
            _pending.merge(snapshot)
        return False
    return True


def generate_json(
    *,
    prompt: str,
    schema: type[T],
    system_instruction: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> JSONResult:
    """Structured-output call. Returns a parsed instance of `schema`.

    Constraining the response with a schema is what keeps classification
    reliable — parsing free text out of a chat response is the main source of
    flake in this kind of pipeline.
    """
    result = get_provider().generate_json(
        prompt=prompt,
        schema=schema,
        system=system_instruction,
        model=model,
        temperature=temperature,
    )
    record_usage(result.usage)
    return result


def embed(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str | None = None,
) -> tuple[list[list[float]], Usage]:
    """Embed a batch of texts. Returns (vectors, usage)."""
    vectors, usage = get_provider().embed(texts, task_type=task_type, model=model)
    record_usage(usage)
    return vectors, usage


def stream_turn(
    *,
    system: str,
    turns: list[Turn],
    tools: list[ToolSpec],
    model: str | None = None,
    temperature: float = 0.2,
) -> Iterator[StreamEvent]:
    """Stream one assistant turn, which may request tool calls.

    Usage is recorded as the terminating `UsageChunk` passes through.
    """
    for event in get_provider().stream_turn(
        system=system, turns=turns, tools=tools, model=model, temperature=temperature
    ):
        if isinstance(event, UsageChunk):
            record_usage(event.usage)
        yield event
