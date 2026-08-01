"""Gemini access.

One place for client construction, retries, usage accounting, and the
structured-output helper. Nothing else in the codebase talks to the SDK.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app import statestore
from app.config import get_settings
from app.db import session_scope

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 4
BASE_DELAY = 1.5
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

_client: genai.Client | None = None
_client_lock = threading.Lock()


class LLMUnavailableError(RuntimeError):
    """No API key configured, or the model is unreachable after retries."""


def get_client() -> genai.Client:
    global _client
    with _client_lock:
        if _client is None:
            key = get_settings().gemini_api_key
            if not key:
                raise LLMUnavailableError(
                    "GEMINI_API_KEY is not set. Copy backend/.env.example to "
                    "backend/.env and add a key from https://aistudio.google.com/apikey"
                )
            _client = genai.Client(api_key=key)
        return _client


def is_configured() -> bool:
    return bool(get_settings().gemini_api_key)


def reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def merge(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass
class JSONResult:
    parsed: Any
    text: str
    usage: Usage = field(default_factory=Usage)


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return _status_of(exc) in RETRYABLE_STATUS
    # Network-level failures surface as plain OSError/httpx errors.
    return isinstance(exc, (OSError, TimeoutError))


def _with_retries(fn, *, what: str):
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last = exc
            if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last  # pragma: no cover - unreachable


def usage_from(response: Any) -> Usage:
    """Extract token counts from a response or a streaming chunk."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage(calls=1)
    return Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
        + (getattr(meta, "thoughts_token_count", 0) or 0),
        calls=1,
    )


_usage_from = usage_from  # backwards-compatible alias


def record_usage(usage: Usage) -> None:
    """Persist running token totals so cost is visible on /api/health."""
    if usage.calls == 0:
        return
    try:
        with session_scope() as session:
            statestore.increment(session, statestore.LLM_CALLS, usage.calls)
            statestore.increment(session, statestore.TOKENS_PROMPT, usage.prompt_tokens)
            statestore.increment(session, statestore.TOKENS_OUTPUT, usage.output_tokens)
    except Exception:  # noqa: BLE001 - accounting must never break the pipeline
        log.exception("Failed to record token usage")


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
    settings = get_settings()
    client = get_client()
    model_id = model or settings.classifier_model

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
    )

    response = _with_retries(
        lambda: client.models.generate_content(
            model=model_id, contents=prompt, config=config
        ),
        what=f"generate_json({model_id})",
    )

    usage = usage_from(response)
    record_usage(usage)

    parsed = getattr(response, "parsed", None)
    if parsed is None:
        # Schema-constrained responses should always parse; if the SDK didn't
        # populate `.parsed`, fall back to validating the raw text ourselves.
        text = (response.text or "").strip()
        if not text:
            raise LLMUnavailableError(f"Empty response from {model_id}")
        parsed = schema.model_validate_json(text)

    return JSONResult(parsed=parsed, text=response.text or "", usage=usage)


def embed(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str | None = None,
) -> tuple[list[list[float]], Usage]:
    """Embed a batch of texts. Returns (vectors, usage)."""
    if not texts:
        return [], Usage()

    settings = get_settings()
    client = get_client()
    model_id = model or settings.embedding_model

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=settings.embedding_dim,
    )

    response = _with_retries(
        lambda: client.models.embed_content(model=model_id, contents=texts, config=config),
        what=f"embed({model_id})",
    )

    vectors = [list(e.values) for e in (response.embeddings or [])]
    usage = Usage(calls=1)
    record_usage(usage)
    return vectors, usage


def list_models() -> list[str]:
    """Available model ids — useful when the published lineup has moved on."""
    client = get_client()
    return [m.name for m in client.models.list()]
