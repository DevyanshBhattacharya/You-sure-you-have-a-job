"""Gemini backend (google-genai)."""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.agent.providers.base import (
    JSONResult,
    LLMUnavailableError,
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

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
BASE_DELAY = 1.5
RETRYABLE_STATUS = {408, 500, 502, 503, 504}

# Retry a 429 in-line only when the server says the wait is short. Free-tier
# limits are frequently per-day, where sleeping is pure waste.
MAX_INLINE_QUOTA_WAIT = 30.0

_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")
_QUOTA_VALUE = re.compile(r"quotaValue':\s*'(\d+)'")


def parse_quota_error(message: str) -> tuple[float | None, int | None]:
    """Pull (retry_after_seconds, daily_limit) out of a 429 body."""
    delay = _RETRY_DELAY.search(message)
    limit = _QUOTA_VALUE.search(message)
    return (
        float(delay.group(1)) if delay else None,
        int(limit.group(1)) if limit else None,
    )


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_quota(exc: Exception) -> bool:
    return _status_of(exc) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    if _is_quota(exc):
        return False  # handled separately in _with_retries
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return _status_of(exc) in RETRYABLE_STATUS
    return isinstance(exc, (OSError, TimeoutError))


def _with_retries(fn, *, what: str):
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below if not retryable
            last = exc

            if _is_quota(exc):
                retry_after, limit = parse_quota_error(str(exc))
                if retry_after is not None and retry_after <= MAX_INLINE_QUOTA_WAIT:
                    log.warning("%s rate-limited; waiting %.0fs", what, retry_after)
                    time.sleep(retry_after + 0.5)
                    continue
                raise QuotaExceededError(
                    f"{what}: quota exhausted"
                    + (f" (limit {limit}/day)" if limit else "")
                    + (f"; retry in {retry_after:.0f}s" if retry_after else ""),
                    retry_after=retry_after,
                ) from exc

            if attempt == MAX_ATTEMPTS or not _is_retryable(exc):
                raise

            delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            log.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                what,
                attempt,
                MAX_ATTEMPTS,
                str(exc)[:160],
                delay,
            )
            time.sleep(delay)
    raise last  # pragma: no cover - unreachable


def usage_from(response: Any) -> Usage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage(calls=1)
    return Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=(getattr(meta, "candidates_token_count", 0) or 0)
        + (getattr(meta, "thoughts_token_count", 0) or 0),
        calls=1,
    )


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self._client: genai.Client | None = None
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    def client(self) -> genai.Client:
        with self._lock:
            if self._client is None:
                key = get_settings().gemini_api_key
                if not key:
                    raise LLMUnavailableError(
                        "GEMINI_API_KEY is not set. Copy backend/.env.example to "
                        "backend/.env and add a key from https://aistudio.google.com/apikey"
                    )
                self._client = genai.Client(api_key=key)
            return self._client

    def reset(self) -> None:
        with self._lock:
            self._client = None

    def is_configured(self) -> bool:
        return bool(get_settings().gemini_api_key)

    def describe(self) -> dict:
        s = get_settings()
        return {
            "provider": "gemini",
            "classifier": s.classifier_model,
            "qa": s.qa_model,
            "embedding": s.embedding_model,
            "configured": self.is_configured(),
        }

    def list_models(self) -> list[str]:
        return [m.name for m in self.client().models.list()]

    # -- capabilities ------------------------------------------------------

    def generate_json(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> JSONResult:
        settings = get_settings()
        model_id = model or settings.classifier_model
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )

        response = _with_retries(
            lambda: self.client().models.generate_content(
                model=model_id, contents=prompt, config=config
            ),
            what=f"generate_json({model_id})",
        )

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            text = (response.text or "").strip()
            if not text:
                raise LLMUnavailableError(f"Empty response from {model_id}")
            parsed = schema.model_validate_json(text)

        return JSONResult(parsed=parsed, text=response.text or "", usage=usage_from(response))

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        model: str | None = None,
    ) -> tuple[list[list[float]], Usage]:
        if not texts:
            return [], Usage()

        settings = get_settings()
        model_id = model or settings.embedding_model
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=settings.embedding_dim
        )

        response = _with_retries(
            lambda: self.client().models.embed_content(
                model=model_id, contents=texts, config=config
            ),
            what=f"embed({model_id})",
        )

        vectors = [list(e.values) for e in (response.embeddings or [])]
        return vectors, Usage(calls=1)

    # -- tool-calling chat -------------------------------------------------

    @staticmethod
    def _to_contents(turns: list[Turn]) -> list[types.Content]:
        contents: list[types.Content] = []
        for turn in turns:
            if turn.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=turn.tool_name or "tool",
                                response={"result": turn.text},
                            )
                        ],
                    )
                )
                continue

            parts: list[types.Part] = []
            if turn.text:
                parts.append(types.Part(text=turn.text))
            for call in turn.tool_calls:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call.name, args=call.arguments)
                    )
                )
            if parts:
                contents.append(
                    types.Content(
                        role="model" if turn.role == "assistant" else "user", parts=parts
                    )
                )
        return contents

    @staticmethod
    def _to_tools(tools: list[ToolSpec]) -> list[types.Tool]:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description,
                        parameters=types.Schema.model_validate(t.parameters),
                    )
                    for t in tools
                ]
            )
        ]

    def stream_turn(
        self,
        *,
        system: str,
        turns: list[Turn],
        tools: list[ToolSpec],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[StreamEvent]:
        settings = get_settings()
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=self._to_tools(tools) if tools else None,
            # Our tools need a DB session, so they're dispatched by the caller.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=temperature,
        )

        try:
            stream = self.client().models.generate_content_stream(
                model=model or settings.qa_model, contents=self._to_contents(turns), config=config
            )
        except Exception as exc:  # noqa: BLE001
            if _is_quota(exc):
                retry_after, limit = parse_quota_error(str(exc))
                raise QuotaExceededError(
                    f"chat: quota exhausted{f' (limit {limit}/day)' if limit else ''}",
                    retry_after=retry_after,
                ) from exc
            raise

        calls: list[ToolCall] = []
        usage = Usage()

        for piece in stream:
            if getattr(piece, "usage_metadata", None) is not None:
                usage = usage_from(piece)

            candidates = getattr(piece, "candidates", None) or []
            if not candidates:
                continue
            content = getattr(candidates[0], "content", None)
            for part in getattr(content, "parts", None) or []:
                fn = getattr(part, "function_call", None)
                if fn:
                    calls.append(
                        ToolCall(
                            id=f"call-{len(calls)}",
                            name=fn.name or "",
                            arguments=dict(fn.args or {}),
                        )
                    )
                elif getattr(part, "text", None):
                    yield TextChunk(part.text)

        if calls:
            yield ToolCallsChunk(calls)
        yield UsageChunk(usage)
