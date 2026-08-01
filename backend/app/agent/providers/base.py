"""Provider-neutral types shared by every LLM backend.

The pipeline talks in these terms only. Anything Gemini- or Ollama-shaped stays
behind the `Provider` protocol, so swapping backends is a config change rather
than an edit to the classifier or the Q&A loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Misconfiguration — no key, no model, nothing to call.

    A permanent condition until someone changes settings, so callers should
    surface it rather than retry.
    """


class DeferWorkError(RuntimeError):
    """The backend could not serve this request *right now*.

    The distinction from a generic failure matters: work that hits this must be
    left undone and retried later. Substituting a degraded result would stamp
    the email as processed and freeze a bad verdict that nothing re-examines.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExceededError(DeferWorkError):
    """Rate or quota limit hit (HTTP 429). Free tiers are often per-day."""


class ProviderUnavailableError(DeferWorkError):
    """The backend is unreachable — Ollama not running, connection refused.

    Systemic rather than per-message: falling back to a heuristic would burn
    the entire backlog with junk before anyone noticed.
    """


# --------------------------------------------------------------------------
# Usage accounting
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Tool calling
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ToolSpec:
    """A callable tool, described as plain JSON Schema.

    Each provider converts this into its own dialect — Gemini's
    `FunctionDeclaration`, Ollama's OpenAI-style `tools` array.
    """

    name: str
    description: str
    parameters: dict


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(slots=True)
class Turn:
    """One conversation turn.

    `role` is "user", "assistant" or "tool". An assistant turn may carry
    `tool_calls`; a tool turn carries the result of exactly one of them.
    """

    role: str
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class TextChunk:
    text: str


@dataclass(slots=True)
class ToolCallsChunk:
    calls: list[ToolCall]


@dataclass(slots=True)
class UsageChunk:
    usage: Usage


StreamEvent = TextChunk | ToolCallsChunk | UsageChunk


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


@runtime_checkable
class Provider(Protocol):
    name: str

    def is_configured(self) -> bool:
        """True when this backend has enough config to be worth calling."""

    def describe(self) -> dict:
        """Human-readable config summary. Must never include secrets."""

    def generate_json(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> JSONResult:
        """Constrained generation. The result validates against `schema`."""

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        model: str | None = None,
    ) -> tuple[list[list[float]], Usage]:
        """Embed a batch. Returns (vectors, usage)."""

    def stream_turn(
        self,
        *,
        system: str,
        turns: list[Turn],
        tools: list[ToolSpec],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[StreamEvent]:
        """Stream one assistant turn, which may request tool calls."""
