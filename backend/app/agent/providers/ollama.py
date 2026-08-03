"""Local Ollama backend.

Talks to Ollama's HTTP API directly (httpx is already a dependency, so no extra
package). No API key, no quota, no per-day ceiling — the cost is latency and
whatever the machine can run.

Endpoints used:
  POST /api/chat    structured output via `format` (a JSON Schema), and
                    tool calling via `tools`
  POST /api/embed   batch embeddings
  GET  /api/tags    installed models
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.agent.providers.base import (
    JSONResult,
    LLMUnavailableError,
    ProviderUnavailableError,
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


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _usage(payload: dict) -> Usage:
    return Usage(
        prompt_tokens=int(payload.get("prompt_eval_count") or 0),
        output_tokens=int(payload.get("eval_count") or 0),
        calls=1,
    )


def strip_thinking(text: str) -> str:
    """Remove inline `<think>...</think>` blocks.

    Recent Ollama returns reasoning in a separate `thinking` field, but older
    builds — and models whose template doesn't declare it — leave it inline in
    the content, where it would otherwise be streamed to the user as the answer.
    """
    if "<think>" not in text.lower():
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    # An unterminated block means generation stopped mid-thought; keeping the
    # fragment would show raw reasoning in the UI.
    head, marker, _ = cleaned.partition("<think>")
    return (head if marker else cleaned).strip()


class _ThinkFilter:
    """Drops inline `<think>` blocks from a token stream.

    The tags arrive split across chunks, so a stateless per-chunk regex would
    leak fragments like `<thi`. Anything that could still turn out to be a tag
    is held back until it's resolved either way.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, text: str) -> str:
        self._buffer += text
        out: list[str] = []

        while self._buffer:
            if self._inside:
                end = self._buffer.find(self.CLOSE)
                if end == -1:
                    # Keep only enough to recognise a close tag spanning chunks.
                    self._buffer = self._buffer[-len(self.CLOSE) :]
                    break
                self._buffer = self._buffer[end + len(self.CLOSE) :]
                self._inside = False
                continue

            start = self._buffer.find(self.OPEN)
            if start != -1:
                out.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self.OPEN) :]
                self._inside = True
                continue

            # No tag yet. Emit everything except a possible partial tag at the end.
            hold = 0
            for size in range(1, len(self.OPEN)):
                if self._buffer.endswith(self.OPEN[:size]):
                    hold = size
            if hold:
                out.append(self._buffer[:-hold])
                self._buffer = self._buffer[-hold:]
            else:
                out.append(self._buffer)
                self._buffer = ""
            break

        return "".join(out)

    def flush(self) -> str:
        """Emit whatever was held back, once the stream is known to be over."""
        tail = "" if self._inside else self._buffer
        self._buffer = ""
        return tail


def _unwrap_refs(schema: dict) -> dict:
    """Inline Pydantic's `$defs`/`$ref` indirection.

    Ollama's `format` wants a self-contained JSON Schema; several models choke
    on `$ref`. Flat schemas (ours) rarely produce defs, but nested ones do.
    """
    defs = schema.get("$defs") or {}
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


class OllamaProvider:
    name = "ollama"

    # -- plumbing ----------------------------------------------------------

    @property
    def _settings(self):
        return get_settings()

    def _url(self, path: str) -> str:
        return f"{self._settings.ollama_base_url.rstrip('/')}{path}"

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(self._settings.ollama_timeout_seconds))

    def _options(self, temperature: float, *, num_ctx: int | None = None) -> dict:
        # num_ctx is not optional. Ollama defaults every model to 4096 tokens no
        # matter its real capacity, then silently discards the overflow from the
        # front of the prompt — taking the system instruction and the question
        # with it. See `_warn_if_truncated`.
        #
        # Nor is one value right for everything. The window is not free: its KV
        # cache is allocated in VRAM alongside the weights, so an oversized one
        # evicts layers to the CPU. Measured on qwen3:4b (5.1 GB) with a ~2k
        # token classification prompt: at num_ctx 16384 only 2.6 GB stayed
        # resident and a single call ran for over ten minutes; the same call
        # sized to the prompt keeps the whole model on the GPU. Callers that
        # need a big window (the Q&A agent, whose tool results are large) ask
        # for one; extraction does not.
        return {
            "temperature": temperature,
            "num_ctx": num_ctx or self._settings.ollama_num_ctx,
        }

    def _warn_if_truncated(self, payload: dict, model_id: str, limit: int | None = None) -> None:
        """Surface a context overflow, which Ollama itself reports as success.

        `prompt_eval_count` is what the model actually saw. When it lands at the
        window size, the prompt was longer than that and the difference was
        dropped — the model answered a question it could no longer read.
        """
        seen = int(payload.get("prompt_eval_count") or 0)
        limit = limit or self._settings.ollama_num_ctx
        if seen and seen >= limit - 8:
            log.warning(
                "Ollama truncated the prompt for %s: %d tokens evaluated against a "
                "%d-token window. The start of the prompt was discarded. Raise "
                "OLLAMA_NUM_CTX or shorten the input.",
                model_id,
                seen,
                limit,
            )

    def is_configured(self) -> bool:
        # No credentials to check. Reachability is verified at call time and
        # reported as ProviderUnavailableError so work is deferred, not faked.
        return bool(self._settings.ollama_base_url)

    def describe(self) -> dict:
        s = self._settings
        return {
            "provider": "ollama",
            "base_url": s.ollama_base_url,
            "classifier": s.ollama_model,
            "qa": s.ollama_qa_model or s.ollama_model,
            "embedding": s.ollama_embedding_model,
            "configured": self.is_configured(),
        }

    def list_models(self) -> list[str]:
        try:
            with self._client() as client:
                response = client.get(self._url("/api/tags"))
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc
        return [m["name"] for m in response.json().get("models", [])]

    def _unreachable(self, exc: Exception) -> ProviderUnavailableError:
        # A read timeout is not an unreachable server — it is a server that
        # accepted the request and is still thinking. Telling someone to run
        # `ollama serve` when it is already running (and visibly busy) sends
        # them to fix the one thing that isn't broken.
        if isinstance(exc, httpx.ReadTimeout | httpx.WriteTimeout | httpx.PoolTimeout):
            return ProviderUnavailableError(
                f"Ollama did not respond within {self._settings.ollama_timeout_seconds:.0f}s. "
                f"The model ({self._settings.ollama_model}) is still generating, not down — "
                "raise OLLAMA_TIMEOUT_SECONDS, lower OLLAMA_NUM_CTX, or use a smaller model.",
                retry_after=30.0,
            )
        return ProviderUnavailableError(
            f"Ollama is not reachable at {self._settings.ollama_base_url} ({exc}). "
            "Start it with `ollama serve`.",
            # Local service — worth trying again soon rather than in an hour.
            retry_after=60.0,
        )

    def _post(self, path: str, payload: dict) -> dict:
        try:
            with self._client() as client:
                response = client.post(self._url(path), json=payload)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc

        if response.status_code == 404:
            # Ollama 404s when the model isn't pulled. Say so explicitly —
            # "404" alone sends people looking in the wrong place.
            raise LLMUnavailableError(
                f"Ollama has no model {payload.get('model')!r}. "
                f"Pull it with `ollama pull {payload.get('model')}`."
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(f"Ollama error {response.status_code}", retry_after=30.0)
        response.raise_for_status()
        return response.json()

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
        model_id = model or self._settings.ollama_model
        extraction_ctx = self._settings.ollama_extraction_num_ctx
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_id,
            "messages": messages,
            # A JSON Schema here constrains decoding, the same guarantee
            # Gemini's response_schema gives.
            "format": _unwrap_refs(schema.model_json_schema()),
            "stream": False,
            # Reasoning must be OFF here, and this is not a preference.
            # A `format` schema constrains decoding from the very first token,
            # but a reasoning model is trained to open with `<think>` — which
            # the grammar forbids. The two fight, and generation degenerates:
            # measured on qwen3:4b, an 810-token prompt produced nothing after
            # 15 minutes with reasoning on, and a correct answer in 22 seconds
            # with it off. Nothing is lost by disabling it, because the schema
            # already prevents prose from reaching the output.
            "think": False,
            # Sized to the prompt, not to the Q&A agent's needs. Extraction
            # inputs are capped (classify.MAX_PROMPT_BODY_CHARS), so a bigger
            # window buys nothing and costs the VRAM the weights need.
            "options": self._options(temperature, num_ctx=extraction_ctx),
        }

        data = self._post("/api/chat", payload)
        self._warn_if_truncated(data, model_id, extraction_ctx)
        text = strip_thinking((data.get("message") or {}).get("content") or "")
        if not text.strip():
            raise LLMUnavailableError(f"Empty response from Ollama model {model_id}")

        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as exc:
            # Small local models occasionally emit prose around the JSON even
            # with a schema set; salvage the object before giving up.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end <= start:
                raise LLMUnavailableError(
                    f"{model_id} returned unparseable output: {text[:200]}"
                ) from exc
            parsed = schema.model_validate_json(text[start : end + 1])

        return JSONResult(parsed=parsed, text=text, usage=_usage(data))

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        model: str | None = None,
    ) -> tuple[list[list[float]], Usage]:
        if not texts:
            return [], Usage()

        model_id = model or self._settings.ollama_embedding_model
        data = self._post("/api/embed", {"model": model_id, "input": texts})
        vectors = [[float(x) for x in v] for v in data.get("embeddings", [])]
        if len(vectors) != len(texts):
            raise LLMUnavailableError(
                f"Ollama returned {len(vectors)} embedding(s) for {len(texts)} input(s)"
            )
        return vectors, _usage(data)

    # -- tool-calling chat -------------------------------------------------

    @staticmethod
    def _to_messages(system: str, turns: list[Turn]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in turns:
            if turn.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "content": turn.text or "",
                        # Ollama matches results to calls by name.
                        "tool_name": turn.tool_name or "",
                    }
                )
            elif turn.role == "assistant":
                message: dict = {"role": "assistant", "content": turn.text or ""}
                if turn.tool_calls:
                    message["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in turn.tool_calls
                    ]
                messages.append(message)
            else:
                messages.append({"role": "user", "content": turn.text or ""})
        return messages

    @staticmethod
    def _to_tools(tools: list[ToolSpec]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
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
        model_id = model or self._settings.ollama_qa_model or self._settings.ollama_model
        payload = {
            "model": model_id,
            "messages": self._to_messages(system, turns),
            "stream": True,
            "think": self._settings.ollama_think,
            "options": self._options(temperature),
        }
        if tools:
            payload["tools"] = self._to_tools(tools)

        calls: list[ToolCall] = []
        usage = Usage()
        # Reasoning normally arrives in its own `thinking` field, which we drop
        # by only reading `content`. This catches models that inline it instead.
        think_filter = _ThinkFilter()

        try:
            with self._client() as client, client.stream(
                "POST", self._url("/api/chat"), json=payload
            ) as response:
                if response.status_code == 404:
                    raise LLMUnavailableError(
                        f"Ollama has no model {model_id!r}. Pull it with `ollama pull {model_id}`."
                    )
                response.raise_for_status()

                # Ollama streams newline-delimited JSON, one object per chunk.
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = chunk.get("message") or {}
                    raw_text = message.get("content") or ""
                    if raw_text:
                        visible = think_filter.feed(raw_text)
                        if visible:
                            yield TextChunk(visible)

                    for raw in message.get("tool_calls") or []:
                        fn = raw.get("function") or {}
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):  # some models emit a JSON string
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        calls.append(
                            ToolCall(
                                id=f"call-{len(calls)}",
                                name=fn.get("name", ""),
                                arguments=args,
                            )
                        )

                    if chunk.get("done"):
                        usage = _usage(chunk)
                        self._warn_if_truncated(chunk, model_id)
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc

        tail = think_filter.flush()
        if tail:
            yield TextChunk(tail)

        if calls:
            yield ToolCallsChunk(calls)
        yield UsageChunk(usage)
