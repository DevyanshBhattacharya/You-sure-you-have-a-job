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


def _usage(payload: dict) -> Usage:
    return Usage(
        prompt_tokens=int(payload.get("prompt_eval_count") or 0),
        output_tokens=int(payload.get("eval_count") or 0),
        calls=1,
    )


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
            "options": {"temperature": temperature},
        }

        data = self._post("/api/chat", payload)
        text = (data.get("message") or {}).get("content") or ""
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
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = self._to_tools(tools)

        calls: list[ToolCall] = []
        usage = Usage()

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
                    text = message.get("content") or ""
                    if text:
                        yield TextChunk(text)

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
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from exc

        if calls:
            yield ToolCallsChunk(calls)
        yield UsageChunk(usage)
