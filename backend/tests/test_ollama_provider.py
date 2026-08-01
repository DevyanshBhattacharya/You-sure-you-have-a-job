"""The local Ollama backend.

Ollama's HTTP surface is stubbed, so these run on a machine that has never
installed it — which is the point: the adapter must be verifiable before the
runtime is present.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from app.agent.providers import base
from app.agent.providers.base import (
    LLMUnavailableError,
    ProviderUnavailableError,
    TextChunk,
    ToolCall,
    ToolCallsChunk,
    ToolSpec,
    Turn,
    UsageChunk,
)
from app.agent.providers.ollama import OllamaProvider, _unwrap_refs


class Extraction(BaseModel):
    is_job_related: bool
    company: str = ""


@pytest.fixture
def provider():
    return OllamaProvider()


def stub_transport(handler):
    """Wire httpx to a callable instead of the network."""
    return httpx.MockTransport(handler)


@pytest.fixture
def route(monkeypatch, provider):
    """Install a request handler for every httpx.Client the provider builds."""

    def install(handler):
        def build(*_a, **_kw):
            return httpx.Client(transport=stub_transport(handler), timeout=5)

        monkeypatch.setattr(provider, "_client", build)

    return install


class TestSchemaFlattening:
    def test_leaves_flat_schemas_alone(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert _unwrap_refs(schema) == schema

    def test_inlines_nested_definitions(self):
        """Ollama wants a self-contained schema; several models choke on $ref."""
        schema = {
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner"}},
            "$defs": {"Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        }
        flat = _unwrap_refs(schema)
        assert "$defs" not in flat
        assert flat["properties"]["inner"]["properties"]["x"] == {"type": "integer"}


class TestStructuredOutput:
    def test_sends_json_schema_and_parses_the_result(self, provider, route):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"is_job_related": true, "company": "Acme"}'},
                    "prompt_eval_count": 120,
                    "eval_count": 30,
                },
            )

        route(handler)
        result = provider.generate_json(prompt="classify", schema=Extraction, system="be terse")

        assert result.parsed.company == "Acme"
        assert result.usage.prompt_tokens == 120
        assert result.usage.output_tokens == 30
        # Constrained decoding is what makes classification reliable.
        assert seen["format"]["type"] == "object"
        assert seen["stream"] is False
        assert seen["messages"][0]["role"] == "system"

    def test_salvages_json_wrapped_in_prose(self, provider, route):
        """Small local models sometimes narrate around the object."""
        route(
            lambda _r: httpx.Response(
                200,
                json={
                    "message": {
                        "content": 'Sure! {"is_job_related": false, "company": ""} Hope that helps.'
                    }
                },
            )
        )
        assert provider.generate_json(prompt="x", schema=Extraction).parsed.is_job_related is False

    def test_unparseable_output_raises_rather_than_guessing(self, provider, route):
        route(lambda _r: httpx.Response(200, json={"message": {"content": "no json here"}}))
        with pytest.raises(LLMUnavailableError):
            provider.generate_json(prompt="x", schema=Extraction)

    def test_missing_model_names_the_pull_command(self, provider, route):
        route(lambda _r: httpx.Response(404, json={"error": "model not found"}))
        with pytest.raises(LLMUnavailableError, match="ollama pull"):
            provider.generate_json(prompt="x", schema=Extraction)


class TestUnreachable:
    def test_connection_refused_defers_work(self, provider, route):
        """Must be a DeferWorkError, not a generic failure: Ollama being down is
        systemic, and a heuristic fallback would burn the whole backlog."""

        def handler(_request):
            raise httpx.ConnectError("connection refused")

        route(handler)
        with pytest.raises(ProviderUnavailableError) as excinfo:
            provider.generate_json(prompt="x", schema=Extraction)

        assert isinstance(excinfo.value, base.DeferWorkError)
        assert "ollama serve" in str(excinfo.value)
        assert excinfo.value.retry_after == 60.0

    def test_server_error_defers_too(self, provider, route):
        route(lambda _r: httpx.Response(503, text="overloaded"))
        with pytest.raises(ProviderUnavailableError):
            provider.generate_json(prompt="x", schema=Extraction)


class TestEmbeddings:
    def test_returns_one_vector_per_input(self, provider, route):
        route(
            lambda _r: httpx.Response(
                200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "prompt_eval_count": 8}
            )
        )
        vectors, usage = provider.embed(["a", "b"])
        assert [len(v) for v in vectors] == [3, 3]
        assert usage.prompt_tokens == 8

    def test_count_mismatch_is_an_error(self, provider, route):
        route(lambda _r: httpx.Response(200, json={"embeddings": [[0.1]]}))
        with pytest.raises(LLMUnavailableError):
            provider.embed(["a", "b"])

    def test_empty_input_makes_no_request(self, provider):
        assert provider.embed([]) == ([], base.Usage())


class TestToolCallingStream:
    def _ndjson(self, *objects: dict) -> bytes:
        return b"".join(json.dumps(o).encode() + b"\n" for o in objects)

    def test_streams_text_then_reports_usage(self, provider, route):
        body = self._ndjson(
            {"message": {"content": "Two "}, "done": False},
            {"message": {"content": "offers."}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 9},
        )
        route(lambda _r: httpx.Response(200, content=body))

        events = list(
            provider.stream_turn(system="s", turns=[Turn(role="user", text="how many?")], tools=[])
        )

        assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "Two offers."
        usage = next(e.usage for e in events if isinstance(e, UsageChunk))
        assert usage.prompt_tokens == 50
        assert usage.output_tokens == 9

    def test_surfaces_tool_calls(self, provider, route):
        body = self._ndjson(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "list_applications",
                                "arguments": {"status": "offer"},
                            }
                        }
                    ],
                },
                "done": True,
            }
        )
        route(lambda _r: httpx.Response(200, content=body))

        calls = [
            e
            for e in provider.stream_turn(
                system="s",
                turns=[Turn(role="user", text="offers?")],
                tools=[ToolSpec("list_applications", "d", {"type": "object"})],
            )
            if isinstance(e, ToolCallsChunk)
        ]

        assert calls[0].calls[0].name == "list_applications"
        assert calls[0].calls[0].arguments == {"status": "offer"}

    def test_tool_arguments_given_as_a_json_string_are_decoded(self, provider, route):
        body = self._ndjson(
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "t", "arguments": '{"k": 3}'}}],
                },
                "done": True,
            }
        )
        route(lambda _r: httpx.Response(200, content=body))

        events = list(provider.stream_turn(system="s", turns=[], tools=[]))
        call = next(e for e in events if isinstance(e, ToolCallsChunk)).calls[0]
        assert call.arguments == {"k": 3}

    def test_tools_are_sent_in_openai_shape(self, provider, route):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, content=self._ndjson({"message": {}, "done": True}))

        route(handler)
        list(
            provider.stream_turn(
                system="s",
                turns=[Turn(role="user", text="q")],
                tools=[ToolSpec("search_emails", "find things", {"type": "object"})],
            )
        )

        assert seen["tools"][0]["type"] == "function"
        assert seen["tools"][0]["function"]["name"] == "search_emails"


class TestMessageMapping:
    def test_roles_map_to_ollama_shape(self, provider):
        turns = [
            Turn(role="user", text="how many offers?"),
            Turn(
                role="assistant",
                text=None,
                tool_calls=[ToolCall(id="c0", name="list_applications", arguments={"s": 1})],
            ),
            Turn(role="tool", text='{"count": 2}', tool_name="list_applications"),
        ]
        messages = provider._to_messages("SYSTEM", turns)

        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        assert messages[2]["tool_calls"][0]["function"]["name"] == "list_applications"
        # Ollama matches a result back to its call by name.
        assert messages[3]["tool_name"] == "list_applications"
