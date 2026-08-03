"""Context-window handling for the local backend.

Ollama defaults every model to a 4096-token window regardless of what the model
supports, and silently drops the overflow from the *front* of the prompt — the
system instruction and the question. Nothing errors; the model simply answers
something it can no longer read. These tests pin the two defences: always send
an explicit `num_ctx`, and bound what we feed back into it.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import BaseModel

from app.agent.providers.base import Turn
from app.agent.providers.ollama import OllamaProvider, _ThinkFilter, strip_thinking
from app.agent.qa import MAX_TOOL_RESULT_CHARS, _encode_tool_result
from app.config import get_settings


class Probe(BaseModel):
    ok: bool


@pytest.fixture
def provider():
    return OllamaProvider()


@pytest.fixture
def route(monkeypatch, provider):
    def install(handler):
        def build(*_a, **_kw):
            return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)

        monkeypatch.setattr(provider, "_client", build)

    return install


class TestExplicitContextWindow:
    def test_generate_json_sends_num_ctx(self, provider, route):
        """Without this the model silently loses the start of the prompt."""
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

        route(handler)
        provider.generate_json(prompt="hi", schema=Probe)

        assert seen["options"]["num_ctx"] == get_settings().ollama_extraction_num_ctx
        assert seen["options"]["num_ctx"] > 4096

    def test_extraction_does_not_pay_for_the_chat_window(self, provider, route):
        """A window is not free: its KV cache sits in VRAM beside the weights,
        so sizing extraction for the Q&A agent's tool results evicts layers to
        the CPU. Measured on qwen3:4b, that took one classification of a
        500-token prompt from seconds to over ten minutes."""
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

        route(handler)
        provider.generate_json(prompt="hi", schema=Probe)

        settings = get_settings()
        assert seen["options"]["num_ctx"] < settings.ollama_num_ctx
        # Still has to clear the capped extraction prompt with room to spare.
        assert seen["options"]["num_ctx"] >= 4096

    def test_stream_turn_sends_num_ctx(self, provider, route):
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            body = json.dumps({"message": {"content": "hi"}, "done": True}) + "\n"
            return httpx.Response(200, content=body.encode())

        route(handler)
        list(provider.stream_turn(system="s", turns=[Turn(role="user", text="q")], tools=[]))

        assert seen["options"]["num_ctx"] == get_settings().ollama_num_ctx

    def test_truncation_is_reported(self, provider, route, caplog):
        """Ollama reports an overflowed prompt as a success; we must not."""
        limit = get_settings().ollama_num_ctx

        def handler(request):
            return httpx.Response(
                200,
                json={
                    "message": {"content": '{"ok": true}'},
                    # Evaluated exactly up to the ceiling: the rest was dropped.
                    "prompt_eval_count": limit,
                },
            )

        route(handler)
        with caplog.at_level(logging.WARNING):
            provider.generate_json(prompt="x" * 100, schema=Probe)

        assert "truncated the prompt" in caplog.text
        assert "OLLAMA_NUM_CTX" in caplog.text

    def test_normal_prompt_does_not_warn(self, provider, route, caplog):
        def handler(request):
            return httpx.Response(
                200, json={"message": {"content": '{"ok": true}'}, "prompt_eval_count": 200}
            )

        route(handler)
        with caplog.at_level(logging.WARNING):
            provider.generate_json(prompt="x", schema=Probe)

        assert "truncated" not in caplog.text


class TestReasoningAndSchemaAreKeptApart:
    """A `format` schema and a reasoning model cannot both go first.

    The schema constrains decoding from token 0; a reasoning model is trained to
    open with `<think>`, which the grammar forbids. Generation then degenerates.
    Measured on qwen3:4b: an 810-token prompt produced nothing in 15 minutes
    with reasoning on, and a correct answer in 22 seconds with it off.
    """

    def test_structured_output_always_disables_reasoning(self, provider, route, monkeypatch):
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

        route(handler)
        # Even with reasoning switched on for chat, extraction must not use it.
        monkeypatch.setattr(get_settings(), "ollama_think", True, raising=False)
        provider.generate_json(prompt="hi", schema=Probe)

        assert seen["think"] is False
        assert seen["format"]["properties"]["ok"]["type"] == "boolean"

    def test_chat_keeps_reasoning_so_ollama_separates_it(self, provider, route, monkeypatch):
        """In free-form chat the reverse holds: reasoning on keeps it out of
        `content`. Turning it off makes the model reason into the answer."""
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            body = json.dumps({"message": {"content": "hi"}, "done": True}) + "\n"
            return httpx.Response(200, content=body.encode())

        route(handler)
        monkeypatch.setattr(get_settings(), "ollama_think", True, raising=False)
        list(provider.stream_turn(system="s", turns=[Turn(role="user", text="q")], tools=[]))

        assert seen["think"] is True


class TestThinkingIsHidden:
    """Reasoning models must not stream their reasoning as the answer.

    Ollama normally splits it into a separate `thinking` field, which the
    adapter ignores by only reading `content`. Some models inline it instead.
    """

    def test_strip_thinking_removes_a_complete_block(self):
        assert strip_thinking("<think>hmm</think>Answer") == "Answer"

    def test_strip_thinking_drops_an_unterminated_block(self):
        # Generation stopped mid-thought; the fragment is not an answer.
        assert strip_thinking("<think>hmm and then") == ""

    def test_untagged_text_is_untouched(self):
        assert strip_thinking("Just the answer") == "Just the answer"

    def test_filter_handles_tags_split_across_chunks(self):
        """Tokens arrive one at a time, so `<think>` never arrives whole."""
        f = _ThinkFilter()
        out = "".join(f.feed(c) for c in ["<th", "ink>rea", "soning</thi", "nk>Real ", "answer"])
        assert f.flush() == ""
        assert out == "Real answer"

    def test_filter_never_leaks_a_partial_tag(self):
        f = _ThinkFilter()
        assert "<" not in f.feed("Answer <thi")
        # Resolved as ordinary text once it turns out not to be a tag.
        assert f.feed("s is fine") + f.flush() == "<this is fine"

    def test_filter_passes_plain_text_straight_through(self):
        f = _ThinkFilter()
        assert "".join(f.feed(c) for c in ["Hello ", "world"]) + f.flush() == "Hello world"


class TestToolResultBudget:
    """Tool output grows with the mailbox; the context window does not."""

    def test_small_results_are_untouched(self):
        result = {"count": 2, "applications": [{"company": "Acme"}, {"company": "Globex"}]}
        assert json.loads(_encode_tool_result(result)) == result

    def test_long_row_lists_are_trimmed_to_fit(self):
        result = {
            "total_applications": 500,
            "counts_by_status": {"applied": 500},
            "applications": [
                {"application_id": i, "company": f"Company {i}", "notes": "x" * 200}
                for i in range(500)
            ],
        }
        encoded = _encode_tool_result(result)
        assert len(encoded) <= MAX_TOOL_RESULT_CHARS

        decoded = json.loads(encoded)
        assert len(decoded["applications"]) < 500
        # The aggregate answers "how many" even though the rows were cut, so a
        # counting question stays correct.
        assert decoded["total_applications"] == 500
        assert "applications_truncated" in decoded

    def test_trimming_is_announced_to_the_model(self):
        result = {"results": [{"text": "y" * 500} for _ in range(200)]}
        decoded = json.loads(_encode_tool_result(result))
        assert "Showing" in decoded["results_truncated"]
