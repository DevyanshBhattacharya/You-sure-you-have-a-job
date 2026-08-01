"""Question answering over the knowledge base.

A tool-calling loop rather than plain RAG: the model decides whether a question
needs the text of an email, a structured query, or both.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

from google.genai import types
from sqlalchemy.orm import Session

from app.agent import llm, tools
from app.config import get_settings

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 6

SYSTEM_INSTRUCTION = """\
You are the assistant for one person's job search. You answer strictly from the \
tools available to you, which read that person's own tracked email and \
applications.

How to work:
- Call tools before answering. Never answer a factual question about \
  applications, companies, dates, or message contents from memory.
- Use `list_applications` for counts, lists, and status questions. Use \
  `search_emails` for what was actually written. Use both when a question needs \
  a number and a quote.
- If a tool returns nothing relevant, say plainly that you found nothing — do \
  not fill the gap with a plausible guess.
- Cite the emails or applications you used, by their subject and company.

Style:
- Lead with the direct answer, then the supporting detail.
- Be concrete: real company names, real dates, real numbers.
- Keep it brief. No preamble, no restating the question.
- Dates in the user's terms ("last Tuesday", "in three days") only when the \
  relative framing is what they asked about; otherwise give the actual date.
"""


@dataclass
class ChatResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


def _system_instruction() -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"{SYSTEM_INSTRUCTION}\nThe current date and time is {now}."


def _to_contents(history: list[dict], message: str) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in history[-12:]:  # keep the prompt bounded
        role = "model" if turn.get("role") == "model" else "user"
        text = (turn.get("content") or "").strip()
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    return contents


def _collect_citations(name: str, result: dict, sink: list[dict], seen: set[tuple]) -> None:
    """Pull citable references out of a tool result."""
    if name == "search_emails":
        for row in result.get("results", []) or []:
            key = ("email", row.get("email_id"))
            if row.get("email_id") and key not in seen:
                seen.add(key)
                sink.append(
                    {
                        "email_id": row.get("email_id"),
                        "application_id": row.get("application_id"),
                        "subject": row.get("subject"),
                        "company": row.get("company"),
                        "received_at": row.get("received_at"),
                    }
                )
    elif name in ("list_applications", "get_upcoming_actions", "get_application_timeline"):
        rows: list[dict] = []
        if "applications" in result:
            rows = result["applications"]
        elif "application" in result:
            rows = [result["application"]]
        rows += result.get("with_deadline", []) + result.get("without_deadline", [])

        for row in rows:
            key = ("app", row.get("application_id"))
            if row.get("application_id") and key not in seen:
                seen.add(key)
                sink.append(
                    {
                        "email_id": None,
                        "application_id": row.get("application_id"),
                        "subject": row.get("role_title"),
                        "company": row.get("company"),
                        "received_at": None,
                    }
                )


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=_system_instruction(),
        tools=tools.TOOLS,
        # Our tools need a DB session, so they're dispatched by hand below.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )


def stream_answer(
    session: Session, message: str, history: list[dict] | None = None
) -> Iterator[dict]:
    """Run the tool loop, yielding progress events.

    Event shapes:
        {"type": "tool",  "name": str, "args": dict}
        {"type": "token", "text": str}
        {"type": "done",  "citations": [...], "tool_calls": [...]}
        {"type": "error", "message": str}
    """
    if not llm.is_configured():
        yield {
            "type": "error",
            "message": (
                "GEMINI_API_KEY is not set, so the assistant can't answer questions. "
                "Add a key to backend/.env and restart."
            ),
        }
        return

    settings = get_settings()
    client = llm.get_client()
    contents = _to_contents(history or [], message)
    config = _config()

    citations: list[dict] = []
    seen: set[tuple] = set()
    called: list[str] = []
    answered = False

    for round_index in range(MAX_TOOL_ROUNDS):
        try:
            stream = client.models.generate_content_stream(
                model=settings.qa_model, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Q&A request failed")
            yield {"type": "error", "message": f"Model request failed: {exc}"}
            return

        function_calls: list[types.FunctionCall] = []
        model_parts: list[types.Part] = []
        text_buffer: list[str] = []
        usage = llm.Usage()

        try:
            for piece in stream:
                if getattr(piece, "usage_metadata", None) is not None:
                    usage = llm.usage_from(piece)

                candidates = getattr(piece, "candidates", None) or []
                if not candidates:
                    continue
                content = getattr(candidates[0], "content", None)
                for part in getattr(content, "parts", None) or []:
                    if getattr(part, "function_call", None):
                        function_calls.append(part.function_call)
                        model_parts.append(part)
                    elif getattr(part, "text", None):
                        text_buffer.append(part.text)
                        model_parts.append(part)
                        yield {"type": "token", "text": part.text}
                        answered = True
        except Exception as exc:  # noqa: BLE001
            log.exception("Q&A stream failed")
            yield {"type": "error", "message": f"Stream failed: {exc}"}
            return
        finally:
            llm.record_usage(usage)

        if not function_calls:
            break

        contents.append(types.Content(role="model", parts=model_parts))

        response_parts: list[types.Part] = []
        for call in function_calls:
            name = call.name or ""
            args = dict(call.args or {})
            called.append(name)
            yield {"type": "tool", "name": name, "args": args}

            result = tools.dispatch(session, name, args)
            _collect_citations(name, result, citations, seen)
            response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=response_parts))

        if round_index == MAX_TOOL_ROUNDS - 1:
            yield {
                "type": "error",
                "message": "The assistant kept requesting tools without producing an answer.",
            }
            return

    if not answered:
        yield {
            "type": "token",
            "text": "I couldn't find anything in your tracked mail that answers that.",
        }

    yield {"type": "done", "citations": citations[:12], "tool_calls": called}


def answer(session: Session, message: str, history: list[dict] | None = None) -> ChatResult:
    """Non-streaming wrapper, used by tests and the JSON endpoint."""
    parts: list[str] = []
    result = ChatResult(answer="")
    for event in stream_answer(session, message, history):
        if event["type"] == "token":
            parts.append(event["text"])
        elif event["type"] == "error":
            parts.append(event["message"])
        elif event["type"] == "done":
            result.citations = event["citations"]
            result.tool_calls = event["tool_calls"]
    result.answer = "".join(parts).strip()
    return result
