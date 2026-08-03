"""Q&A endpoints."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent import qa
from app.db import SessionLocal, get_db
from app.schemas import ChatRequest, ChatResponse, Citation

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Blocking answer. Use /api/chat/stream for a live-typing UI."""
    history = [t.model_dump() for t in payload.history]
    result = qa.answer(db, payload.message, history)
    return ChatResponse(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        tool_calls=result.tool_calls,
    )


@router.post("/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    """Server-sent events: tool-call progress, then the answer token by token."""
    history = [t.model_dump() for t in payload.history]

    def generate() -> Iterator[str]:
        # This generator outlives the request-scoped dependency, so it owns
        # its own session.
        session = SessionLocal()
        try:
            for event in qa.stream_answer(session, payload.message, history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:  # noqa: BLE001
            # Logged in full, reported in outline. An unhandled exception string
            # can carry file paths, connection strings and query fragments; the
            # person reading the chat window can do nothing with any of it.
            log.exception("Chat stream failed")
            message = "Something went wrong answering that. The server log has the details."
            yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
        finally:
            session.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
