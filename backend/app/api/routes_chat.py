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
        except Exception as exc:  # noqa: BLE001
            log.exception("Chat stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
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
