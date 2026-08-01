"""Backfill control and health."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import backfill, statestore
from app.config import get_settings
from app.db import get_db
from app.gmail import auth as gmail_auth
from app.models import Application, Email
from app.schemas import BackfillRequest, HealthResponse, SyncStatus

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["sync"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    from app.gmail.watcher import is_running as watcher_running

    # access_status never triggers a browser consent flow, so it is safe to
    # call from an HTTP request.
    access = gmail_auth.access_status()
    if not access.usable:
        log.debug("Gmail unavailable: %s", access.error)

    return HealthResponse(
        status="ok",
        gmail_authorised=access.authorised,
        gmail_usable=access.usable,
        gmail_address=access.address,
        gmail_error=access.error,
        gmail_hint=access.hint,
        emails_stored=db.scalar(select(func.count()).select_from(Email)) or 0,
        applications=db.scalar(select(func.count()).select_from(Application)) or 0,
        watcher_running=watcher_running(),
        llm_calls=statestore.get_int(db, statestore.LLM_CALLS),
        prompt_tokens=statestore.get_int(db, statestore.TOKENS_PROMPT),
        output_tokens=statestore.get_int(db, statestore.TOKENS_OUTPUT),
    )


@router.get("/sync/status", response_model=SyncStatus)
def sync_status() -> SyncStatus:
    return SyncStatus(**backfill.status())


@router.post("/sync/backfill", response_model=SyncStatus, status_code=202)
def start_backfill(payload: BackfillRequest | None = None) -> SyncStatus:
    from app.agent.pipeline import submit_email_id

    settings = get_settings()
    days = (payload.days if payload and payload.days else None) or settings.backfill_default_days

    access = gmail_auth.access_status()
    if not access.usable:
        # Report the actual cause. Telling someone to re-run the consent flow
        # when the real problem is a disabled API just wastes their time.
        detail = access.error or "Gmail is not available."
        if access.hint:
            detail = f"{detail} {access.hint}"
        raise HTTPException(status_code=409, detail=detail)

    if not backfill.start(days, on_email=submit_email_id):
        raise HTTPException(status_code=409, detail="A backfill is already running")

    return SyncStatus(**backfill.status())
