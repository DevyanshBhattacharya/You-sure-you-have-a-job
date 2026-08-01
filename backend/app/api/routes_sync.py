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
from app.gmail import client as gmail_client
from app.models import Application, Email
from app.schemas import BackfillRequest, HealthResponse, SyncStatus

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["sync"])


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    from app.gmail.watcher import is_running as watcher_running

    gmail_ok = False
    address: str | None = None
    try:
        # Never trigger a browser consent flow from an HTTP request.
        gmail_auth.load_credentials(allow_interactive=False)
        address = gmail_client.get_profile().get("emailAddress")
        gmail_ok = True
    except Exception as exc:  # noqa: BLE001
        log.debug("Gmail not authorised: %s", exc)

    return HealthResponse(
        status="ok",
        gmail_authorised=gmail_ok,
        gmail_address=address,
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

    try:
        gmail_auth.load_credentials(allow_interactive=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=409,
            detail=(
                "Gmail is not authorised yet. Run `python -m app.gmail.auth` "
                "from the backend directory to grant access."
            ),
        ) from exc

    if not backfill.start(days, on_email=submit_email_id):
        raise HTTPException(status_code=409, detail="A backfill is already running")

    return SyncStatus(**backfill.status())
