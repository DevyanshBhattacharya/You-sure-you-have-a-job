"""Persisting fetched mail.

Idempotent on `gmail_id`: history pages overlap, backfills re-run, and the
watcher can replay after a crash. None of that may create duplicate rows.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.gmail import client as gmail_client
from app.gmail.client import ParsedEmail
from app.models import Email

log = logging.getLogger(__name__)


def upsert_email(session: Session, parsed: ParsedEmail) -> tuple[Email, bool]:
    """Insert or refresh an email row. Returns (row, created)."""
    existing = session.scalar(select(Email).where(Email.gmail_id == parsed.gmail_id))

    if existing is not None:
        # Labels and history id drift as the message is read/archived; the
        # classification fields are deliberately left alone so a re-sync never
        # discards work the agent already did.
        existing.labels = parsed.labels
        existing.history_id = parsed.history_id or existing.history_id
        return existing, False

    email = Email(
        gmail_id=parsed.gmail_id,
        thread_id=parsed.thread_id,
        history_id=parsed.history_id,
        from_addr=parsed.from_addr,
        from_name=parsed.from_name,
        to_addr=parsed.to_addr,
        subject=parsed.subject,
        snippet=parsed.snippet,
        body_text=parsed.body_text,
        received_at=parsed.received_at,
        labels=parsed.labels,
        raw_headers=parsed.headers,
    )
    session.add(email)
    session.flush()  # assign id for downstream stages
    return email, True


def fetch_and_store(session: Session, gmail_id: str) -> tuple[Email | None, bool]:
    """Fetch one message from Gmail and persist it.

    Returns (row, created). `(None, False)` means the message no longer exists.
    """
    raw = gmail_client.get_message(gmail_id)
    if raw is None:
        return None, False
    parsed = gmail_client.parse_message(raw)
    return upsert_email(session, parsed)


def existing_gmail_ids(session: Session, gmail_ids: list[str]) -> set[str]:
    """Which of these ids are already stored (chunked to stay under SQLite limits)."""
    found: set[str] = set()
    for i in range(0, len(gmail_ids), 500):
        batch = gmail_ids[i : i + 500]
        rows = session.scalars(select(Email.gmail_id).where(Email.gmail_id.in_(batch))).all()
        found.update(rows)
    return found
