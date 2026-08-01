"""Shared builders for test emails.

`raw_message` produces the shape Gmail's `format=full` actually returns, so the
MIME tests exercise the real parser rather than a simplified stand-in.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

from app.models import Email

BASE_TIME = datetime(2026, 3, 2, 9, 30, tzinfo=UTC)


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def raw_message(
    *,
    gmail_id: str = "msg1",
    thread_id: str = "thread1",
    subject: str = "Subject",
    sender: str = "Talent Team <careers@acme.com>",
    to: str = "me@example.com",
    plain: str | None = "Hello",
    html: str | None = None,
    labels: list[str] | None = None,
    received: datetime | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict:
    headers = [
        {"name": "From", "value": sender},
        {"name": "To", "value": to},
        {"name": "Subject", "value": subject},
    ]
    for name, value in (extra_headers or {}).items():
        headers.append({"name": name, "value": value})

    parts = []
    if plain is not None:
        parts.append({"mimeType": "text/plain", "body": {"data": b64(plain)}})
    if html is not None:
        parts.append({"mimeType": "text/html", "body": {"data": b64(html)}})

    if len(parts) == 1:
        payload = {"mimeType": parts[0]["mimeType"], "headers": headers, "body": parts[0]["body"]}
    else:
        payload = {"mimeType": "multipart/alternative", "headers": headers, "parts": parts}

    stamp = int((received or BASE_TIME).timestamp() * 1000)
    return {
        "id": gmail_id,
        "threadId": thread_id,
        "historyId": "12345",
        "internalDate": str(stamp),
        "snippet": (plain or html or "")[:120],
        "labelIds": labels if labels is not None else ["INBOX"],
        "payload": payload,
    }


def make_email(
    db,
    *,
    gmail_id: str = "msg1",
    thread_id: str = "thread1",
    subject: str = "Hello there",
    sender: str = "someone@example.com",
    sender_name: str = "",
    body: str = "Here is the update you asked for.",
    labels: list[str] | None = None,
    headers: dict[str, str] | None = None,
    days_ago: int = 0,
) -> Email:
    """Insert an Email row directly, bypassing Gmail.

    Defaults are deliberately *neutral* — no job-related words in the subject,
    sender or body. A test that wants job content states it explicitly, so a
    fixture default can never accidentally satisfy the thing under test.
    """
    email = Email(
        gmail_id=gmail_id,
        thread_id=thread_id,
        from_addr=sender,
        from_name=sender_name,
        to_addr="me@example.com",
        subject=subject,
        snippet=body[:120],
        body_text=body,
        received_at=BASE_TIME - timedelta(days=days_ago),
        labels=labels if labels is not None else ["INBOX"],
        raw_headers=headers or {},
    )
    db.add(email)
    db.flush()
    return email
