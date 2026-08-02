"""Gmail API access and MIME normalisation.

Everything downstream consumes `ParsedEmail`, never the raw Gmail payload.
"""

from __future__ import annotations

import base64
import binascii
import logging
import random
import re
import socket
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parseaddr
from typing import TypeVar

import httplib2
from bs4 import BeautifulSoup
from google.auth.exceptions import TransportError
from googleapiclient.errors import HttpError

from app.gmail.auth import get_service

log = logging.getLogger(__name__)

T = TypeVar("T")

# Transport faults, not API rejections. These say nothing about the request and
# everything about the network between here and Google — a proxy resetting a
# connection, a flaky link, a DNS hiccup. `[SSL: WRONG_VERSION_NUMBER]` is the
# usual shape when something intercepts TLS mid-handshake.
_TRANSIENT_EXCEPTIONS = (
    ssl.SSLError,
    socket.gaierror,  # DNS
    ConnectionError,
    TimeoutError,  # socket.timeout is an alias of this
    TransportError,
    httplib2.HttpLib2Error,
)

# Rate limiting and Google-side faults. Anything else (401, 403, 404) is a real
# answer and retrying it just wastes time.
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

MAX_ATTEMPTS = 5


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, HttpError):
        return getattr(exc.resp, "status", None) in _TRANSIENT_STATUSES
    return isinstance(exc, _TRANSIENT_EXCEPTIONS)


def with_retries(call: Callable[[], T], *, what: str) -> T:
    """Retry a Gmail call through transient network failure.

    Without this a single dropped connection aborts an entire import — which is
    what turns "run the backfill once" into "run the backfill again and again".
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
            if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                raise
            # Exponential backoff with jitter, so a whole backfill's worth of
            # in-flight calls doesn't retry in lockstep.
            delay = min(2 ** (attempt - 1), 16) + random.uniform(0, 0.5)
            log.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                what,
                attempt,
                MAX_ATTEMPTS,
                str(exc)[:200],
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover

# Headers worth keeping; the full set is noisy and mostly useless downstream.
KEEP_HEADERS = {
    "from",
    "to",
    "cc",
    "subject",
    "date",
    "reply-to",
    "message-id",
    "in-reply-to",
    "references",
    "list-unsubscribe",
    "list-id",
    "precedence",
    "auto-submitted",
    "return-path",
}

MAX_BODY_CHARS = 20_000


@dataclass
class ParsedEmail:
    gmail_id: str
    thread_id: str | None
    history_id: str | None
    from_addr: str | None
    from_name: str | None
    to_addr: str | None
    subject: str | None
    snippet: str | None
    body_text: str | None
    received_at: datetime | None
    labels: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------


def list_message_ids(
    query: str | None = None,
    *,
    max_results: int | None = None,
    label_ids: list[str] | None = None,
) -> list[str]:
    """Return message ids matching a Gmail search query, newest first."""
    service = get_service()
    ids: list[str] = []
    page_token: str | None = None

    while True:
        page_size = 500
        if max_results is not None:
            remaining = max_results - len(ids)
            if remaining <= 0:
                break
            page_size = min(500, remaining)

        req = service.users().messages().list(
            userId="me",
            q=query,
            labelIds=label_ids,
            maxResults=page_size,
            pageToken=page_token,
        )
        resp = with_retries(req.execute, what="messages.list")
        ids.extend(m["id"] for m in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return ids


def get_message(gmail_id: str) -> dict | None:
    """Fetch a full message. Returns None if it has since been deleted."""
    service = get_service()
    req = service.users().messages().get(userId="me", id=gmail_id, format="full")
    try:
        return with_retries(req.execute, what=f"messages.get({gmail_id})")
    except HttpError as exc:
        if exc.resp.status == 404:
            log.info("Message %s no longer exists, skipping", gmail_id)
            return None
        raise


def get_profile() -> dict:
    req = get_service().users().getProfile(userId="me")
    return with_retries(req.execute, what="getProfile")


class HistoryTooOldError(RuntimeError):
    """Gmail expired the history window; a full re-sync is needed."""


def list_history(start_history_id: str) -> tuple[list[str], str | None]:
    """Return (new message ids, latest historyId) since `start_history_id`.

    Raises HistoryTooOldError if Gmail has expired that cursor (HTTP 404).
    """
    service = get_service()
    message_ids: list[str] = []
    latest = start_history_id
    page_token: str | None = None

    while True:
        req = (
            service.users()
            .history()
            .list(
                userId="me",
                startHistoryId=start_history_id,
                historyTypes=["messageAdded"],
                pageToken=page_token,
            )
        )
        try:
            resp = with_retries(req.execute, what="history.list")
        except HttpError as exc:
            if exc.resp.status == 404:
                raise HistoryTooOldError(
                    f"historyId {start_history_id} is no longer available"
                ) from exc
            raise

        for record in resp.get("history", []):
            for added in record.get("messagesAdded", []):
                msg = added.get("message", {})
                mid = msg.get("id")
                # Skip anything Gmail already filed as spam or trash.
                labels = set(msg.get("labelIds", []))
                if mid and not labels & {"SPAM", "TRASH"}:
                    message_ids.append(mid)

        latest = resp.get("historyId", latest)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # History pages can repeat a message id; preserve order, drop duplicates.
    seen: set[str] = set()
    unique = [m for m in message_ids if not (m in seen or seen.add(m))]
    return unique, latest


# --------------------------------------------------------------------------
# MIME parsing
# --------------------------------------------------------------------------


def _decode_part(data: str | None) -> str:
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (binascii.Error, ValueError):
        return ""
    return raw.decode("utf-8", errors="replace")


def _walk_parts(payload: dict, plain: list[str], html: list[str]) -> None:
    mime = (payload.get("mimeType") or "").lower()
    body = payload.get("body", {}) or {}
    parts = payload.get("parts") or []

    # Attachments carry an attachmentId instead of inline data; skip them.
    if body.get("attachmentId"):
        return

    if mime == "text/plain":
        plain.append(_decode_part(body.get("data")))
    elif mime == "text/html":
        html.append(_decode_part(body.get("data")))

    for part in parts:
        _walk_parts(part, plain, html)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "title", "meta"]):
        tag.decompose()
    # Keep link targets — job postings and scheduling links matter.
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        label = a.get_text(" ", strip=True)
        if href and href.startswith("http") and label and href not in label:
            a.replace_with(f"{label} <{href}>")
    return soup.get_text("\n", strip=True)


# Quoted-reply / forwarded-thread boundaries. Everything from the first match
# onward is a copy of an earlier message we have already stored.
_QUOTE_BOUNDARIES = [
    re.compile(r"^\s*On .{5,120}\bwrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*_{10,}\s*$"),
    re.compile(r"^\s*From:\s.+$", re.IGNORECASE),
    re.compile(r"^\s*Sent from my \w+", re.IGNORECASE),
]

_SIGNATURE_BOUNDARY = re.compile(r"^--\s*$")


def strip_quoted(text: str) -> str:
    """Trim quoted replies and signatures.

    These inflate token cost and routinely confuse a classifier into scoring an
    old rejection as if it just arrived.
    """
    lines = text.splitlines()
    cut = len(lines)

    for idx, line in enumerate(lines):
        if any(pat.match(line) for pat in _QUOTE_BOUNDARIES):
            cut = idx
            break
        if _SIGNATURE_BOUNDARY.match(line) and idx > 3:
            cut = idx
            break

    kept = lines[:cut]

    # Drop residual ">" quote blocks.
    kept = [ln for ln in kept if not ln.lstrip().startswith(">")]

    # Collapse runs of blank lines.
    out: list[str] = []
    blank = 0
    for ln in kept:
        if ln.strip():
            blank = 0
            out.append(ln.rstrip())
        else:
            blank += 1
            if blank <= 1:
                out.append("")

    return "\n".join(out).strip()


def parse_message(raw: dict) -> ParsedEmail:
    """Normalise a Gmail `format=full` message into a ParsedEmail."""
    payload = raw.get("payload", {}) or {}

    headers: dict[str, str] = {}
    for h in payload.get("headers", []) or []:
        name = (h.get("name") or "").lower()
        if name in KEEP_HEADERS:
            headers[name] = h.get("value") or ""

    plain_parts: list[str] = []
    html_parts: list[str] = []
    _walk_parts(payload, plain_parts, html_parts)

    plain = "\n".join(p for p in plain_parts if p.strip())
    if plain.strip():
        body = plain
    else:
        body = "\n".join(html_to_text(h) for h in html_parts if h.strip())

    body = strip_quoted(body)
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n[truncated]"

    from_name, from_addr = parseaddr(headers.get("from", ""))
    _, to_addr = parseaddr(headers.get("to", ""))

    received_at: datetime | None = None
    internal = raw.get("internalDate")
    if internal:
        try:
            received_at = datetime.fromtimestamp(int(internal) / 1000, tz=UTC)
        except (TypeError, ValueError, OSError):
            received_at = None

    return ParsedEmail(
        gmail_id=raw["id"],
        thread_id=raw.get("threadId"),
        history_id=str(raw["historyId"]) if raw.get("historyId") else None,
        from_addr=(from_addr or None) and from_addr.lower(),
        from_name=from_name or None,
        to_addr=(to_addr or None) and to_addr.lower(),
        subject=headers.get("subject") or None,
        snippet=raw.get("snippet"),
        body_text=body or None,
        received_at=received_at,
        labels=raw.get("labelIds", []) or [],
        headers=headers,
    )
