"""Detects new mail and hands it to the agent.

This is the only module that knows *how* arrival is detected. It polls Gmail's
history API for a cheap incremental delta and publishes one `EmailWork` item
per new message. Adding a Pub/Sub push webhook later means adding a second
publisher that emits the same item — nothing downstream changes.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from app import ingest, statestore
from app.config import get_settings
from app.db import session_scope
from app.events import EmailWork, work_queue
from app.gmail import client as gmail_client
from app.gmail.client import HistoryTooOldError

log = logging.getLogger(__name__)

# How far back to sweep when the history cursor has expired.
RESYNC_WINDOW_DAYS = 3

_thread: threading.Thread | None = None
_stop = threading.Event()
_last_error: str | None = None


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def last_error() -> str | None:
    return _last_error


def start() -> bool:
    """Start the poll loop. Returns False if already running."""
    global _thread
    if is_running():
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="gmail-watcher", daemon=True)
    _thread.start()
    log.info("Gmail watcher started")
    return True


def stop(timeout: float = 5.0) -> None:
    _stop.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=timeout)
    log.info("Gmail watcher stopped")


def _loop() -> None:
    global _last_error
    interval = get_settings().poll_interval_seconds

    while not _stop.is_set():
        try:
            _poll_once()
            _last_error = None
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            _last_error = str(exc)
            log.exception("Watcher poll failed; retrying next cycle")

        _stop.wait(interval)


def _poll_once() -> None:
    from app import backfill

    # A running backfill owns the cursor; polling alongside it would double-fetch.
    if backfill.is_running():
        log.debug("Backfill in progress; skipping poll")
        return

    with session_scope() as session:
        cursor = statestore.get(session, statestore.LAST_HISTORY_ID)

    if not cursor:
        # First run with no backfill: anchor on the current historyId so we
        # start from "now" instead of importing the entire mailbox.
        profile = gmail_client.get_profile()
        with session_scope() as session:
            statestore.set_(session, statestore.LAST_HISTORY_ID, str(profile.get("historyId")))
        log.info("Watcher anchored at historyId %s", profile.get("historyId"))
        return

    try:
        message_ids, latest = gmail_client.list_history(cursor)
    except HistoryTooOldError:
        log.warning("History cursor %s expired; falling back to a window sweep", cursor)
        message_ids, latest = _resync_window()

    if message_ids:
        log.info("Watcher found %d new message(s)", len(message_ids))
        for gmail_id in message_ids:
            work_queue.submit_threadsafe(EmailWork(gmail_id=gmail_id))

    with session_scope() as session:
        if latest:
            statestore.set_(session, statestore.LAST_HISTORY_ID, str(latest))
        statestore.set_(session, statestore.LAST_SYNC_AT, datetime.now(UTC).isoformat())


def _resync_days() -> int:
    """How far back to sweep when the history cursor has expired.

    Gmail keeps roughly a week of history, so a machine that was off for longer
    comes back to a dead cursor. Sweeping a fixed three days would then skip
    every message that arrived in between — silently, and permanently, since
    the cursor is reset afterwards either way. Measure the gap instead, and use
    the fixed window only as a floor.
    """
    with session_scope() as session:
        last = statestore.get(session, statestore.LAST_SYNC_AT)

    if not last:
        return get_settings().backfill_default_days

    try:
        since = datetime.fromisoformat(last)
    except ValueError:
        return get_settings().backfill_default_days

    if since.tzinfo is None:
        since = since.replace(tzinfo=UTC)

    gap = (datetime.now(UTC) - since).days + 1
    # A day of slack for timezone skew, capped so a long-dead install doesn't
    # silently re-import a decade of mail.
    return max(RESYNC_WINDOW_DAYS, min(gap + 1, get_settings().backfill_default_days))


def _resync_window() -> tuple[list[str], str | None]:
    """Recover from an expired cursor by sweeping a recent window."""
    days = _resync_days()
    after = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y/%m/%d")
    log.info("Resync sweeping %d day(s) back to %s", days, after)
    ids = gmail_client.list_message_ids(f"after:{after} -in:chats")

    with session_scope() as session:
        already = ingest.existing_gmail_ids(session, ids)

    pending = [i for i in ids if i not in already]
    latest = str(gmail_client.get_profile().get("historyId"))
    return pending, latest
