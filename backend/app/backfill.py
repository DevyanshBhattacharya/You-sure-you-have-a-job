"""Historical mail import.

Runs in a background thread so a multi-thousand-message import doesn't block
the API. Progress is written to `sync_state` so the dashboard can poll it.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from app import ingest, statestore
from app.db import session_scope
from app.gmail import client as gmail_client

log = logging.getLogger(__name__)

BACKFILL_STATUS = "backfill_status"
BACKFILL_TOTAL = "backfill_total"
BACKFILL_DONE = "backfill_done"
BACKFILL_ERROR = "backfill_error"

_thread: threading.Thread | None = None
_lock = threading.Lock()


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def status() -> dict:
    with session_scope() as session:
        state = statestore.get(session, BACKFILL_STATUS, "idle")
        # A process that died mid-import leaves "listing"/"fetching" behind
        # forever, which reads as though work is still happening.
        if not is_running() and state in ("listing", "fetching"):
            state = "interrupted"
        return {
            "running": is_running(),
            "status": state,
            "total": statestore.get_int(session, BACKFILL_TOTAL),
            "done": statestore.get_int(session, BACKFILL_DONE),
            "error": statestore.get(session, BACKFILL_ERROR),
            "last_history_id": statestore.get(session, statestore.LAST_HISTORY_ID),
        }


def start(days: int, *, on_email=None) -> bool:
    """Kick off a backfill. Returns False if one is already running.

    `on_email` is called with each newly-created Email id, letting the caller
    push it through the agent pipeline.
    """
    global _thread
    with _lock:
        if is_running():
            return False
        _thread = threading.Thread(
            target=_run, args=(days, on_email), name="backfill", daemon=True
        )
        _thread.start()
        return True


def _run(days: int, on_email) -> None:
    try:
        _backfill(days, on_email)
    except Exception as exc:  # noqa: BLE001 - background thread must not die silently
        log.exception("Backfill failed")
        with session_scope() as session:
            statestore.set_(session, BACKFILL_STATUS, "error")
            statestore.set_(session, BACKFILL_ERROR, str(exc))


def _backfill(days: int, on_email) -> None:
    after = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y/%m/%d")
    # Exclude chats, spam and trash; Gmail's `in:anywhere` would drag those in.
    query = f"after:{after} -in:chats"

    with session_scope() as session:
        statestore.set_(session, BACKFILL_STATUS, "listing")
        statestore.set_(session, BACKFILL_ERROR, None)
        statestore.set_(session, BACKFILL_DONE, "0")
        statestore.set_(session, BACKFILL_TOTAL, "0")

    log.info("Backfill: listing messages with query %r", query)
    ids = gmail_client.list_message_ids(query)
    log.info("Backfill: %d messages match", len(ids))

    with session_scope() as session:
        already = ingest.existing_gmail_ids(session, ids)
    pending = [i for i in ids if i not in already]

    with session_scope() as session:
        statestore.set_(session, BACKFILL_STATUS, "fetching")
        statestore.set_(session, BACKFILL_TOTAL, str(len(pending)))

    log.info("Backfill: %d new, %d already stored", len(pending), len(already))

    created_ids: list[int] = []
    done = 0
    for gmail_id in pending:
        try:
            with session_scope() as session:
                email, created = ingest.fetch_and_store(session, gmail_id)
                if created and email is not None:
                    created_ids.append(email.id)
        except Exception:  # noqa: BLE001 - one bad message must not stop the import
            log.exception("Backfill: failed on message %s", gmail_id)

        done += 1
        if done % 25 == 0 or done == len(pending):
            with session_scope() as session:
                statestore.set_(session, BACKFILL_DONE, str(done))

    # Anchor the watcher cursor so incremental sync picks up from here.
    try:
        profile = gmail_client.get_profile()
        with session_scope() as session:
            statestore.set_(session, statestore.LAST_HISTORY_ID, str(profile.get("historyId")))
    except Exception:  # noqa: BLE001
        log.exception("Backfill: could not read profile historyId")

    with session_scope() as session:
        statestore.set_(session, BACKFILL_STATUS, "complete")
        statestore.set_(session, statestore.BACKFILL_COMPLETE, "true")
        statestore.set_(session, statestore.LAST_SYNC_AT, datetime.now(UTC).isoformat())

    log.info("Backfill: stored %d new emails", len(created_ids))

    if on_email is not None:
        for email_id in created_ids:
            try:
                on_email(email_id)
            except Exception:  # noqa: BLE001
                log.exception("Backfill: pipeline handoff failed for email %s", email_id)
