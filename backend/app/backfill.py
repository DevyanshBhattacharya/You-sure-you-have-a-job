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
BACKFILL_DAYS = "backfill_days"

# Gmail calls already retry transient faults individually (see gmail.client).
# Reaching this many failures in a row means retrying didn't help.
MAX_CONSECUTIVE_FAILURES = 10

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
            "last_sync_at": statestore.get(session, statestore.LAST_SYNC_AT),
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


def resume_if_interrupted(*, on_email=None) -> bool:
    """Restart an import that a crash or network failure left unfinished.

    Already-stored messages are skipped on the next pass, so this picks up where
    it stopped rather than starting over.
    """
    from app.config import get_settings

    with session_scope() as session:
        state = statestore.get(session, BACKFILL_STATUS, "idle")
        days = statestore.get_int(session, BACKFILL_DAYS) or get_settings().backfill_default_days
        error = statestore.get(session, BACKFILL_ERROR)

    # "listing"/"fetching" without a live thread means the process died mid-import.
    if state not in ("error", "listing", "fetching"):
        return False

    log.info(
        "Resuming an import that ended as %r (%s); window %d days",
        state,
        (error or "no error recorded")[:120],
        days,
    )
    return start(days, on_email=on_email)


def start_if_never_run(*, on_email=None) -> bool:
    """Run the first import automatically. Returns False if one already happened.

    Without this the app sits empty until someone notices the "Import mail"
    button, and every restart looks like it did nothing. Keyed on
    `backfill_complete` rather than on the row count, so a mailbox that
    genuinely has no job mail isn't re-imported on every boot.
    """
    from app.config import get_settings

    with session_scope() as session:
        if statestore.get(session, statestore.BACKFILL_COMPLETE) == "true":
            return False
        days = statestore.get_int(session, BACKFILL_DAYS) or get_settings().backfill_default_days

    log.info("No import has completed yet; starting one automatically (%d days)", days)
    return start(days, on_email=on_email)


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
        # Remember the window so an interrupted import can resume over the same
        # range instead of guessing a default.
        statestore.set_(session, BACKFILL_DAYS, str(days))

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

    created = 0
    done = 0
    consecutive_failures = 0

    for gmail_id in pending:
        try:
            with session_scope() as session:
                email, was_created = ingest.fetch_and_store(session, gmail_id)
                email_id = email.id if (was_created and email is not None) else None
            consecutive_failures = 0
        except Exception:  # noqa: BLE001 - one bad message must not stop the import
            log.exception("Backfill: failed on message %s", gmail_id)
            email_id = None
            consecutive_failures += 1
            # Individual failures are expected; a long unbroken run of them means
            # the network or the token is gone, and grinding through thousands
            # more messages just to fail on each is pointless. Stop and keep the
            # progress so far — the next run resumes from here.
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Stopped after {consecutive_failures} consecutive failures "
                    f"at message {done + 1} of {len(pending)}. "
                    "Progress is saved; start the import again to resume."
                ) from None

        # Hand each email to the agent as it lands, rather than batching the
        # whole import to the end. A crash then costs the remaining messages,
        # not the classification of every message already fetched.
        if email_id is not None:
            created += 1
            if on_email is not None:
                try:
                    on_email(email_id)
                except Exception:  # noqa: BLE001
                    log.exception("Backfill: pipeline handoff failed for email %s", email_id)

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

    log.info("Backfill: stored %d new emails", created)
