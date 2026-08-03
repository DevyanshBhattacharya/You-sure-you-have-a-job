"""The agent pipeline.

Consumes `EmailWork` items and runs each through:

    ingest -> prefilter -> classify -> resolve -> index -> notify

Stages are synchronous (SQLAlchemy and the Gemini SDK are both blocking), so
each item is processed in a worker thread while the event loop stays free to
serve HTTP and WebSocket traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy.orm import Session

from app import ingest
from app.agent import classify as classify_mod
from app.agent import llm
from app.agent import resolve as resolve_mod
from app.db import session_scope
from app.events import (
    APPLICATION_UPDATED,
    EMAIL_PROCESSED,
    NOTIFICATION_CREATED,
    EmailWork,
    bus,
    work_queue,
)
from app.kb import indexer
from app.models import Email, utcnow

log = logging.getLogger(__name__)

WORKER_COUNT = 2

_tasks: list[asyncio.Task] = []


# --------------------------------------------------------------------------
# Submission helpers (safe to call from any thread)
# --------------------------------------------------------------------------


def submit_email_id(email_id: int, *, reclassify: bool = False) -> None:
    work_queue.submit_threadsafe(EmailWork(email_id=email_id, reclassify=reclassify))


def submit_gmail_id(gmail_id: str) -> None:
    work_queue.submit_threadsafe(EmailWork(gmail_id=gmail_id))


def count_unprocessed() -> int:
    from sqlalchemy import func, select

    with session_scope() as session:
        return (
            session.scalar(
                select(func.count()).select_from(Email).where(Email.processed_at.is_(None))
            )
            or 0
        )


def enqueue_unprocessed(limit: int = 1000) -> int:
    """Queue stored-but-unclassified mail. Returns how many were queued.

    The work queue lives in memory, so anything in flight when the process
    stops is lost — a backfill interrupted halfway leaves rows stranded with
    `processed_at IS NULL` and nothing to pick them up again. Sweeping for
    those on startup makes the pipeline recoverable rather than best-effort.
    """
    from sqlalchemy import select

    from app.agent import prefilter

    with session_scope() as session:
        rows = list(
            session.scalars(
                select(Email)
                .where(Email.processed_at.is_(None))
                .order_by(Email.received_at.desc().nullslast())
            ).all()
        )

        # Spend the day's quota on the most promising mail first. Free Gemini
        # tiers allow only a handful of calls per day, so draining a large
        # backlog in arrival order would burn the entire allowance on
        # newsletters before reaching a single interview invite.
        rows.sort(
            key=lambda e: (
                not prefilter.has_job_signal(e),
                -(e.received_at.timestamp() if e.received_at else 0),
            )
        )
        ids = [e.id for e in rows[:limit]]

    for email_id in ids:
        submit_email_id(email_id)

    if ids:
        log.info("Queued %d unprocessed email(s) for classification", len(ids))
    return len(ids)


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------


async def _backlog_sweeper(interval: float, limit: int) -> None:
    """Re-queue stored-but-unclassified mail, forever.

    The startup sweep alone isn't enough. It is capped at `limit`, so a backlog
    larger than that keeps a remainder that nothing picks up; and the work queue
    is in memory, so anything queued when the process stops is lost while its
    row stays `processed_at IS NULL`. Either way the dashboard shows mail that
    is imported but never classified, and the only cure is a restart — which is
    exactly the "press Import again" loop this is meant to end.

    Only sweeps once the queue has drained, so it never stacks duplicates on
    top of work already in flight.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            if work_queue.qsize() > 0:
                continue
            queued = await asyncio.to_thread(enqueue_unprocessed, limit)
            if queued:
                log.info("Backlog sweep queued %d email(s)", queued)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a sweep failure must not end the loop
            log.exception("Backlog sweep failed")


def start_workers(count: int = WORKER_COUNT) -> None:
    from app.config import get_settings

    loop = asyncio.get_running_loop()

    # Drop handles left over from a previous loop (a second app instance in the
    # same process). Awaiting those would raise rather than shut down cleanly.
    survivors = [t for t in _tasks if not t.done() and t.get_loop() is loop]
    _tasks.clear()
    _tasks.extend(survivors)
    if _tasks:
        return

    for i in range(count):
        _tasks.append(asyncio.create_task(_worker(i), name=f"pipeline-worker-{i}"))
    log.info("Started %d pipeline worker(s)", count)

    settings = get_settings()
    if settings.backlog_sweep_seconds > 0:
        _tasks.append(
            asyncio.create_task(
                _backlog_sweeper(settings.backlog_sweep_seconds, settings.backlog_batch_limit),
                name="backlog-sweeper",
            )
        )


async def stop_workers() -> None:
    loop = asyncio.get_running_loop()
    ours = [t for t in _tasks if t.get_loop() is loop]

    for task in ours:
        task.cancel()
    for task in ours:
        try:
            await task
        except asyncio.CancelledError:
            pass

    _tasks.clear()
    log.info("Pipeline workers stopped")


# Set when the API reports exhausted quota. Until it clears, workers drain the
# queue without doing work — the emails stay unprocessed in the database and
# are re-queued on the next start, so nothing is lost by dropping them here.
_quota_blocked_until: float = 0.0
_quota_reason: str = ""
_quota_deferred: int = 0


def quota_state() -> dict:
    remaining = max(0.0, _quota_blocked_until - time.monotonic())
    return {
        "blocked": remaining > 0,
        "retry_in_seconds": int(remaining),
        "reason": _quota_reason if remaining > 0 else "",
        "deferred": _quota_deferred,
    }


def _note_quota_block(exc: llm.DeferWorkError) -> None:
    global _quota_blocked_until, _quota_reason, _quota_deferred
    # No stated delay means a per-day quota; back off for an hour and let the
    # startup sweep pick things up later.
    wait = exc.retry_after if exc.retry_after else 3600.0
    _quota_blocked_until = max(_quota_blocked_until, time.monotonic() + wait)
    _quota_reason = str(exc)
    _quota_deferred = 0
    log.error("LLM unavailable (%s). Deferring classification for %.0fs.", exc, wait)


async def _worker(index: int) -> None:
    global _quota_deferred
    while True:
        item = await work_queue.get()
        try:
            if time.monotonic() < _quota_blocked_until:
                _quota_deferred += 1
                continue  # left unprocessed on purpose; re-queued next start

            result = await asyncio.to_thread(process_work, item)
            if result:
                _publish(result)
        except asyncio.CancelledError:
            raise
        except llm.DeferWorkError as exc:
            _note_quota_block(exc)
            _quota_deferred += 1
        except Exception:  # noqa: BLE001 - one bad email must not kill the worker
            log.exception("Worker %d failed on %r", index, item)
        finally:
            # Outside the item's transaction, so the counter write cannot
            # deadlock against it on SQLite.
            await asyncio.to_thread(llm.flush_usage)
            work_queue.task_done()


def _publish(result: dict) -> None:
    bus.publish(EMAIL_PROCESSED, result["email"])
    if result.get("application"):
        bus.publish(APPLICATION_UPDATED, result["application"])
    for notification in result.get("notifications", []):
        bus.publish(NOTIFICATION_CREATED, notification)


# --------------------------------------------------------------------------
# The pipeline itself (synchronous; runs in a worker thread)
# --------------------------------------------------------------------------


def process_work(item: EmailWork) -> dict | None:
    with session_scope() as session:
        email = _load_or_fetch(session, item)
        if email is None:
            return None

        if email.processed_at is not None and not item.reclassify:
            log.debug("Email %s already processed; skipping", email.id)
            return None

        return process_email(session, email, reclassify=item.reclassify)


def _load_or_fetch(session: Session, item: EmailWork) -> Email | None:
    if item.email_id is not None:
        return session.get(Email, item.email_id)
    if item.gmail_id:
        email, _created = ingest.fetch_and_store(session, item.gmail_id)
        return email
    log.warning("Work item with neither email_id nor gmail_id: %r", item)
    return None


def process_email(session: Session, email: Email, *, reclassify: bool = False) -> dict:
    """Run one stored email through classification and downstream stages."""
    verdict = classify_mod.classify(session, email)

    email.is_job_related = verdict.is_job_related
    email.classification_confidence = verdict.confidence
    email.classification_source = verdict.source
    email.classification_raw = verdict.raw
    email.processed_at = utcnow()

    result: dict = {
        "email": {
            "id": email.id,
            "subject": email.subject,
            "from_addr": email.from_addr,
            "from_name": email.from_name,
            "received_at": email.received_at.isoformat() if email.received_at else None,
            "is_job_related": email.is_job_related,
            "confidence": email.classification_confidence,
        },
        "application": None,
        "notifications": [],
    }

    if not verdict.is_job_related:
        if reclassify:
            # A correction has to undo the previous verdict's work, or a fixed
            # classifier can never clean up after a broken one.
            result["retracted_applications"] = resolve_mod.retract(session, email)
            # Including its searchable text: indexing only ever ran on the way
            # in, so a corrected email stayed in the knowledge base and the Q&A
            # agent went on quoting it as evidence about the job search.
            indexer.remove_email(session, email)
        return result

    outcome = resolve_mod.apply(session, email, verdict, reclassify=reclassify)
    session.flush()

    if outcome is None:
        # Job-related, but not an application this person made — a job alert, an
        # advert, cold outreach. It stays searchable in the knowledge base and
        # visible in the inbox; it just doesn't reach the board.
        if reclassify:
            result["retracted_applications"] = resolve_mod.retract(session, email)
        result["tracked"] = False
    else:
        result["tracked"] = True
        result["application"] = outcome.application_payload()
        result["notifications"] = [n.payload() for n in outcome.notifications]

    application_id = outcome.application.id if outcome is not None else None
    try:
        indexer.index_email(session, email, application_id=application_id)
    except Exception:  # noqa: BLE001 - a KB failure must not lose the extraction
        log.exception("Indexing failed for email %s", email.id)

    return result
