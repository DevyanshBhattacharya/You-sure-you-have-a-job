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

from sqlalchemy.orm import Session

from app import ingest
from app.agent import classify as classify_mod
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


# --------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------


def start_workers(count: int = WORKER_COUNT) -> None:
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


async def _worker(index: int) -> None:
    while True:
        item = await work_queue.get()
        try:
            result = await asyncio.to_thread(process_work, item)
            if result:
                _publish(result)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one bad email must not kill the worker
            log.exception("Worker %d failed on %r", index, item)
        finally:
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
        return result

    outcome = resolve_mod.apply(session, email, verdict, reclassify=reclassify)
    session.flush()

    result["application"] = outcome.application_payload()
    result["notifications"] = [n.payload() for n in outcome.notifications]

    try:
        indexer.index_email(session, email, application_id=outcome.application.id)
    except Exception:  # noqa: BLE001 - a KB failure must not lose the extraction
        log.exception("Indexing failed for email %s", email.id)

    return result
