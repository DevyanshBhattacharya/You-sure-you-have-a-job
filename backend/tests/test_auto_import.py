"""Importing without being asked twice.

Every path here exists because the app otherwise looked idle while there was
work left, and the only affordance on screen was to press "Import mail" again —
which correctly did nothing, because the messages had already been fetched.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app import backfill, statestore
from app.agent import pipeline
from app.config import get_settings
from app.gmail import watcher


@pytest.fixture
def started(monkeypatch):
    """Capture what would be launched, without touching Gmail."""
    calls: list[int] = []
    monkeypatch.setattr(
        backfill, "start", lambda days, on_email=None: (calls.append(days), True)[1]
    )
    return calls


class TestFirstImportIsAutomatic:
    """A fresh install should fill itself in, not wait to be discovered."""

    def test_imports_when_nothing_has_ever_completed(self, db, started):
        assert backfill.start_if_never_run() is True
        assert started == [get_settings().backfill_default_days]

    def test_does_not_repeat_a_completed_import(self, db, started):
        statestore.set_(db, statestore.BACKFILL_COMPLETE, "true")
        db.commit()
        assert backfill.start_if_never_run() is False
        assert started == []

    def test_an_empty_mailbox_is_not_re_imported_every_boot(self, db, started):
        """Keyed on the completion flag, not the row count — a mailbox with no
        job mail at all still counts as imported."""
        statestore.set_(db, statestore.BACKFILL_COMPLETE, "true")
        statestore.set_(db, backfill.BACKFILL_TOTAL, "0")
        db.commit()
        assert backfill.start_if_never_run() is False

    def test_reuses_the_window_from_a_previous_attempt(self, db, started):
        statestore.set_(db, backfill.BACKFILL_DAYS, "365")
        db.commit()
        backfill.start_if_never_run()
        assert started == [365]


class TestExpiredCursorSweepsTheRealGap:
    """Gmail keeps about a week of history.

    A machine off for longer comes back to a dead cursor. Sweeping a fixed
    three days would skip everything that arrived in between — silently and
    permanently, because the cursor is reset afterwards either way.
    """

    def _set_sync(self, db, *, days_ago: float) -> None:
        when = datetime.now(UTC) - timedelta(days=days_ago)
        statestore.set_(db, statestore.LAST_SYNC_AT, when.isoformat())
        db.commit()

    def test_covers_a_long_absence(self, db):
        self._set_sync(db, days_ago=30)
        assert watcher._resync_days() >= 30

    def test_short_gaps_still_use_the_floor(self, db):
        self._set_sync(db, days_ago=0.5)
        assert watcher._resync_days() == watcher.RESYNC_WINDOW_DAYS

    def test_a_very_old_install_is_capped(self, db):
        self._set_sync(db, days_ago=5000)
        assert watcher._resync_days() == get_settings().backfill_default_days

    def test_no_recorded_sync_falls_back_to_the_default_window(self, db):
        assert watcher._resync_days() == get_settings().backfill_default_days

    def test_a_corrupt_timestamp_does_not_crash_the_watcher(self, db):
        statestore.set_(db, statestore.LAST_SYNC_AT, "not a date")
        db.commit()
        assert watcher._resync_days() == get_settings().backfill_default_days


class TestBacklogKeepsDraining:
    """The startup sweep alone leaves work stranded.

    It is capped at `backlog_batch_limit`, and the work queue is in memory, so
    anything queued when the process stops is lost while its row stays
    `processed_at IS NULL`. Without a repeating sweep the only cure is a
    restart.
    """

    def _run_one_sweep(self, monkeypatch, *, qsize: int) -> list[int]:
        swept: list[int] = []
        monkeypatch.setattr(pipeline.work_queue, "qsize", lambda: qsize)
        monkeypatch.setattr(
            pipeline, "enqueue_unprocessed", lambda limit: (swept.append(limit), 0)[1]
        )

        async def drive() -> None:
            task = asyncio.create_task(pipeline._backlog_sweeper(0.01, 500))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(drive())
        return swept

    def test_sweeps_once_the_queue_is_empty(self, monkeypatch):
        assert self._run_one_sweep(monkeypatch, qsize=0)

    def test_does_not_stack_work_on_top_of_a_busy_queue(self, monkeypatch):
        assert self._run_one_sweep(monkeypatch, qsize=7) == []

    def test_a_failed_sweep_does_not_end_the_loop(self, monkeypatch):
        attempts = {"n": 0}

        def boom(limit):
            attempts["n"] += 1
            raise RuntimeError("database is locked")

        monkeypatch.setattr(pipeline.work_queue, "qsize", lambda: 0)
        monkeypatch.setattr(pipeline, "enqueue_unprocessed", boom)

        async def drive() -> None:
            task = asyncio.create_task(pipeline._backlog_sweeper(0.01, 500))
            await asyncio.sleep(0.08)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(drive())
        assert attempts["n"] > 1
