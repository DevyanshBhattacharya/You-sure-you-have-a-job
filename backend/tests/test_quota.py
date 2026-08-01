"""Quota exhaustion must defer work, never fake a verdict.

A 429 is transient. Substituting a heuristic guess and stamping `processed_at`
freezes a low-confidence result built from the sender's name, and because
nothing re-examines processed mail, that bad verdict is permanent.
"""

from __future__ import annotations

import pytest

from app.agent import classify as classify_mod
from app.agent import llm, pipeline
from app.agent.providers import gemini as gemini_provider
from app.models import Application, Email
from tests.fixtures import make_email

QUOTA_BODY = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current "
    "quota', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaValue': '20'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '57s'}]}}"
)


class TestParsing:
    """Quota parsing is Gemini's error dialect, so it lives with that provider."""

    def test_extracts_retry_delay_and_daily_limit(self):
        retry_after, limit = gemini_provider.parse_quota_error(QUOTA_BODY)
        assert retry_after == 57.0
        assert limit == 20

    def test_missing_fields_are_none(self):
        assert gemini_provider.parse_quota_error("plain 429") == (None, None)


class TestClassifyPropagates:
    def test_quota_error_is_not_downgraded_to_a_heuristic(self, db, monkeypatch):
        monkeypatch.setattr(llm, "is_configured", lambda: True)
        monkeypatch.setattr(
            classify_mod.llm,
            "generate_json",
            lambda **_kw: (_ for _ in ()).throw(llm.QuotaExceededError("quota", retry_after=57)),
        )

        email = make_email(db, subject="Interview invitation for Backend Engineer")

        with pytest.raises(llm.QuotaExceededError):
            classify_mod.classify(db, email)

    def test_other_api_errors_still_fall_back(self, db, monkeypatch):
        """A one-off blip should not strand the email — only quota does."""
        monkeypatch.setattr(llm, "is_configured", lambda: True)
        monkeypatch.setattr(
            classify_mod.llm,
            "generate_json",
            lambda **_kw: (_ for _ in ()).throw(RuntimeError("transient blip")),
        )

        email = make_email(db, subject="Interview invitation for Backend Engineer")
        verdict = classify_mod.classify(db, email)

        assert verdict.source == "heuristic"


class TestPipelineLeavesWorkUndone:
    def test_email_stays_unprocessed_so_it_can_be_retried(self, db, monkeypatch):
        monkeypatch.setattr(
            pipeline.classify_mod,
            "classify",
            lambda _s, _e: (_ for _ in ()).throw(llm.QuotaExceededError("quota", retry_after=57)),
        )

        email = make_email(db, subject="Offer from Acme")
        db.commit()

        with pytest.raises(llm.QuotaExceededError):
            pipeline.process_email(db, email)
        db.rollback()

        refreshed = db.get(Email, email.id)
        assert refreshed.processed_at is None, "must remain eligible for a retry"
        assert db.query(Application).count() == 0, "no junk application from a failed call"

    def test_enqueue_unprocessed_finds_deferred_mail(self, db, monkeypatch):
        make_email(db, gmail_id="a")
        make_email(db, gmail_id="b")
        processed = make_email(db, gmail_id="c")
        processed.processed_at = __import__("app.models", fromlist=["utcnow"]).utcnow()
        db.commit()

        submitted: list[int] = []
        monkeypatch.setattr(pipeline, "submit_email_id", lambda i, **_k: submitted.append(i))

        assert pipeline.enqueue_unprocessed() == 2
        assert len(submitted) == 2

    def test_count_unprocessed_matches(self, db):
        make_email(db, gmail_id="a")
        make_email(db, gmail_id="b")
        db.commit()
        assert pipeline.count_unprocessed() == 2

    def test_job_signal_mail_is_queued_first(self, db, monkeypatch):
        """With ~15 calls/day of free quota, arrival order would spend the lot
        on newsletters before reaching an interview invite."""
        make_email(db, gmail_id="new-noise", subject="Your parcel shipped", days_ago=0)
        make_email(
            db,
            gmail_id="old-job",
            subject="Interview invitation - Backend Engineer",
            sender="recruiting@acme.com",
            days_ago=30,
        )
        db.commit()

        submitted: list[int] = []
        monkeypatch.setattr(pipeline, "submit_email_id", lambda i, **_k: submitted.append(i))
        pipeline.enqueue_unprocessed()

        first = db.get(Email, submitted[0])
        assert first.gmail_id == "old-job", "job-signal mail must outrank newer noise"


class TestQuotaState:
    def test_block_is_recorded_and_expires(self, monkeypatch):
        pipeline._quota_blocked_until = 0.0
        pipeline._note_quota_block(llm.QuotaExceededError("limit 20/day", retry_after=57))

        state = pipeline.quota_state()
        assert state["blocked"] is True
        assert 0 < state["retry_in_seconds"] <= 57
        assert "20/day" in state["reason"]

        pipeline._quota_blocked_until = 0.0
        assert pipeline.quota_state()["blocked"] is False

    def test_missing_retry_after_backs_off_for_an_hour(self):
        """A per-day limit reports no delay; retrying in seconds is pointless."""
        pipeline._quota_blocked_until = 0.0
        pipeline._note_quota_block(llm.QuotaExceededError("daily limit"))

        assert pipeline.quota_state()["retry_in_seconds"] > 3500
        pipeline._quota_blocked_until = 0.0
