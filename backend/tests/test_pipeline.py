"""End-to-end pipeline behaviour with the LLM stubbed out."""

from __future__ import annotations

import pytest

from app.agent import classify as classify_mod
from app.agent import pipeline
from app.agent.classify import Extraction, Verdict
from app.models import Application, ApplicationEvent, ApplicationStatus, KBChunk, Notification
from tests.fixtures import make_email


def canned(**kwargs) -> Verdict:
    job_related = kwargs.pop("is_job_related", True)
    extraction = Extraction(is_job_related=job_related, confidence=0.93, **kwargs)
    return Verdict(
        is_job_related=job_related,
        confidence=0.93,
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


@pytest.fixture
def stub_classifier(monkeypatch):
    """Install a canned verdict for the next classify() call."""

    def install(verdict: Verdict):
        monkeypatch.setattr(classify_mod, "classify", lambda session, email: verdict)
        monkeypatch.setattr(pipeline.classify_mod, "classify", lambda session, email: verdict)

    return install


class TestJobRelatedEmail:
    def test_creates_application_event_and_notification(self, db, stub_classifier):
        stub_classifier(
            canned(
                company="Acme",
                role_title="Backend Engineer",
                event_type="interview_scheduled",
                summary="Interview on the 14th",
            )
        )
        email = make_email(db, subject="Interview invitation")

        result = pipeline.process_email(db, email)
        db.commit()

        assert result["email"]["is_job_related"] is True
        assert db.query(Application).count() == 1
        assert db.query(ApplicationEvent).count() == 1
        assert db.query(Notification).count() == 1

        app = db.query(Application).one()
        assert app.company == "Acme"
        assert app.status is ApplicationStatus.INTERVIEWING

    def test_marks_the_email_processed(self, db, stub_classifier):
        stub_classifier(canned(company="Acme", event_type="applied"))
        email = make_email(db)

        pipeline.process_email(db, email)
        db.commit()

        assert email.processed_at is not None
        assert email.classification_source == "llm"
        assert email.classification_confidence == pytest.approx(0.93)

    def test_indexes_the_body_into_the_knowledge_base(self, db, stub_classifier):
        stub_classifier(canned(company="Acme", event_type="applied"))
        email = make_email(db, body="We would like to schedule a technical screen next week.")

        pipeline.process_email(db, email)
        db.commit()

        chunks = db.query(KBChunk).all()
        assert len(chunks) == 1
        assert "technical screen" in chunks[0].text
        # No API key in tests, so text is stored but not embedded.
        assert chunks[0].embedding is None

    def test_result_payload_carries_what_the_dashboard_needs(self, db, stub_classifier):
        stub_classifier(
            canned(company="Acme", role_title="Backend Engineer", event_type="offer")
        )
        email = make_email(db)

        result = pipeline.process_email(db, email)

        assert result["application"]["company"] == "Acme"
        assert result["application"]["status"] == "offer"
        assert result["notifications"][0]["priority"] == "high"


class TestNonJobEmail:
    def test_creates_nothing_downstream(self, db, stub_classifier):
        stub_classifier(canned(is_job_related=False))
        email = make_email(db, subject="Your parcel is out for delivery")

        result = pipeline.process_email(db, email)
        db.commit()

        assert result["email"]["is_job_related"] is False
        assert result["application"] is None
        assert db.query(Application).count() == 0
        assert db.query(Notification).count() == 0
        assert db.query(KBChunk).count() == 0

    def test_is_still_marked_processed(self, db, stub_classifier):
        stub_classifier(canned(is_job_related=False))
        email = make_email(db)

        pipeline.process_email(db, email)
        db.commit()

        assert email.processed_at is not None


class TestIdempotency:
    def test_second_pass_does_not_duplicate_rows(self, db, stub_classifier):
        stub_classifier(canned(company="Acme", event_type="applied"))
        email = make_email(db)

        pipeline.process_email(db, email)
        db.commit()
        pipeline.process_email(db, email, reclassify=True)
        db.commit()

        assert db.query(Application).count() == 1
        assert db.query(ApplicationEvent).count() == 1
        # Reclassifying is a correction, not a new event worth alerting on.
        assert db.query(Notification).count() == 1

    def test_reindexing_replaces_rather_than_appends_chunks(self, db, stub_classifier):
        stub_classifier(canned(company="Acme", event_type="applied"))
        email = make_email(db, body="A short body.")

        pipeline.process_email(db, email)
        db.commit()
        before = db.query(KBChunk).count()

        pipeline.process_email(db, email, reclassify=True)
        db.commit()

        assert db.query(KBChunk).count() == before


class TestThreadedConversation:
    def test_a_reply_chain_stays_one_application(self, db, stub_classifier):
        stub_classifier(canned(company="Acme", role_title="SRE", event_type="applied"))
        first = make_email(db, gmail_id="a", thread_id="thread-x")
        pipeline.process_email(db, first)
        db.commit()

        stub_classifier(canned(company="Acme", role_title="SRE", event_type="offer"))
        second = make_email(db, gmail_id="b", thread_id="thread-x")
        pipeline.process_email(db, second)
        db.commit()

        assert db.query(Application).count() == 1
        assert db.query(ApplicationEvent).count() == 2
        assert db.query(Application).one().status is ApplicationStatus.OFFER


class TestClassifierFailureHandling:
    def test_a_classifier_exception_does_not_lose_the_email(self, db, monkeypatch):
        """Without a key configured, classify() falls back to the heuristic
        rather than raising — an API blip must never drop mail."""
        email = make_email(
            db, subject="Interview invitation for the Backend Engineer role"
        )

        result = pipeline.process_email(db, email)
        db.commit()

        assert email.processed_at is not None
        assert email.classification_source == "heuristic"
        assert result["email"]["is_job_related"] is True
