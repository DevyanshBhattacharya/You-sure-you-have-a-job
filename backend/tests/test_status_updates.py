"""New mail has to move the application, not just log against it.

The board is only worth reading if a rejection lands on `rejected` without
anyone touching it. A real Sauce Labs rejection — "we have decided not to move
forward with your candidacy" — came back from the model as `event_type: other`
with the summary "Sauce Labs rejected Devyansh for the AI Architect role". The
extraction understood it perfectly and still filed it as nothing, so the
application sat on `applied` indefinitely.
"""

from __future__ import annotations

import pytest

from app.agent.classify import Extraction, Verdict, event_type_of, infer_event_type
from app.agent.resolve import apply, next_status
from app.models import ApplicationStatus, EventType
from tests.fixtures import make_email

REJECTION_BODY = """\
Hi Devyansh,

Thank you for your interest in Sauce Labs regarding the AI Architect role.

At this time, we have decided not to move forward with your candidacy. We
encourage you to stay connected via our Sauce Labs community on LinkedIn.

Best,
The Sauce Labs Recruiting Team
"""


def verdict(**kwargs) -> Verdict:
    kwargs.setdefault("recipient_applied", True)
    extraction = Extraction(is_job_related=True, confidence=0.95, **kwargs)
    return Verdict(
        is_job_related=True,
        confidence=0.95,
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


class TestRecoveringAMisfiledOutcome:
    """`other` is the model's shrug. A stated decision must survive it."""

    def test_the_real_sauce_labs_rejection(self, db):
        email = make_email(db, subject="Sauce Labs Application Update", body=REJECTION_BODY)
        extraction = Extraction(
            is_job_related=True,
            confidence=0.95,
            event_type="other",
            summary="Sauce Labs rejected Devyansh for the AI Architect role.",
        )
        assert infer_event_type(email, extraction) is EventType.REJECTION

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("We regret to inform you that we are moving ahead with others.", EventType.REJECTION),
            ("You have not been selected for this role.", EventType.REJECTION),
            ("We will not be progressing your application further.", EventType.REJECTION),
            ("We are pursuing other candidates at this time.", EventType.REJECTION),
            ("We are delighted to offer you the position.", EventType.OFFER),
            ("Please find your offer of employment attached.", EventType.OFFER),
            ("Your interview is scheduled for Thursday at 3pm.", EventType.INTERVIEW_SCHEDULED),
            ("Please complete the online assessment by Friday.", EventType.ASSESSMENT_SENT),
        ],
    )
    def test_plainly_stated_outcomes(self, db, body, expected):
        email = make_email(db, body=body)
        extraction = Extraction(is_job_related=True, confidence=0.9, event_type="other")
        assert infer_event_type(email, extraction) is expected

    def test_ordinary_mail_is_left_alone(self, db):
        """Silence beats a guess: an unrecognised email stays `other`."""
        email = make_email(db, body="Thanks for your time earlier, speak soon.")
        extraction = Extraction(is_job_related=True, confidence=0.9, event_type="other")
        assert infer_event_type(email, extraction) is None

    def test_a_confident_verdict_is_never_overruled(self, db):
        """These patterns read words; the model reads context. It wins whenever
        it commits to something specific."""
        make_email(db, body=REJECTION_BODY)
        extraction = Extraction(
            is_job_related=True, confidence=0.95, event_type="interview_scheduled"
        )
        # The recovery path only runs for "other"; event_type_of passes it through.
        assert event_type_of(extraction) is EventType.INTERVIEW_SCHEDULED


class TestStatusFollowsTheMail:
    def test_a_rejection_moves_an_open_application_to_rejected(self, db):
        applied = make_email(db, gmail_id="a", thread_id="t", subject="Thanks for applying")
        outcome = apply(db, applied, verdict(company="Sauce Labs", event_type="applied"))
        assert outcome.application.status is ApplicationStatus.APPLIED

        rejected = make_email(
            db, gmail_id="b", thread_id="t", subject="Application Update", body=REJECTION_BODY
        )
        outcome = apply(db, rejected, verdict(company="Sauce Labs", event_type="rejection"))

        assert outcome.application.status is ApplicationStatus.REJECTED
        assert outcome.status_before is ApplicationStatus.APPLIED

    def test_a_rejection_reaches_rejected_from_any_open_stage(self, db):
        for stage in (
            ApplicationStatus.DISCOVERED,
            ApplicationStatus.APPLIED,
            ApplicationStatus.ACKNOWLEDGED,
            ApplicationStatus.SCREENING,
            ApplicationStatus.ASSESSMENT,
            ApplicationStatus.INTERVIEWING,
            ApplicationStatus.OFFER,
        ):
            assert next_status(stage, EventType.REJECTION) is ApplicationStatus.REJECTED

    def test_an_offer_still_beats_a_stale_acknowledgement(self, db):
        """Guard against the fix reintroducing backwards movement."""
        assert next_status(ApplicationStatus.OFFER, EventType.ACKNOWLEDGEMENT) is (
            ApplicationStatus.OFFER
        )
