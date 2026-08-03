"""Matching emails to applications, and merging extracted detail."""

from __future__ import annotations

from app.agent.classify import Extraction, Verdict
from app.agent.resolve import (
    apply,
    find_application,
    normalize_company,
    normalize_role,
    parse_iso,
)
from app.models import Application, ApplicationStatus
from tests.fixtures import make_email


def verdict(**kwargs) -> Verdict:
    # `recipient_applied` defaults true here because these tests are about what
    # resolve does *once* an email is established as the user's own application.
    # The gate that decides that is exercised in TestOnlyOwnApplications below.
    kwargs.setdefault("recipient_applied", True)
    extraction = Extraction(is_job_related=True, confidence=0.95, **kwargs)
    return Verdict(
        is_job_related=True,
        confidence=0.95,
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


class TestNormalisation:
    def test_company_strips_legal_suffixes(self):
        assert normalize_company("Acme Technologies Pvt. Ltd.") == "acme"
        assert normalize_company("Acme Inc") == "acme"
        assert normalize_company("ACME") == "acme"

    def test_company_variants_collapse_together(self):
        assert normalize_company("Acme Corp") == normalize_company("acme corporation")

    def test_role_strips_seniority(self):
        assert normalize_role("Senior Software Engineer") == "software engineer"
        assert normalize_role("Sr. Software Engineer II") == "software engineer"

    def test_empty_input(self):
        assert normalize_company(None) == ""
        assert normalize_role("") == ""


class TestParseISO:
    def test_parses_date_only(self):
        assert parse_iso("2026-04-15").year == 2026

    def test_parses_datetime_with_z(self):
        dt = parse_iso("2026-04-15T14:30:00Z")
        assert dt.hour == 14
        assert dt.tzinfo is not None

    def test_returns_none_for_junk(self):
        assert parse_iso("next Tuesday") is None
        assert parse_iso("") is None
        assert parse_iso(None) is None


class TestMatching:
    def test_creates_application_when_none_exists(self, db):
        email = make_email(db, subject="Application received")
        outcome = apply(db, email, verdict(company="Acme", role_title="Backend Engineer"))
        assert outcome.created_application is True
        assert outcome.application.company == "Acme"
        assert outcome.application.company_normalized == "acme"

    def test_matches_existing_by_company_and_role(self, db):
        first = make_email(db, gmail_id="a", thread_id="t1")
        outcome1 = apply(db, first, verdict(company="Acme Inc", role_title="Backend Engineer"))

        second = make_email(db, gmail_id="b", thread_id="t2")
        outcome2 = apply(db, second, verdict(company="Acme", role_title="Sr Backend Engineer"))

        assert outcome2.created_application is False
        assert outcome2.application.id == outcome1.application.id

    def test_matches_by_thread_even_when_company_differs(self, db):
        """A reply chain is unambiguously one application, whatever the model
        decides to call the company in a follow-up message."""
        first = make_email(db, gmail_id="a", thread_id="shared")
        outcome1 = apply(db, first, verdict(company="Acme", role_title="Backend Engineer"))

        reply = make_email(db, gmail_id="b", thread_id="shared")
        outcome2 = apply(db, reply, verdict(company="", role_title=""))

        assert outcome2.application.id == outcome1.application.id

    def test_fuzzy_company_match(self, db):
        first = make_email(db, gmail_id="a", thread_id="t1")
        outcome1 = apply(db, first, verdict(company="Cloudflare", role_title="SRE"))

        second = make_email(db, gmail_id="b", thread_id="t2")
        outcome2 = apply(db, second, verdict(company="Cloudflaire", role_title="SRE"))

        assert outcome2.application.id == outcome1.application.id

    def test_different_companies_stay_separate(self, db):
        a = make_email(db, gmail_id="a", thread_id="t1")
        apply(db, a, verdict(company="Acme", role_title="Backend Engineer"))

        b = make_email(db, gmail_id="b", thread_id="t2")
        apply(db, b, verdict(company="Globex", role_title="Backend Engineer"))

        assert db.query(Application).count() == 2

    def test_same_company_different_roles_stay_separate(self, db):
        a = make_email(db, gmail_id="a", thread_id="t1")
        apply(db, a, verdict(company="Acme", role_title="Backend Engineer"))

        b = make_email(db, gmail_id="b", thread_id="t2")
        apply(db, b, verdict(company="Acme", role_title="Product Designer"))

        assert db.query(Application).count() == 2

    def test_no_match_without_company_or_thread(self, db):
        email = make_email(db, gmail_id="x", thread_id="lonely")
        assert find_application(db, email, None) is None


class TestMergingDetail:
    def test_fills_blanks_without_overwriting(self, db):
        first = make_email(db, gmail_id="a", thread_id="t")
        outcome = apply(
            db,
            first,
            verdict(company="Acme", role_title="Backend Engineer", location="Bengaluru"),
        )
        app_id = outcome.application.id

        second = make_email(db, gmail_id="b", thread_id="t")
        apply(db, second, verdict(company="Acme", role_title="Backend Engineer", location="Remote"))

        app = db.get(Application, app_id)
        assert app.location == "Bengaluru"  # first value wins

    def test_newest_next_action_replaces_older_one(self, db):
        first = make_email(db, gmail_id="a", thread_id="t")
        outcome = apply(
            db, first, verdict(company="Acme", next_action="Confirm your availability")
        )
        app_id = outcome.application.id

        second = make_email(db, gmail_id="b", thread_id="t")
        apply(
            db,
            second,
            verdict(company="Acme", next_action="Complete the assessment", deadline="2026-04-20"),
        )

        app = db.get(Application, app_id)
        assert app.next_action == "Complete the assessment"
        assert app.next_action_due is not None
        assert app.next_action_due.strftime("%Y-%m-%d") == "2026-04-20"


class TestStatusProgression:
    def test_status_advances_through_the_pipeline(self, db):
        email1 = make_email(db, gmail_id="a", thread_id="t")
        out = apply(db, email1, verdict(company="Acme", event_type="applied"))
        assert out.application.status is ApplicationStatus.APPLIED

        email2 = make_email(db, gmail_id="b", thread_id="t")
        out = apply(db, email2, verdict(company="Acme", event_type="interview_scheduled"))
        assert out.application.status is ApplicationStatus.INTERVIEWING

    def test_late_acknowledgement_cannot_undo_progress(self, db):
        email1 = make_email(db, gmail_id="a", thread_id="t")
        apply(db, email1, verdict(company="Acme", event_type="offer"))

        email2 = make_email(db, gmail_id="b", thread_id="t")
        out = apply(db, email2, verdict(company="Acme", event_type="acknowledgement"))

        assert out.application.status is ApplicationStatus.OFFER
        assert out.status_before is ApplicationStatus.OFFER


class TestIdempotency:
    def test_reprocessing_the_same_email_reuses_one_timeline_entry(self, db):
        email = make_email(db, gmail_id="a", thread_id="t")
        v = verdict(company="Acme", event_type="applied")

        first = apply(db, email, v)
        second = apply(db, email, v, reclassify=True)

        assert first.event.id == second.event.id
        assert len(second.application.events) == 1

    def test_notifications_are_not_duplicated_on_reclassify(self, db):
        email = make_email(db, gmail_id="a", thread_id="t")
        v = verdict(company="Acme", event_type="offer")

        first = apply(db, email, v)
        second = apply(db, email, v, reclassify=True)

        assert len(first.notifications) == 1
        assert second.notifications == []


class TestNotifications:
    def test_offer_is_high_priority(self, db):
        email = make_email(db)
        out = apply(db, email, verdict(company="Acme", event_type="offer"))
        assert out.notifications[0].row.priority == "high"

    def test_acknowledgement_is_low_priority(self, db):
        email = make_email(db)
        out = apply(db, email, verdict(company="Acme", event_type="acknowledgement"))
        assert out.notifications[0].row.priority == "low"

    def test_a_deadline_escalates_priority(self, db):
        email = make_email(db)
        out = apply(
            db,
            email,
            verdict(company="Acme", event_type="follow_up", deadline="2026-04-20"),
        )
        assert out.notifications[0].row.priority == "high"

    def test_title_includes_company_and_role(self, db):
        email = make_email(db)
        out = apply(
            db,
            email,
            verdict(company="Acme", role_title="Backend Engineer", event_type="rejection"),
        )
        title = out.notifications[0].row.title
        assert "Acme" in title
        assert "Backend Engineer" in title


class TestOnlyOwnApplications:
    """The board holds applications the user made — not adverts for jobs.

    Before this gate, any job-related email opened an application, named after
    the sender's display name when no company was extracted. A real mailbox
    produced a board of "LinkedIn", "Coursera", "Smriti Jain" and "Handshake AI
    Team": 37 of 40 entries were digests and connection invites.
    """

    def _verdict(self, *, source="llm", applied=True, **kwargs) -> Verdict:
        extraction = Extraction(
            is_job_related=True, confidence=0.95, recipient_applied=applied, **kwargs
        )
        return Verdict(
            is_job_related=True,
            confidence=0.95,
            source=source,
            raw=extraction.model_dump(),
            extraction=extraction,
        )

    def test_an_application_the_user_made_is_tracked(self, db):
        email = make_email(db, subject="Thank you for applying to Acme")
        outcome = apply(db, email, self._verdict(company="Acme", event_type="acknowledgement"))
        assert outcome is not None
        assert outcome.application.company == "Acme"

    def test_a_job_advert_is_not_tracked(self, db):
        """"Acme is hiring" names a company, but the user never applied."""
        email = make_email(db, subject="Acme is hiring for a Remote role")
        outcome = apply(
            db, email, self._verdict(company="Acme", applied=False, event_type="other")
        )
        assert outcome is None
        assert db.query(Application).count() == 0

    def test_cold_recruiter_outreach_is_not_tracked(self, db):
        email = make_email(db, subject="Exciting opportunity at Acme")
        outcome = apply(
            db,
            email,
            self._verdict(company="Acme", applied=False, event_type="recruiter_outreach"),
        )
        assert outcome is None

    def test_a_heuristic_guess_never_opens_an_application(self, db):
        """No extraction means no idea who applied to what. This is the exact
        path that filled the board with sender display names."""
        email = make_email(db, subject="Devyansh, your posts got 120 impressions")
        heuristic = Verdict(is_job_related=True, confidence=0.3, source="heuristic")
        assert apply(db, email, heuristic) is None
        assert db.query(Application).count() == 0

    def test_an_unnamed_company_never_opens_an_application(self, db):
        email = make_email(db, subject="Update on your application")
        outcome = apply(db, email, self._verdict(company="", event_type="acknowledgement"))
        assert outcome is None

    def test_an_interview_is_tracked_even_if_the_model_forgets_the_flag(self, db):
        """Nobody schedules an interview with someone who never applied, so the
        event type is trusted over a wrongly-false `recipient_applied`."""
        email = make_email(db, subject="Interview scheduled with Acme")
        outcome = apply(
            db,
            email,
            self._verdict(company="Acme", applied=False, event_type="interview_scheduled"),
        )
        assert outcome is not None
        assert outcome.application.status == ApplicationStatus.INTERVIEWING

    def test_a_reply_on_a_tracked_thread_still_counts(self, db):
        """Follow-ups often look like nothing in isolation; thread lineage is
        unambiguous, so it overrides the gate."""
        first = make_email(db, gmail_id="a", thread_id="t9", subject="Thanks for applying")
        apply(db, first, self._verdict(company="Acme", event_type="acknowledgement"))

        reply = make_email(db, gmail_id="b", thread_id="t9", subject="Re: Thanks for applying")
        outcome = apply(db, reply, self._verdict(company="", applied=False, event_type="other"))
        assert outcome is not None
        assert outcome.created_application is False
        assert db.query(Application).count() == 1
