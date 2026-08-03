"""The heuristic gate in front of the classifier.

Its only job is to reject mail that is *obviously* not job related. The cost of
a false negative (dropping a real interview invite) is far higher than a false
positive (one wasted Flash call), so the bar for rejecting is deliberately high.
"""

from __future__ import annotations

import pytest

from app.agent.prefilter import prefilter
from tests.fixtures import make_email

BULK = {"list-unsubscribe": "<mailto:unsub@x.com>"}


@pytest.mark.parametrize(
    ("subject", "sender", "headers", "labels"),
    [
        ("Your Amazon order has shipped", "ship-confirm@amazon.in", BULK, ["CATEGORY_PROMOTIONS"]),
        ("50% off this weekend", "deals@swiggy.in", BULK, ["CATEGORY_PROMOTIONS"]),
        ("Someone mentioned you", "notify@facebookmail.com", BULK, ["CATEGORY_SOCIAL"]),
        ("Your weekly digest", "digest@medium.com", BULK, ["CATEGORY_PROMOTIONS"]),
        ("New login to your account", "no-reply@accounts.google.com", {}, ["INBOX"]),
    ],
)
def test_rejects_obvious_non_job_mail(db, subject, sender, headers, labels):
    email = make_email(db, subject=subject, sender=sender, headers=headers, labels=labels)
    assert prefilter(email).verdict == "reject"


@pytest.mark.parametrize(
    ("subject", "sender"),
    [
        ("Your application to Acme Corp", "careers@acme.com"),
        ("Interview invitation — Backend Engineer", "jane@startup.io"),
        ("Update on your candidature", "no-reply@greenhouse.io"),
        ("Coding assessment for Data Analyst role", "noreply@hackerrank.com"),
        ("We regret to inform you", "hr@bigco.com"),
        ("Next steps in the hiring process", "talent@example.com"),
        ("Your offer letter", "people@example.com"),
    ],
)
def test_passes_clear_job_mail(db, subject, sender):
    email = make_email(db, subject=subject, sender=sender)
    assert prefilter(email).verdict == "pass"


def test_job_signal_survives_bulk_headers(db):
    """ATS platforms send through bulk infrastructure. Rejecting on headers
    alone would silently drop most real application mail."""
    email = make_email(
        db,
        subject="Your application to Acme — Software Engineer",
        sender="no-reply@greenhouse.io",
        headers=BULK,
        labels=["CATEGORY_UPDATES"],
    )
    assert prefilter(email).verdict == "pass"


def test_job_signal_survives_promotions_label(db):
    email = make_email(
        db,
        subject="Interview scheduled with Acme",
        sender="recruiting@acme.com",
        labels=["CATEGORY_PROMOTIONS"],
        headers=BULK,
    )
    assert prefilter(email).verdict == "pass"


class TestJobBoardNoise:
    """Job boards send far more noise than application updates.

    The original rule treated "linkedin" in the sender as a job signal, which
    exempted every message from that domain from the bulk-mail checks below —
    so connection invites, newsletters and "X is hiring" alerts all reached the
    classifier, and each one could open an application. Being a job board is
    now worth nothing on its own; only mail naming the recipient's own
    application gets through.
    """

    @pytest.mark.parametrize(
        ("subject", "sender"),
        [
            ("Abdullah just messaged you", "messaging-digest-noreply@linkedin.com"),
            ("Sunishka accepted your invitation, explore her network", "invitations@linkedin.com"),
            ("I want to connect", "invitations@linkedin.com"),
            ("Devyansh, your posts got 120 impressions last week", "notifications@linkedin.com"),
            ("Torinit is hiring for a Remote role", "jobs-noreply@linkedin.com"),
            ("Devyansh, apply now to 'AI Engineer'", "jobs-noreply@linkedin.com"),
            ("Your job's expiring on Aug 7: Associate - AI Engineer", "jobs-noreply@linkedin.com"),
            ("Trending internships based on your profile!", "noreply@unstop.news"),
            ("Adobe is hiring interns at an INR 1.1 lac stipend", "noreply@unstop.news"),
            ("How to Improve Your Resume Score to 90+", "recommendationnc@naukri.com"),
            ("Don't miss 3 months of Google AI Pro", "coursera@m.learn.coursera.org"),
            ("You're invited to a webinar on Outlier", "no-reply@outlier.ai"),
        ],
    )
    def test_alerts_and_social_noise_are_rejected(self, db, subject, sender):
        email = make_email(
            db, subject=subject, sender=sender, headers=BULK, labels=["CATEGORY_PROMOTIONS"]
        )
        assert prefilter(email).verdict == "reject"

    def test_a_board_forwarding_a_real_application_update_still_passes(self, db):
        """LinkedIn does relay genuine confirmations; those must survive."""
        email = make_email(
            db,
            subject="Devyansh, your application was sent to Triomics",
            sender="jobs-noreply@linkedin.com",
            headers=BULK,
            labels=["CATEGORY_UPDATES"],
        )
        assert prefilter(email).verdict == "pass"

    def test_an_employer_is_not_treated_as_a_board(self, db):
        """`careers@` at a real employer keeps its exemption from bulk rules."""
        email = make_email(
            db,
            subject="Please confirm your identity",
            sender="careers@recruitment.americanexpress.com",
            headers=BULK,
            labels=["CATEGORY_PROMOTIONS"],
        )
        assert prefilter(email).verdict == "pass"


def test_automated_transactional_mail_is_rejected(db):
    """Receipts and boarding passes have no bulk headers, but nobody recruits
    from `no-reply@`."""
    email = make_email(
        db, subject="Boarding gate announced for flight AI 427", sender="noreply@airindia.com"
    )
    assert prefilter(email).verdict == "reject"


def test_a_person_writing_directly_is_always_deferred(db):
    """The expensive miss. A recruiter mailing from their own address may use
    no keyword at all, so personal mail goes to the LLM regardless."""
    email = make_email(db, subject="Are you free to chat Thursday?", sender="jane@startup.io")
    assert prefilter(email).verdict == "pass"


def test_ambiguous_mail_is_deferred_to_the_llm(db):
    email = make_email(db, subject="Quick question", sender="someone@unknown.com")
    result = prefilter(email)
    assert result.verdict == "pass"
    assert "ambiguous" in result.reason


def test_empty_message_is_rejected(db):
    email = make_email(db, subject="", body="")
    email.subject = None
    email.body_text = None
    email.snippet = None
    assert prefilter(email).verdict == "reject"
