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


def test_recruiter_via_linkedin_is_not_rejected(db):
    email = make_email(
        db,
        subject="A recruiter sent you a message",
        sender="messages-noreply@linkedin.com",
        headers=BULK,
        labels=["CATEGORY_UPDATES"],
    )
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
