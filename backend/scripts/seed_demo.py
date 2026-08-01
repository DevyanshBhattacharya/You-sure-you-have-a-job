"""Populate the database with fabricated sample data.

Lets you exercise the dashboard before Gmail or a Gemini key is wired up.
Classification is supplied directly rather than inferred, so the board looks
realistic without spending any API calls.

    python scripts/seed_demo.py          # add the sample set
    python scripts/seed_demo.py --clear  # remove it again

Everything it creates is marked with a `demo-` gmail_id prefix, so `--clear`
removes exactly what it added and leaves real mail untouched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.agent.classify import Extraction, Verdict  # noqa: E402
from app.agent.resolve import apply as apply_verdict  # noqa: E402
from app.db import init_db, session_scope  # noqa: E402
from app.kb import indexer  # noqa: E402
from app.models import Application, ApplicationEvent, Email  # noqa: E402

PREFIX = "demo-"

# (days_ago, subject, sender, body, extraction fields)
SAMPLES: list[tuple[int, str, str, str, dict]] = [
    (
        28,
        "Thanks for applying to Northwind Labs",
        "careers@northwindlabs.com",
        "Hi,\n\nThanks for applying to the Backend Engineer role at Northwind Labs. "
        "Our team is reviewing applications and will be in touch.\n\n— Northwind Talent",
        {
            "event_type": "acknowledgement",
            "company": "Northwind Labs",
            "role_title": "Backend Engineer",
            "location": "Remote",
            "source": "Greenhouse",
            "summary": "Northwind Labs acknowledged the Backend Engineer application.",
        },
    ),
    (
        21,
        "Northwind Labs — technical screen",
        "priya@northwindlabs.com",
        "Hi,\n\nWe would like to set up a 45-minute technical screen. Are you free "
        "Thursday at 3pm IST?\n\nBest,\nPriya",
        {
            "event_type": "screen_scheduled",
            "company": "Northwind Labs",
            "role_title": "Backend Engineer",
            "contact_name": "Priya",
            "next_action": "Confirm availability for the technical screen",
            "summary": "Technical screen proposed for Thursday 3pm IST.",
        },
    ),
    (
        9,
        "Northwind Labs — onsite loop",
        "priya@northwindlabs.com",
        "Great feedback from the screen. We'd like to move to the final loop: "
        "four interviews across one day.",
        {
            "event_type": "interview_scheduled",
            "company": "Northwind Labs",
            "role_title": "Backend Engineer",
            "contact_name": "Priya",
            "event_datetime": (datetime.now(UTC) + timedelta(days=4)).strftime("%Y-%m-%d"),
            "next_action": "Prepare for the four-stage onsite loop",
            "summary": "Onsite interview loop scheduled.",
        },
    ),
    (
        18,
        "Your application to Kestrel Analytics",
        "no-reply@greenhouse.io",
        "Kestrel Analytics has received your application for Data Engineer.",
        {
            "event_type": "applied",
            "company": "Kestrel Analytics",
            "role_title": "Data Engineer",
            "source": "Greenhouse",
            "summary": "Applied to Kestrel Analytics for Data Engineer.",
        },
    ),
    (
        14,
        "Kestrel Analytics — coding assessment",
        "assessments@kestrelanalytics.com",
        "Please complete the take-home assessment within 7 days. "
        "The link expires after that.",
        {
            "event_type": "assessment_sent",
            "company": "Kestrel Analytics",
            "role_title": "Data Engineer",
            "next_action": "Complete the take-home assessment",
            "deadline": (datetime.now(UTC) + timedelta(days=3)).strftime("%Y-%m-%d"),
            "summary": "Take-home assessment sent, due in 7 days.",
        },
    ),
    (
        30,
        "Update on your application — Vertex Systems",
        "hr@vertexsystems.com",
        "After careful consideration we have decided not to move forward with your "
        "application for the Platform Engineer role. We wish you the best.",
        {
            "event_type": "rejection",
            "company": "Vertex Systems",
            "role_title": "Platform Engineer",
            "summary": "Vertex Systems declined the Platform Engineer application.",
        },
    ),
    (
        6,
        "Offer — Meridian Software",
        "people@meridiansoftware.com",
        "We're delighted to offer you the Senior Software Engineer position. "
        "The written offer is attached; please respond within 10 days.",
        {
            "event_type": "offer",
            "company": "Meridian Software",
            "role_title": "Senior Software Engineer",
            "salary_text": "Competitive",
            "next_action": "Review and respond to the offer",
            "deadline": (datetime.now(UTC) + timedelta(days=6)).strftime("%Y-%m-%d"),
            "summary": "Offer received for Senior Software Engineer.",
        },
    ),
    (
        3,
        "Opportunity at Harbour Digital",
        "recruiting@harbourdigital.com",
        "I came across your profile and thought you'd be a strong fit for our "
        "Full Stack Engineer opening. Would you be open to a chat?",
        {
            "event_type": "recruiter_outreach",
            "company": "Harbour Digital",
            "role_title": "Full Stack Engineer",
            "contact_email": "recruiting@harbourdigital.com",
            "next_action": "Reply if interested in an intro call",
            "summary": "Recruiter outreach about a Full Stack Engineer role.",
        },
    ),
    (
        40,
        "Thanks for your interest in Solstice Cloud",
        "careers@solsticecloud.com",
        "Thanks for applying to the Site Reliability Engineer role. We will review "
        "your application shortly.",
        {
            "event_type": "acknowledgement",
            "company": "Solstice Cloud",
            "role_title": "Site Reliability Engineer",
            "summary": "Solstice Cloud acknowledged the SRE application.",
        },
    ),
]


def build_verdict(fields: dict) -> Verdict:
    extraction = Extraction(is_job_related=True, confidence=0.94, **fields)
    return Verdict(
        is_job_related=True,
        confidence=0.94,
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


def clear(session) -> tuple[int, int]:
    """Remove demo emails *and* the applications they created.

    Deleting an Email cascades only to its ApplicationEvents — the Application
    itself survives, so a naive clear leaves orphaned rows sitting on the board
    forever. Collect the affected applications first, then drop any left with
    no events.
    """
    rows = session.scalars(select(Email).where(Email.gmail_id.like(f"{PREFIX}%"))).all()
    touched = {
        event.application_id for email in rows for event in email.events if event.application_id
    }

    for row in rows:
        session.delete(row)
    session.flush()

    removed_apps = 0
    for application_id in touched:
        application = session.get(Application, application_id)
        if application is None:
            continue
        remaining = session.scalar(
            select(func.count())
            .select_from(ApplicationEvent)
            .where(ApplicationEvent.application_id == application_id)
        )
        if not remaining:
            session.delete(application)
            removed_apps += 1

    return len(rows), removed_apps


def seed(session) -> int:
    created = 0
    for index, (days_ago, subject, sender, body, fields) in enumerate(SAMPLES):
        gmail_id = f"{PREFIX}{index:03d}"
        if session.scalar(select(Email).where(Email.gmail_id == gmail_id)):
            continue

        received = datetime.now(UTC) - timedelta(days=days_ago)
        email = Email(
            gmail_id=gmail_id,
            thread_id=f"{PREFIX}thread-{fields['company'].lower().replace(' ', '-')}",
            from_addr=sender,
            from_name=fields["company"],
            to_addr="me@example.com",
            subject=subject,
            snippet=body[:140],
            body_text=body,
            received_at=received,
            labels=["INBOX"],
            raw_headers={"from": sender, "subject": subject},
            is_job_related=True,
            classification_confidence=0.94,
            classification_source="llm",
            processed_at=datetime.now(UTC),
        )
        session.add(email)
        session.flush()

        verdict = build_verdict(fields)
        email.classification_raw = verdict.raw

        outcome = apply_verdict(session, email, verdict)
        session.flush()
        indexer.index_email(session, email, application_id=outcome.application.id)
        created += 1

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear", action="store_true", help="Remove demo data and exit")
    args = parser.parse_args()

    init_db()
    with session_scope() as session:
        if args.clear:
            emails, applications = clear(session)
            print(f"Removed {emails} demo email(s) and {applications} application(s).")
            return 0

        created = seed(session)
        print(f"Seeded {created} demo email(s).")
        if created == 0:
            print("(Demo data was already present - run with --clear first to reset.)")

    print("Open the dashboard to see the populated board.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
