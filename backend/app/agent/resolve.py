"""Match an email to an application, and advance that application's state.

The status field is a *derived* value guarded by an explicit state machine.
Without one, a "thanks for applying" auto-reply arriving late would drag an
application that already reached `offer` back to `applied`.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.classify import Extraction, Verdict, event_type_of
from app.models import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
    Email,
    EventType,
    Notification,
    NotificationPriority,
    utcnow,
)

log = logging.getLogger(__name__)

FUZZY_COMPANY_THRESHOLD = 0.88

# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------

# Progression rank. Higher wins; equal or lower is recorded but doesn't move
# the application backwards.
_RANK: dict[ApplicationStatus, int] = {
    ApplicationStatus.DISCOVERED: 0,
    ApplicationStatus.APPLIED: 1,
    ApplicationStatus.ACKNOWLEDGED: 2,
    ApplicationStatus.SCREENING: 3,
    ApplicationStatus.ASSESSMENT: 4,
    ApplicationStatus.INTERVIEWING: 5,
    ApplicationStatus.OFFER: 6,
}

# Declared outcomes. Reached from anywhere, and (except for ghosted) sticky.
_TERMINAL = {
    ApplicationStatus.ACCEPTED,
    ApplicationStatus.REJECTED,
    ApplicationStatus.WITHDRAWN,
}

_EVENT_TO_STATUS: dict[EventType, ApplicationStatus | None] = {
    EventType.APPLIED: ApplicationStatus.APPLIED,
    EventType.ACKNOWLEDGEMENT: ApplicationStatus.ACKNOWLEDGED,
    EventType.RECRUITER_OUTREACH: ApplicationStatus.DISCOVERED,
    EventType.SCREEN_SCHEDULED: ApplicationStatus.SCREENING,
    EventType.ASSESSMENT_SENT: ApplicationStatus.ASSESSMENT,
    EventType.INTERVIEW_SCHEDULED: ApplicationStatus.INTERVIEWING,
    EventType.OFFER: ApplicationStatus.OFFER,
    EventType.REJECTION: ApplicationStatus.REJECTED,
    EventType.WITHDRAWN: ApplicationStatus.WITHDRAWN,
    EventType.FOLLOW_UP: None,
    EventType.OTHER: None,
}


def next_status(current: ApplicationStatus, event: EventType) -> ApplicationStatus:
    """Return the status after `event`, honouring the transition rules."""
    proposed = _EVENT_TO_STATUS.get(event)
    if proposed is None or proposed == current:
        return current

    # Declared outcomes are sticky. `ghosted` is inferred, so it can be undone.
    if current in _TERMINAL:
        return current

    if proposed in _TERMINAL:
        return proposed

    if current == ApplicationStatus.GHOSTED:
        return proposed

    return proposed if _RANK.get(proposed, -1) > _RANK.get(current, -1) else current


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|inc\.|llc|ltd|ltd\.|limited|pvt|private|corp|corporation|co|company|"
    r"gmbh|plc|llp|technologies|technology|labs|solutions|systems|software|"
    r"services|group|holdings|india|global)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")

_SENIORITY = re.compile(
    r"\b(senior|sr|junior|jr|lead|principal|staff|associate|intern|trainee|"
    r"i{1,3}|1|2|3|entry[- ]level|graduate|new[- ]grad)\b",
    re.IGNORECASE,
)


def normalize_company(name: str | None) -> str:
    if not name:
        return ""
    text = _NON_ALNUM.sub(" ", name.lower())
    text = _LEGAL_SUFFIXES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalize_role(title: str | None) -> str:
    if not title:
        return ""
    text = _NON_ALNUM.sub(" ", title.lower())
    text = _SENIORITY.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def find_application(
    session: Session, email: Email, extraction: Extraction | None
) -> Application | None:
    """Find the application this email belongs to, if any.

    Order matters: thread lineage is the strongest signal because a reply chain
    is unambiguously about one application, regardless of how the model named
    the company.
    """
    # 1. Same Gmail thread as an email we already filed.
    if email.thread_id:
        row = session.execute(
            select(Application)
            .join(ApplicationEvent, ApplicationEvent.application_id == Application.id)
            .join(Email, Email.id == ApplicationEvent.email_id)
            .where(Email.thread_id == email.thread_id, Email.id != email.id)
            .order_by(ApplicationEvent.occurred_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is not None:
            return row

    company_norm = normalize_company(extraction.company if extraction else None)
    if not company_norm:
        return None

    role_norm = normalize_role(extraction.role_title if extraction else None)

    # 2. Exact company + role.
    if role_norm:
        row = session.scalar(
            select(Application).where(
                Application.company_normalized == company_norm,
                Application.role_normalized == role_norm,
            )
        )
        if row is not None:
            return row

    candidates = session.scalars(
        select(Application).where(Application.company_normalized == company_norm)
    ).all()

    # 3. Same company, and either we have no role to compare or it's close.
    if candidates:
        if not role_norm:
            return max(candidates, key=lambda a: a.last_activity_at)
        best = max(candidates, key=lambda a: _similar(role_norm, a.role_normalized or ""))
        if _similar(role_norm, best.role_normalized or "") >= 0.7 or not best.role_normalized:
            return best
        return None

    # 4. Fuzzy company match, guarded by role agreement when both are known.
    for app in session.scalars(select(Application)).all():
        if _similar(company_norm, app.company_normalized) < FUZZY_COMPANY_THRESHOLD:
            continue
        if role_norm and app.role_normalized and _similar(role_norm, app.role_normalized) < 0.6:
            continue
        return app

    return None


# --------------------------------------------------------------------------
# Applying an email to an application
# --------------------------------------------------------------------------


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, f"{text}T00:00:00+00:00"):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


@dataclass
class NotificationDraft:
    row: Notification

    def payload(self) -> dict:
        return {
            "id": self.row.id,
            "kind": self.row.kind,
            "priority": self.row.priority.value
            if isinstance(self.row.priority, NotificationPriority)
            else str(self.row.priority),
            "title": self.row.title,
            "body": self.row.body,
            "application_id": self.row.application_id,
            "email_id": self.row.email_id,
            "created_at": self.row.created_at.isoformat() if self.row.created_at else None,
        }


@dataclass
class Outcome:
    application: Application
    event: ApplicationEvent
    created_application: bool
    status_before: ApplicationStatus
    status_after: ApplicationStatus
    notifications: list[NotificationDraft] = field(default_factory=list)

    def application_payload(self) -> dict:
        app = self.application
        return {
            "id": app.id,
            "company": app.company,
            "role_title": app.role_title,
            "status": app.status.value if isinstance(app.status, ApplicationStatus) else app.status,
            "status_before": self.status_before.value,
            "created": self.created_application,
            "last_activity_at": app.last_activity_at.isoformat()
            if app.last_activity_at
            else None,
            "next_action": app.next_action,
            "next_action_due": app.next_action_due.isoformat() if app.next_action_due else None,
        }


def apply(
    session: Session, email: Email, verdict: Verdict, *, reclassify: bool = False
) -> Outcome:
    extraction = verdict.extraction
    event_type = event_type_of(extraction)

    app = find_application(session, email, extraction)
    created = False

    if app is None:
        company = (extraction.company if extraction else "") or _fallback_company(email)
        role = (extraction.role_title if extraction else "") or None
        app = Application(
            company=company,
            company_normalized=normalize_company(company),
            role_title=role,
            role_normalized=normalize_role(role),
            status=ApplicationStatus.DISCOVERED,
            first_seen_at=email.received_at or utcnow(),
            last_activity_at=email.received_at or utcnow(),
        )
        session.add(app)
        session.flush()
        created = True

    status_before = _as_status(app.status)
    status_after = next_status(status_before, event_type)

    if status_after != status_before:
        app.status = status_after
    elif _EVENT_TO_STATUS.get(event_type) not in (None, status_before):
        log.info(
            "Application %s: event %s would move %s -> %s; blocked by state machine",
            app.id,
            event_type.value,
            status_before.value,
            _EVENT_TO_STATUS.get(event_type),
        )

    _merge_details(app, extraction, email)

    occurred = email.received_at or utcnow()
    if app.last_activity_at is None or occurred > app.last_activity_at:
        app.last_activity_at = occurred

    event = _upsert_event(
        session,
        app,
        email,
        event_type,
        verdict,
        occurred,
        status_before,
        status_after,
        reclassify=reclassify,
    )

    outcome = Outcome(
        application=app,
        event=event,
        created_application=created,
        status_before=status_before,
        status_after=status_after,
    )

    if not reclassify:
        outcome.notifications = _make_notifications(session, app, email, event_type, extraction)

    return outcome


def retract(session: Session, email: Email) -> list[int]:
    """Undo what a previous, wrong verdict recorded for this email.

    Re-classification can flip an email from job-related to not. Without this,
    the timeline entry and the application built from the bad verdict survive
    the correction, so a fixed classifier can never clean up after a broken one.

    Returns the ids of applications deleted for having no events left.
    """
    events = list(
        session.scalars(select(ApplicationEvent).where(ApplicationEvent.email_id == email.id)).all()
    )
    if not events:
        return []

    touched = {e.application_id for e in events}
    for event in events:
        session.delete(event)

    # Notifications reference the email; a retracted event should not leave an
    # alert about an interview that was never real.
    stale_notes = session.scalars(
        select(Notification).where(Notification.email_id == email.id)
    ).all()
    for note in stale_notes:
        session.delete(note)

    session.flush()

    removed: list[int] = []
    for application_id in touched:
        remaining = session.scalar(
            select(ApplicationEvent).where(ApplicationEvent.application_id == application_id)
        )
        if remaining is not None:
            continue
        # The application existed only because of this email.
        app = session.get(Application, application_id)
        if app is not None:
            session.delete(app)
            removed.append(application_id)

    if removed:
        log.info("Retracted email %s; removed %d empty application(s)", email.id, len(removed))
    session.flush()
    return removed


def _as_status(value) -> ApplicationStatus:
    if isinstance(value, ApplicationStatus):
        return value
    try:
        return ApplicationStatus(value)
    except (ValueError, TypeError):
        return ApplicationStatus.DISCOVERED


def _fallback_company(email: Email) -> str:
    """Last resort when the model couldn't name a company."""
    if email.from_name:
        return email.from_name
    if email.from_addr and "@" in email.from_addr:
        return email.from_addr.rsplit("@", 1)[1]
    return "Unknown"


def _merge_details(app: Application, extraction: Extraction | None, email: Email) -> None:
    """Fill blanks without overwriting what we already know."""
    if extraction is None:
        return

    if not app.role_title and extraction.role_title:
        app.role_title = extraction.role_title
        app.role_normalized = normalize_role(extraction.role_title)

    for attr, value in (
        ("location", extraction.location),
        ("source", extraction.source),
        ("job_url", extraction.job_url),
        ("salary_text", extraction.salary_text),
        ("contact_name", extraction.contact_name),
    ):
        if value and not getattr(app, attr):
            setattr(app, attr, value)

    contact_email = extraction.contact_email or email.from_addr
    if contact_email and not app.contact_email:
        app.contact_email = contact_email

    # The newest email wins on what to do next — that is the whole point of it.
    if extraction.next_action:
        app.next_action = extraction.next_action
        app.next_action_due = parse_iso(extraction.deadline) or parse_iso(
            extraction.event_datetime
        )
    elif extraction.event_datetime:
        app.next_action_due = parse_iso(extraction.event_datetime) or app.next_action_due


def _upsert_event(
    session: Session,
    app: Application,
    email: Email,
    event_type: EventType,
    verdict: Verdict,
    occurred: datetime,
    status_before: ApplicationStatus,
    status_after: ApplicationStatus,
    *,
    reclassify: bool,
) -> ApplicationEvent:
    existing = session.scalar(
        select(ApplicationEvent).where(
            ApplicationEvent.application_id == app.id,
            ApplicationEvent.email_id == email.id,
        )
    )

    summary = (verdict.extraction.summary if verdict.extraction else None) or email.subject

    if existing is not None:
        if reclassify:
            existing.event_type = event_type
            existing.summary = summary
            existing.extracted = verdict.raw
            existing.confidence = verdict.confidence
            existing.status_before = status_before.value
            existing.status_after = status_after.value
        return existing

    event = ApplicationEvent(
        application_id=app.id,
        email_id=email.id,
        event_type=event_type,
        occurred_at=occurred,
        summary=summary,
        extracted=verdict.raw,
        confidence=verdict.confidence,
        status_before=status_before.value,
        status_after=status_after.value,
    )
    session.add(event)
    session.flush()
    return event


_HIGH_PRIORITY_EVENTS = {
    EventType.OFFER,
    EventType.INTERVIEW_SCHEDULED,
    EventType.SCREEN_SCHEDULED,
    EventType.ASSESSMENT_SENT,
}

_EVENT_TITLES: dict[EventType, str] = {
    EventType.APPLIED: "Application sent to {company}",
    EventType.ACKNOWLEDGEMENT: "{company} acknowledged your application",
    EventType.RECRUITER_OUTREACH: "Recruiter outreach from {company}",
    EventType.SCREEN_SCHEDULED: "Screening call scheduled with {company}",
    EventType.INTERVIEW_SCHEDULED: "Interview scheduled with {company}",
    EventType.ASSESSMENT_SENT: "{company} sent an assessment",
    EventType.OFFER: "Offer from {company}",
    EventType.REJECTION: "{company} passed on your application",
    EventType.FOLLOW_UP: "Follow-up from {company}",
    EventType.WITHDRAWN: "Application to {company} withdrawn",
    EventType.OTHER: "Update from {company}",
}


def _make_notifications(
    session: Session,
    app: Application,
    email: Email,
    event_type: EventType,
    extraction: Extraction | None,
) -> list[NotificationDraft]:
    title = _EVENT_TITLES.get(event_type, "Update from {company}").format(company=app.company)
    if app.role_title:
        title = f"{title} — {app.role_title}"

    if event_type in _HIGH_PRIORITY_EVENTS:
        priority = NotificationPriority.HIGH
    elif event_type in (EventType.APPLIED, EventType.ACKNOWLEDGEMENT):
        priority = NotificationPriority.LOW
    else:
        priority = NotificationPriority.NORMAL

    # Anything with a hard deadline is worth surfacing loudly.
    if extraction and (extraction.deadline or extraction.event_datetime):
        priority = NotificationPriority.HIGH

    body = (extraction.summary if extraction else None) or email.snippet
    if extraction and extraction.next_action:
        body = f"{body}\n\nNext: {extraction.next_action}" if body else extraction.next_action

    row = Notification(
        kind=event_type.value,
        priority=priority,
        title=title,
        body=body,
        application_id=app.id,
        email_id=email.id,
    )
    session.add(row)
    session.flush()
    return [NotificationDraft(row=row)]
