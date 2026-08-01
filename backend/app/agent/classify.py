"""Job-relevance classification and structured field extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent import llm, prefilter
from app.config import get_settings
from app.models import Email, EventType

log = logging.getLogger(__name__)

MAX_PROMPT_BODY_CHARS = 6_000

EventTypeLiteral = Literal[
    "applied",
    "acknowledgement",
    "recruiter_outreach",
    "screen_scheduled",
    "interview_scheduled",
    "assessment_sent",
    "offer",
    "rejection",
    "follow_up",
    "withdrawn",
    "other",
]


class Extraction(BaseModel):
    """Schema the model is constrained to.

    Every optional field is a plain string defaulting to "" rather than a
    nullable type — it keeps the generated JSON schema simple and avoids
    null-vs-missing ambiguity in the response.
    """

    is_job_related: bool = Field(
        description="True only if this email concerns the recipient's own job search."
    )
    confidence: float = Field(description="Confidence in is_job_related, from 0.0 to 1.0.")
    event_type: EventTypeLiteral = "other"
    company: str = Field(default="", description="Hiring company. Not the job board.")
    role_title: str = Field(default="", description="Job title applied for.")
    location: str = ""
    source: str = Field(default="", description="Where it came from, e.g. LinkedIn, Greenhouse.")
    job_url: str = ""
    salary_text: str = ""
    contact_name: str = ""
    contact_email: str = ""
    event_datetime: str = Field(
        default="", description="ISO-8601 datetime of a scheduled interview or call, else ''."
    )
    deadline: str = Field(
        default="", description="ISO-8601 date of any deadline the recipient must meet, else ''."
    )
    next_action: str = Field(
        default="", description="What the recipient must do next. Empty if nothing is required."
    )
    summary: str = Field(default="", description="One sentence, factual, under 25 words.")


SYSTEM_INSTRUCTION = """\
You classify a single email from a job seeker's personal inbox and extract \
structured facts about it.

`is_job_related` is true ONLY when the email concerns THIS PERSON'S OWN job \
search or employment applications. That includes: application confirmations, \
recruiter outreach about a specific role, interview scheduling, assessments and \
coding tests, offers, and rejections.

It is FALSE for: job-board digests and "jobs you may like" newsletters, generic \
marketing from career platforms, course and certification promotions, \
networking-site notifications, and anything relating to someone else's hiring.

A single generic job alert listing many roles is NOT job related. A message \
about one specific application or one specific role IS.

Extraction rules:
- `company` is the hiring employer, never the applicant tracking system or job \
  board. "via Greenhouse" means the company is whoever Greenhouse is mailing for.
- Use "" for anything not stated. Never guess, and never infer a company from \
  the sender's domain when that domain is a job board.
- Dates must be ISO-8601. Resolve relative dates ("this Thursday at 3pm IST") \
  against the email's received timestamp given below.
- `next_action` is only for something the recipient must actively do. An \
  acknowledgement that says "we will be in touch" requires no action.
- Pick the single most specific `event_type`. A rejection is `rejection` even if \
  it also thanks the applicant for applying.
"""


@dataclass
class Verdict:
    is_job_related: bool
    confidence: float
    source: str  # prefilter | llm | heuristic
    raw: dict = field(default_factory=dict)
    extraction: Extraction | None = None


def build_prompt(email: Email) -> str:
    body = (email.body_text or email.snippet or "").strip()
    if len(body) > MAX_PROMPT_BODY_CHARS:
        body = body[:MAX_PROMPT_BODY_CHARS] + "\n[truncated]"

    received = email.received_at.isoformat() if email.received_at else "unknown"
    sender = email.from_addr or "unknown"
    if email.from_name:
        sender = f"{email.from_name} <{sender}>"

    return (
        f"From: {sender}\n"
        f"To: {email.to_addr or 'unknown'}\n"
        f"Received: {received}\n"
        f"Subject: {email.subject or '(no subject)'}\n"
        f"\n--- BODY ---\n{body or '(empty)'}\n--- END BODY ---"
    )


def classify(session: Session, email: Email) -> Verdict:
    """Classify one email, spending an LLM call only when it might be worth it."""
    gate = prefilter.prefilter(email)
    if gate.verdict == "reject":
        return Verdict(
            is_job_related=False,
            confidence=0.95,
            source="prefilter",
            raw={"reason": gate.reason},
        )

    if not llm.is_configured():
        # Degraded mode: keep the app usable without a Gemini key.
        signal = prefilter.has_job_signal(email)
        return Verdict(
            is_job_related=signal,
            confidence=0.4 if signal else 0.3,
            source="heuristic",
            raw={"reason": "GEMINI_API_KEY not set; heuristic fallback"},
        )

    settings = get_settings()
    try:
        result = llm.generate_json(
            prompt=build_prompt(email),
            schema=Extraction,
            system_instruction=SYSTEM_INSTRUCTION,
            model=settings.classifier_model,
        )
    except llm.DeferWorkError:
        # Quota exhausted, or the backend is unreachable (Ollama not running).
        # Deliberately NOT downgraded to a heuristic guess: that would mark the
        # email processed and permanently freeze a low-confidence verdict built
        # from the sender's name. Both conditions are systemic, so the fallback
        # would burn the whole backlog before anyone noticed. Propagating
        # leaves the email unprocessed, to be retried later.
        raise
    except Exception as exc:  # noqa: BLE001 - never lose an email to an API blip
        log.warning("Classification failed for email %s: %s", email.id, str(exc)[:200])
        signal = prefilter.has_job_signal(email)
        return Verdict(
            is_job_related=signal,
            confidence=0.3,
            source="heuristic",
            raw={"error": str(exc)},
        )

    extraction: Extraction = result.parsed
    return Verdict(
        is_job_related=extraction.is_job_related,
        confidence=max(0.0, min(1.0, extraction.confidence)),
        source="llm",
        raw=extraction.model_dump(),
        extraction=extraction,
    )


def event_type_of(extraction: Extraction | None) -> EventType:
    if extraction is None:
        return EventType.OTHER
    try:
        return EventType(extraction.event_type)
    except ValueError:
        return EventType.OTHER
