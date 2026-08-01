"""Cheap heuristic gate in front of the classifier.

This is a **cost** lever, not a correctness one. It only rejects mail it is
confident about, and is deliberately biased toward false positives — letting a
newsletter through costs one Flash call, while dropping a real interview invite
costs the user an interview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Email


@dataclass(slots=True)
class PrefilterResult:
    verdict: str  # "reject" | "pass"
    reason: str


# Senders that are never job mail, however they phrase the subject line.
_NEVER_DOMAINS = {
    "noreply@github.com",
    "notifications@github.com",
    "no-reply@accounts.google.com",
    "noreply@medium.com",
    "info@meetup.com",
}

_NEVER_DOMAIN_SUFFIXES = (
    "facebookmail.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "reddit.com",
    "youtube.com",
    "netflix.com",
    "spotify.com",
    "uber.com",
    "swiggy.in",
    "zomato.com",
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "paytm.com",
    "phonepe.com",
)

# Any of these in the subject or sender means "let the LLM decide", even if the
# message otherwise looks like bulk mail.
_JOB_SIGNALS = re.compile(
    r"""
    \b(
        applicat(?:ion|ions)|applied|apply|
        interview|screening|screen\s+call|
        recruit(?:er|ing|ment)|talent\s+acquisition|hiring|
        candidat(?:e|ure)|
        job|role|position|vacancy|opening|opportunit(?:y|ies)|
        offer\s+letter|offer|
        assessment|coding\s+(?:test|challenge|round)|hackerrank|codility|hackerearth|
        take[-\s]?home|
        onsite|on-?site|
        shortlist(?:ed)?|
        resume|cv|cover\s+letter|
        careers?|
        no\s+longer\s+(?:be\s+)?(?:moving|considering)|
        not\s+(?:be\s+)?(?:moving\s+forward|selected)|
        regret\s+to\s+inform|
        we(?:'| a)re\s+moving\s+forward
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_JOB_SENDER_HINTS = (
    "greenhouse",
    "lever.co",
    "myworkday",
    "workday",
    "smartrecruiters",
    "ashbyhq",
    "jobvite",
    "icims",
    "taleo",
    "successfactors",
    "bamboohr",
    "recruitee",
    "workable",
    "breezy",
    "linkedin",
    "naukri",
    "indeed",
    "glassdoor",
    "wellfound",
    "angellist",
    "hired.com",
    "instahyre",
    "cutshort",
    "hirist",
    "internshala",
    "unstop",
    "recruit",
    "talent",
    "careers",
    "hiring",
    "hr@",
    "jobs@",
    "no-reply@hire",
)

# Bulk-mail markers. Only used to reject when there is *no* job signal at all.
_BULK_HEADERS = ("list-unsubscribe", "list-id")
_BULK_PRECEDENCE = {"bulk", "list", "junk"}

_PROMO_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS", "SPAM", "TRASH"}


def _text_for_signals(email: Email) -> str:
    parts = [email.subject or "", email.from_name or "", email.from_addr or "", email.snippet or ""]
    return " ".join(parts)


def has_job_signal(email: Email) -> bool:
    if _JOB_SIGNALS.search(_text_for_signals(email)):
        return True
    sender = (email.from_addr or "").lower()
    display = (email.from_name or "").lower()
    return any(hint in sender or hint in display for hint in _JOB_SENDER_HINTS)


def prefilter(email: Email) -> PrefilterResult:
    sender = (email.from_addr or "").lower()
    labels = set(email.labels or [])
    headers = {k.lower(): v for k, v in (email.raw_headers or {}).items()}

    signal = has_job_signal(email)

    # Hard sender blocks still lose to an explicit job signal — recruiters do
    # reach out through LinkedIn and similar platforms.
    if not signal:
        if sender in _NEVER_DOMAINS:
            return PrefilterResult("reject", f"sender {sender} never sends job mail")

        domain = sender.rsplit("@", 1)[-1] if "@" in sender else ""
        if any(domain == d or domain.endswith("." + d) for d in _NEVER_DOMAIN_SUFFIXES):
            return PrefilterResult("reject", f"domain {domain} never sends job mail")

        if labels & _PROMO_LABELS:
            return PrefilterResult("reject", "promotional/social category with no job signal")

        if any(h in headers for h in _BULK_HEADERS):
            return PrefilterResult("reject", "bulk mail headers with no job signal")

        precedence = (headers.get("precedence") or "").strip().lower()
        if precedence in _BULK_PRECEDENCE:
            return PrefilterResult("reject", f"precedence: {precedence} with no job signal")

        if headers.get("auto-submitted", "").lower().startswith("auto-generated"):
            return PrefilterResult("reject", "auto-generated with no job signal")

    if not (email.subject or email.body_text or email.snippet):
        return PrefilterResult("reject", "empty message")

    return PrefilterResult("pass", "job signal present" if signal else "ambiguous, defer to LLM")
