"""Cheap heuristic gate in front of the classifier.

Two jobs, in order of importance:

1. Keep mail that is not about **the recipient's own applications** away from
   the tracker. Job boards send far more alerts, digests and course ads than
   they do application updates, and every one of those that reaches the
   classifier is a chance to invent an application that never existed.
2. Save classifier calls. On a local model that is the difference between an
   import finishing in minutes and finishing overnight.

The rule that makes both work is the split between a **strong** signal and a
**weak** one. A strong signal — "your application", an interview, an
assessment, or mail from an employer's own recruiting address — overrides every
bulk-mail rule below, because an ATS acknowledgement is bulk mail by every
technical measure. A weak signal (the word "job" somewhere, or a sender with
"careers" in it) is not enough to override anything; it only decides whether an
otherwise-unremarkable message is worth a classifier call.

Getting that backwards is what filled the board with LinkedIn digests: "linkedin"
in the sender counted as a signal, which skipped the promotional-label and
`List-Unsubscribe` checks, so every connection invite and newsletter went to the
classifier and came back as a tracked application.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import Email


@dataclass(slots=True)
class PrefilterResult:
    verdict: str  # "reject" | "pass"
    reason: str


# --------------------------------------------------------------------------
# Strong signals — these override every rejection rule
# --------------------------------------------------------------------------

# Language that appears when a message is about the recipient's own application
# or an active hiring process, and rarely otherwise. Deliberately anchored:
# "your application" qualifies, a bare "job" or "opportunity" does not, because
# every job alert ever sent contains those.
_OWN_APPLICATION = re.compile(
    r"""
    \b(
        your(?:\s+\S+){0,3}\s+application\b |       # "your application", "your IBM application"
        application\s+(?:has\s+been\s+)?
            (?:received|submitted|status|update|confirmation|id|number|reference) |
        (?:received|reviewed)\s+your\s+application |
        thank\s+you\s+for\s+(?:applying|your\s+application|your\s+interest\s+in\s+the) |
        thanks\s+for\s+applying |
        applied\s+(?:to|for)\s+ |
        your\s+candidat(?:ure|acy) |
        your\s+(?:profile|resume|cv|candidature)\s+(?:has\s+been\s+)?
            (?:shortlisted|selected|reviewed) |
        shortlisted |
        interview\s+(?:invitation|invite|schedul|confirm|request|round) |
        (?:schedule|scheduling|book)\s+(?:an?\s+)?(?:interview|call|chat|screen) |
        (?:phone|technical|final|onsite|on-site|hr)\s+(?:interview|round|screen) |
        screening\s+(?:call|round|interview) |
        (?:coding|online|technical|hackerrank|codility|hackerearth)\s+assessment |
        assessment\s+(?:link|invitation|invite|for\s+completion|deadline) |
        take[-\s]?home\s+(?:assignment|test|exercise) |
        offer\s+letter |
        we\s+(?:would|'d)\s+like\s+to\s+(?:invite|schedule|speak|move\s+forward) |
        moving\s+forward\s+with\s+your |
        next\s+steps\s+(?:in|for|on)\s+your |
        regret\s+to\s+inform |
        not\s+(?:be\s+)?(?:moving\s+forward|selected|shortlisted|proceeding) |
        no\s+longer\s+(?:be\s+)?(?:moving|considering|under\s+consideration) |
        decided\s+not\s+to\s+(?:move|proceed|continue) |
        unfortunately[,]?\s+(?:we|your\s+application)
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Applicant tracking systems. They exist only to mail candidates about their own
# applications, so the sender alone is a strong signal.
_ATS_DOMAINS = (
    "greenhouse.io",
    "greenhouse-mail.io",
    "us.greenhouse-mail.io",
    "lever.co",
    "hire.lever.co",
    "myworkday.com",
    "workday.com",
    "wd1.myworkdayjobs.com",
    "smartrecruiters.com",
    "ashbyhq.com",
    "jobvite.com",
    "icims.com",
    "taleo.net",
    "successfactors.com",
    "bamboohr.com",
    "recruitee.com",
    "workable.com",
    "workablemail.com",
    "breezy.hr",
    "eightfold.ai",
    "phenompeople.com",
    "avature.net",
)

# Local-parts an *employer* mails candidates from. Only counts on a domain that
# is not a job board — `jobs-noreply@linkedin.com` matches "jobs" but sends far
# more alerts than application updates.
_EMPLOYER_LOCALPARTS = (
    "careers",
    "career",
    "recruiting",
    "recruitment",
    "recruiter",
    "talent",
    "hiring",
    "candidate",
    "candidates",
    "applications",
    "apply",
    "hr",
    "people",
    "campus",
)

# --------------------------------------------------------------------------
# Platforms and noise
# --------------------------------------------------------------------------

# Job boards, career platforms and professional networks. Mail from these is
# neutral: they send genuine application updates *and* a far larger volume of
# alerts, digests and promotions. Only an explicit strong signal gets one
# through, which is why they are listed separately from the employer rules.
_PLATFORM_DOMAINS = (
    "linkedin.com",
    "em.linkedin.com",
    "naukri.com",
    "unstop.com",
    "unstop.news",
    "internshala.com",
    "indeed.com",
    "glassdoor.com",
    "joinhandshake.com",
    "g.joinhandshake.com",
    "wellfound.com",
    "angel.co",
    "instahyre.com",
    "cutshort.io",
    "hirist.com",
    "shine.com",
    "monster.com",
    "foundit.in",
    "dorahacks.io",
    "devfolio.co",
    "codingninjas.com",
    "unstop.email",
    "unstop.events",
    "coursera.org",
    "udemy.com",
    "outlier.ai",
    "turing.com",
    "topmate.io",
)

# Notification channels. A platform mails application updates from its main
# address; these local-parts carry social noise, digests and marketing only.
_NOISE_LOCALPART = re.compile(
    r"""
    ^(?:
        invitations? | invites? |
        notifications? | notify |
        newsletters? | news |
        digest | .*-digest | messaging-digest |
        groups? | community |
        updates? | announce(?:ments?)? |
        recommendations? | recommendation\w* |
        marketing | promo(?:tions?)? | offers? | deals? |
        social | connect | network |
        webinars? | events? | hackathons? |
        billing | receipts? | invoice
    )(?:[-.](?:noreply|no-reply|donotreply))?$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Subject lines that identify bulk job-search noise on their own, whatever the
# sender's technical headers say. Anchored to phrasing that never appears in a
# message about an application the recipient actually submitted.
_NOISE_SUBJECT = re.compile(
    r"""
    (
        \bis\s+hiring\b |                             # "Acme is hiring for a Remote role"
        \bwe(?:'re|\s+are)\s+hiring\b |
        \bapply\s+now\b |
        \bjobs?\s+(?:for\s+you|you\s+may|alerts?|picks?|matches)\b |
        \b\d+\+?\s+(?:new\s+)?(?:jobs?|openings?|opportunit(?:y|ies)|internships?)\b |
        \bnew\s+jobs?\s+(?:in|at|for)\b |
        \brecommended\s+(?:for\s+you|jobs?|internships?)\b |
        \btrending\s+(?:jobs?|internships?|opportunit)\b |
        \bbased\s+on\s+your\s+profile\b |
        \byour\s+job\s+alert\b |
        \bjob\s+(?:alert|digest)\b |
        \bexpir(?:es|ing)\s+(?:soon|on|in)\b |
        \bjust\s+messaged\s+you\b |
        \baccepted\s+your\s+invitation\b |
        \b(?:wants?|want)\s+to\s+connect\b |
        \bpeople\s+you\s+may\s+know\b |
        \bexplore\s+their\s+network\b |
        \byour\s+posts?\s+got\b |
        \bimpressions\s+last\s+week\b |
        \binvitation\s+to\s+connect\b |
        \bwebinar\b |
        \bfree\s+(?:course|trial|certificate)\b |
        \b(?:enroll|register)\s+now\b |
        \bstipend\b |
        \bhackathon\b |
        \bnewsletter\b |
        \bunsubscribe\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Senders that are never job mail, however they phrase the subject line.
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
    "flipkart.com",
    "paytm.com",
    "phonepe.com",
    "razorpay.com",
    "goindigo.in",
    "fnp.com",
    "fitpass.email",
    "fitpass.co.in",
    "shiprocket.in",
    "accounts.google.com",
    "medium.com",
    "meetup.com",
    "github.com",
    "policybazaar.com",
    # Course marketing that phrases every send as "confirm your application".
    "certifications.codingninjas.com",
)

# Weak hints. Not enough to override anything — they only keep an otherwise
# unremarkable message alive for the classifier to judge.
_WEAK_JOB_WORDS = re.compile(
    r"\b(applicat(?:ion|ions)|applied|interview|recruit(?:er|ing|ment)|"
    r"candidat(?:e|ure)|hiring|assessment|offer|role|position|vacancy|opening)\b",
    re.IGNORECASE,
)

# Addresses no human writes from. Used only at the very end, to separate
# genuinely personal mail (worth an LLM call on principle) from transactional
# receipts that merely lack bulk headers — order confirmations, boarding passes,
# file-share notices.
_AUTOMATED_LOCALPART = re.compile(
    r"^(?:no[-_.]?reply|donot[-_.]?reply|noreply\S*|support|help|service|mailer|"
    r"postmaster|bounce\S*|automated|system|alerts?|store|billing|orders?|"
    r"tickets?|booking\S*|welcome|hello|info|contact)(?:[-+.]\S*)?$",
    re.IGNORECASE,
)

_BULK_HEADERS = ("list-unsubscribe", "list-id")
_BULK_PRECEDENCE = {"bulk", "list", "junk"}
_PROMO_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS", "SPAM", "TRASH"}


# --------------------------------------------------------------------------
# Sender classification
# --------------------------------------------------------------------------


def _split(addr: str | None) -> tuple[str, str]:
    sender = (addr or "").lower().strip()
    if "@" not in sender:
        return sender, ""
    local, _, domain = sender.rpartition("@")
    return local, domain


def _domain_matches(domain: str, suffixes: tuple[str, ...]) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in suffixes)


def is_platform_sender(email: Email) -> bool:
    _, domain = _split(email.from_addr)
    return bool(domain) and _domain_matches(domain, _PLATFORM_DOMAINS)


def is_ats_sender(email: Email) -> bool:
    _, domain = _split(email.from_addr)
    return bool(domain) and _domain_matches(domain, _ATS_DOMAINS)


def is_employer_sender(email: Email) -> bool:
    """An employer's own recruiting address, e.g. `careers@americanexpress.com`.

    Excludes job boards on purpose: they use the same local-parts for alerts.
    """
    local, domain = _split(email.from_addr)
    if not domain or _domain_matches(domain, _PLATFORM_DOMAINS):
        return False
    if _domain_matches(domain, _NEVER_DOMAIN_SUFFIXES):
        return False
    # `.jobs` is a sponsored TLD restricted to employers advertising their own
    # vacancies, so `noreply@mail.amazon.jobs` qualifies on the domain alone.
    if domain.endswith(".jobs"):
        return True
    # Match on token boundaries so `hr@` counts but `chairman@` does not.
    tokens = set(re.split(r"[^a-z0-9]+", local))
    if tokens & set(_EMPLOYER_LOCALPARTS):
        return True
    # Employers often mail from a dedicated subdomain, e.g.
    # `careers@recruitment.americanexpress.com`.
    return any(part in _EMPLOYER_LOCALPARTS for part in domain.split(".")[:-2])


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------


def _subject_and_snippet(email: Email) -> str:
    return " ".join([email.subject or "", email.snippet or ""])


def has_job_signal(email: Email) -> bool:
    """True when the message looks like it concerns the recipient's own application.

    This is the strong signal. It drives three things: overriding the bulk-mail
    rejections below, ordering the backlog so real mail is classified first, and
    the degraded verdict used when no LLM is configured at all. All three want
    the strict reading — a connection invite from LinkedIn is not a job signal.
    """
    if _OWN_APPLICATION.search(_subject_and_snippet(email)):
        return True
    return is_ats_sender(email) or is_employer_sender(email)


def has_weak_job_signal(email: Email) -> bool:
    """True when the message merely mentions hiring. Not enough to override anything."""
    return bool(_WEAK_JOB_WORDS.search(_subject_and_snippet(email)))


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def prefilter(email: Email) -> PrefilterResult:
    if not (email.subject or email.body_text or email.snippet):
        return PrefilterResult("reject", "empty message")

    strong = has_job_signal(email)
    platform = is_platform_sender(email)

    # An employer or ATS is exempt from every rule below: a genuine
    # acknowledgement carries List-Unsubscribe and usually lands in Promotions,
    # so the bulk-mail checks would throw away exactly the mail that matters.
    #
    # Job boards get no such exemption. They are the ones that dress marketing
    # in application language ("Confirm your application to build a portfolio"),
    # so their mail is checked for noise first and only then credited.
    if strong and not platform:
        return PrefilterResult("pass", "own-application signal")

    subject = email.subject or ""
    if _NOISE_SUBJECT.search(subject):
        return PrefilterResult("reject", "job-board alert or social notification")

    local, domain = _split(email.from_addr)

    if domain and _domain_matches(domain, _NEVER_DOMAIN_SUFFIXES):
        return PrefilterResult("reject", f"domain {domain} never sends job mail")

    if _NOISE_LOCALPART.match(local):
        return PrefilterResult("reject", f"notification channel {local}@")

    if platform:
        # A board does forward real application updates ("your application was
        # sent to Triomics"), so a surviving strong signal still counts.
        if strong:
            return PrefilterResult("pass", f"{domain} naming the recipient's own application")
        return PrefilterResult("reject", f"job board {domain} with no application signal")

    labels = set(email.labels or [])
    if labels & _PROMO_LABELS:
        return PrefilterResult("reject", "promotional/social category with no application signal")

    headers = {k.lower(): v for k, v in (email.raw_headers or {}).items()}

    if any(h in headers for h in _BULK_HEADERS):
        return PrefilterResult("reject", "bulk mail headers with no application signal")

    precedence = (headers.get("precedence") or "").strip().lower()
    if precedence in _BULK_PRECEDENCE:
        return PrefilterResult("reject", f"precedence: {precedence} with no application signal")

    if headers.get("auto-submitted", "").lower().startswith("auto-generated"):
        return PrefilterResult("reject", "auto-generated with no application signal")

    weak = has_weak_job_signal(email)

    # What's left has no bulk markers at all. Most of it is still transactional
    # — order confirmations, boarding passes, file shares — which is worth
    # separating out, because the sender address gives it away and nobody
    # recruits from `no-reply@`.
    if not weak and _AUTOMATED_LOCALPART.match(local):
        return PrefilterResult("reject", f"automated sender {local}@ with no job signal")

    # Genuinely personal, one-to-one mail. This is where a miss costs the most
    # and where keywords help the least — a recruiter writing directly may only
    # say "are you free to chat Thursday?" — so all of it goes to the LLM. It is
    # also the smallest bucket by far, which is what makes that affordable; the
    # volume was never here, it was in the digests above.
    hint = "mentions hiring" if weak else "personal mail"
    return PrefilterResult("pass", f"ambiguous ({hint}), defer to LLM")
