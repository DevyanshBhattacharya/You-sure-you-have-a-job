"""Tools the Q&A agent can call.

Semantic search alone answers "what did the recruiter at Acme say", but not
"how many applications are in the interview stage" — that is a SQL question.
Giving the model both retrieval and structured queries covers each kind.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from google.genai import types
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.kb import indexer
from app.kb.store import hydrate_hits, store
from app.models import Application, ApplicationEvent, ApplicationStatus, Email, utcnow

log = logging.getLogger(__name__)

MAX_ROWS = 100


# --------------------------------------------------------------------------
# Declarations
# --------------------------------------------------------------------------

FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_emails",
        description=(
            "Semantic search over the full text of job-related emails. Use for questions "
            "about what someone said, asked for, or wrote — anything needing the wording of "
            "a message rather than a count or a list."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="Natural-language description of what to find.",
                ),
                "k": types.Schema(
                    type=types.Type.INTEGER,
                    description="How many passages to return. Default 8, max 20.",
                ),
                "company": types.Schema(
                    type=types.Type.STRING,
                    description="Optional company name to restrict the search to.",
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="list_applications",
        description=(
            "List tracked job applications with their current status. Use for counting, "
            "filtering, or enumerating applications."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "status": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "Optional status filter, one of: "
                        + ", ".join(s.value for s in ApplicationStatus)
                    ),
                ),
                "company": types.Schema(
                    type=types.Type.STRING, description="Optional company substring filter."
                ),
                "days": types.Schema(
                    type=types.Type.INTEGER,
                    description="Optional: only applications active in the last N days.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_application_timeline",
        description=(
            "Full chronological history of one application: every email event, what it was, "
            "and how the status changed. Call after list_applications to get the id."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "application_id": types.Schema(
                    type=types.Type.INTEGER, description="Application id."
                ),
            },
            required=["application_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_upcoming_actions",
        description=(
            "Pending things the user has to do, and their deadlines. Use for questions about "
            "what is due, what is scheduled, or what needs a response."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "days": types.Schema(
                    type=types.Type.INTEGER,
                    description="Look-ahead window in days. Default 14.",
                ),
            },
        ),
    ),
]

TOOLS = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------


def _status_value(app: Application) -> str:
    return app.status.value if isinstance(app.status, ApplicationStatus) else str(app.status)


def _app_row(app: Application) -> dict:
    return {
        "application_id": app.id,
        "company": app.company,
        "role_title": app.role_title,
        "status": _status_value(app),
        "location": app.location,
        "source": app.source,
        "next_action": app.next_action,
        "next_action_due": app.next_action_due.isoformat() if app.next_action_due else None,
        "first_seen": app.first_seen_at.strftime("%Y-%m-%d") if app.first_seen_at else None,
        "last_activity": app.last_activity_at.strftime("%Y-%m-%d")
        if app.last_activity_at
        else None,
    }


def search_emails(
    session: Session, *, query: str, k: int = 8, company: str | None = None
) -> dict:
    k = max(1, min(int(k or 8), 20))

    application_id: int | None = None
    if company:
        app = session.scalar(
            select(Application)
            .where(Application.company.ilike(f"%{company}%"))
            .order_by(Application.last_activity_at.desc())
        )
        if app is not None:
            application_id = app.id

    vector = indexer.embed_query(query)
    if vector is None:
        return _keyword_fallback(session, query, k, reason="embeddings unavailable")

    hits = store.search(session, vector, k=k, application_id=application_id)
    if not hits:
        return _keyword_fallback(session, query, k, reason="no semantic matches")

    return {"results": hydrate_hits(session, hits)}


def _keyword_fallback(session: Session, query: str, k: int, *, reason: str) -> dict:
    """Substring search so the agent degrades instead of going blind."""
    pattern = f"%{query}%"
    rows = session.scalars(
        select(Email)
        .where(
            Email.is_job_related.is_(True),
            (Email.subject.ilike(pattern)) | (Email.body_text.ilike(pattern)),
        )
        .order_by(Email.received_at.desc())
        .limit(k)
    ).all()

    return {
        "note": f"Keyword search used ({reason}).",
        "results": [
            {
                "email_id": e.id,
                "subject": e.subject,
                "from": e.from_addr,
                "received_at": e.received_at.isoformat() if e.received_at else None,
                "text": (e.body_text or e.snippet or "")[:1500],
            }
            for e in rows
        ],
    }


def list_applications(
    session: Session,
    *,
    status: str | None = None,
    company: str | None = None,
    days: int | None = None,
) -> dict:
    stmt = select(Application)

    if status:
        normalised = status.strip().lower()
        valid = {s.value for s in ApplicationStatus}
        if normalised not in valid:
            return {
                "error": f"Unknown status '{status}'.",
                "valid_statuses": sorted(valid),
            }
        stmt = stmt.where(Application.status == normalised)

    if company:
        stmt = stmt.where(Application.company.ilike(f"%{company}%"))

    if days:
        stmt = stmt.where(Application.last_activity_at >= utcnow() - timedelta(days=int(days)))

    rows = session.scalars(
        stmt.order_by(Application.last_activity_at.desc()).limit(MAX_ROWS)
    ).all()

    total = session.scalar(select(func.count()).select_from(Application)) or 0
    by_status = dict(
        session.execute(select(Application.status, func.count()).group_by(Application.status)).all()
    )

    return {
        "count": len(rows),
        "total_applications": total,
        "counts_by_status": {str(k): v for k, v in by_status.items()},
        "applications": [_app_row(a) for a in rows],
    }


def get_application_timeline(session: Session, *, application_id: int) -> dict:
    app = session.get(Application, int(application_id))
    if app is None:
        return {"error": f"No application with id {application_id}"}

    events = session.scalars(
        select(ApplicationEvent)
        .where(ApplicationEvent.application_id == app.id)
        .order_by(ApplicationEvent.occurred_at)
    ).all()

    return {
        "application": _app_row(app),
        "timeline": [
            {
                "occurred_at": e.occurred_at.isoformat() if e.occurred_at else None,
                "event_type": e.event_type.value
                if hasattr(e.event_type, "value")
                else str(e.event_type),
                "summary": e.summary,
                "status_after": e.status_after,
                "email_id": e.email_id,
            }
            for e in events
        ],
    }


def get_upcoming_actions(session: Session, *, days: int = 14) -> dict:
    horizon = utcnow() + timedelta(days=int(days or 14))
    closed = [
        ApplicationStatus.REJECTED.value,
        ApplicationStatus.WITHDRAWN.value,
        ApplicationStatus.ACCEPTED.value,
    ]

    dated = session.scalars(
        select(Application)
        .where(
            Application.next_action_due.is_not(None),
            Application.next_action_due <= horizon,
            Application.status.notin_(closed),
        )
        .order_by(Application.next_action_due)
        .limit(MAX_ROWS)
    ).all()

    undated = session.scalars(
        select(Application)
        .where(
            Application.next_action.is_not(None),
            Application.next_action_due.is_(None),
            Application.status.notin_(closed),
        )
        .order_by(Application.last_activity_at.desc())
        .limit(MAX_ROWS)
    ).all()

    return {
        "now": utcnow().isoformat(),
        "window_days": int(days or 14),
        "with_deadline": [_app_row(a) for a in dated],
        "without_deadline": [_app_row(a) for a in undated],
    }


REGISTRY: dict[str, Callable[..., dict]] = {
    "search_emails": search_emails,
    "list_applications": list_applications,
    "get_application_timeline": get_application_timeline,
    "get_upcoming_actions": get_upcoming_actions,
}


def dispatch(session: Session, name: str, args: dict[str, Any]) -> dict:
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(session, **(args or {}))
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001 - the model should see the failure
        log.exception("Tool %s failed", name)
        return {"error": f"{name} failed: {exc}"}
