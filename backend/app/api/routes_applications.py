"""Application board, list, detail, and manual overrides."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.models import Application, ApplicationStatus, utcnow
from app.schemas import (
    ApplicationDetail,
    ApplicationPage,
    ApplicationSummary,
    BoardColumn,
    BoardResponse,
    DashboardStats,
    NotesUpdate,
    StatusOverride,
    TimelineEntry,
)

router = APIRouter(prefix="/api/applications", tags=["applications"])

# Column order on the board, left to right.
BOARD_ORDER = [
    ApplicationStatus.DISCOVERED,
    ApplicationStatus.APPLIED,
    ApplicationStatus.ACKNOWLEDGED,
    ApplicationStatus.SCREENING,
    ApplicationStatus.ASSESSMENT,
    ApplicationStatus.INTERVIEWING,
    ApplicationStatus.OFFER,
    ApplicationStatus.ACCEPTED,
    ApplicationStatus.REJECTED,
]

_ACTIVE = {
    ApplicationStatus.DISCOVERED,
    ApplicationStatus.APPLIED,
    ApplicationStatus.ACKNOWLEDGED,
    ApplicationStatus.SCREENING,
    ApplicationStatus.ASSESSMENT,
    ApplicationStatus.INTERVIEWING,
    ApplicationStatus.OFFER,
}


def _status_value(app: Application) -> str:
    return app.status.value if isinstance(app.status, ApplicationStatus) else str(app.status)


def _to_summary(app: Application) -> ApplicationSummary:
    return ApplicationSummary(
        id=app.id,
        company=app.company,
        role_title=app.role_title,
        location=app.location,
        source=app.source,
        status=_status_value(app),
        job_url=app.job_url,
        salary_text=app.salary_text,
        contact_name=app.contact_name,
        contact_email=app.contact_email,
        next_action=app.next_action,
        next_action_due=app.next_action_due,
        first_seen_at=app.first_seen_at,
        last_activity_at=app.last_activity_at,
    )


@router.get("", response_model=ApplicationPage)
def list_applications(
    db: Session = Depends(get_db),
    status: str | None = Query(None),
    company: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ApplicationPage:
    stmt = select(Application)
    count_stmt = select(func.count()).select_from(Application)

    if status:
        stmt = stmt.where(Application.status == status)
        count_stmt = count_stmt.where(Application.status == status)
    if company:
        pattern = f"%{company}%"
        stmt = stmt.where(Application.company.ilike(pattern))
        count_stmt = count_stmt.where(Application.company.ilike(pattern))

    total = db.scalar(count_stmt) or 0
    rows = db.scalars(
        stmt.order_by(Application.last_activity_at.desc()).limit(limit).offset(offset)
    ).all()

    return ApplicationPage(
        items=[_to_summary(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get("/board", response_model=BoardResponse)
def board(db: Session = Depends(get_db)) -> BoardResponse:
    rows = db.scalars(select(Application).order_by(Application.last_activity_at.desc())).all()

    grouped: dict[str, list[ApplicationSummary]] = {s.value: [] for s in BOARD_ORDER}
    for app in rows:
        grouped.setdefault(_status_value(app), []).append(_to_summary(app))

    columns = [
        BoardColumn(
            status=s.value,
            count=len(grouped.get(s.value, [])),
            items=grouped.get(s.value, []),
        )
        for s in BOARD_ORDER
    ]
    # Anything in a status outside the standard board (withdrawn, ghosted).
    for status_value, items in grouped.items():
        if status_value not in {s.value for s in BOARD_ORDER} and items:
            columns.append(BoardColumn(status=status_value, count=len(items), items=items))

    return BoardResponse(columns=columns)


@router.get("/stats", response_model=DashboardStats)
def stats(db: Session = Depends(get_db)) -> DashboardStats:
    # Key by the plain string: StrEnum members hash to their value, so a dict
    # keyed by enums would be hit by *both* `counts[s]` and `counts[s.value]`
    # and double-count anything summed over the two.
    counts: dict[str, int] = {}
    for status_value, n in db.execute(
        select(Application.status, func.count()).group_by(Application.status)
    ).all():
        counts[str(status_value)] = counts.get(str(status_value), 0) + n

    def count_of(*statuses: ApplicationStatus) -> int:
        return sum(counts.get(s.value, 0) for s in statuses)

    horizon = utcnow() + timedelta(days=7)
    actions_due = (
        db.scalar(
            select(func.count())
            .select_from(Application)
            .where(
                Application.next_action_due.is_not(None),
                Application.next_action_due <= horizon,
                Application.status.notin_(
                    [
                        ApplicationStatus.REJECTED.value,
                        ApplicationStatus.WITHDRAWN.value,
                        ApplicationStatus.ACCEPTED.value,
                    ]
                ),
            )
        )
        or 0
    )

    awaiting = (
        db.scalar(
            select(func.count())
            .select_from(Application)
            .where(
                Application.status.in_(
                    [ApplicationStatus.APPLIED.value, ApplicationStatus.ACKNOWLEDGED.value]
                ),
                Application.last_activity_at < utcnow() - timedelta(days=14),
            )
        )
        or 0
    )

    return DashboardStats(
        total=sum(counts.values()),
        active=count_of(*_ACTIVE),
        interviewing=count_of(
            ApplicationStatus.INTERVIEWING,
            ApplicationStatus.SCREENING,
            ApplicationStatus.ASSESSMENT,
        ),
        offers=count_of(ApplicationStatus.OFFER, ApplicationStatus.ACCEPTED),
        rejected=count_of(ApplicationStatus.REJECTED),
        awaiting_reply=awaiting,
        actions_due_7d=actions_due,
    )


@router.get("/{application_id}", response_model=ApplicationDetail)
def get_application(application_id: int, db: Session = Depends(get_db)) -> ApplicationDetail:
    app = db.scalar(
        select(Application)
        .options(selectinload(Application.events))
        .where(Application.id == application_id)
    )
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    timeline = [
        TimelineEntry(
            id=e.id,
            email_id=e.email_id,
            event_type=e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
            occurred_at=e.occurred_at,
            summary=e.summary,
            confidence=e.confidence,
            status_before=e.status_before,
            status_after=e.status_after,
        )
        for e in sorted(app.events, key=lambda x: x.occurred_at)
    ]

    base = _to_summary(app)
    return ApplicationDetail(**base.model_dump(), notes=app.notes, timeline=timeline)


@router.post("/{application_id}/status", response_model=ApplicationSummary)
def override_status(
    application_id: int, payload: StatusOverride, db: Session = Depends(get_db)
) -> ApplicationSummary:
    """Manual status change. Bypasses the state machine on purpose — the user
    knows things the mailbox doesn't."""
    app = db.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")

    try:
        new_status = ApplicationStatus(payload.status)
    except ValueError as exc:
        valid = ", ".join(s.value for s in ApplicationStatus)
        raise HTTPException(
            status_code=422, detail=f"Unknown status '{payload.status}'. Valid: {valid}"
        ) from exc

    app.status = new_status
    app.last_activity_at = utcnow()
    db.commit()
    db.refresh(app)
    return _to_summary(app)


@router.post("/{application_id}/notes", response_model=ApplicationSummary)
def update_notes(
    application_id: int, payload: NotesUpdate, db: Session = Depends(get_db)
) -> ApplicationSummary:
    app = db.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")
    app.notes = payload.notes
    db.commit()
    db.refresh(app)
    return _to_summary(app)


@router.delete("/{application_id}", status_code=204)
def delete_application(application_id: int, db: Session = Depends(get_db)) -> None:
    app = db.get(Application, application_id)
    if app is None:
        raise HTTPException(status_code=404, detail="Application not found")
    db.delete(app)
    db.commit()
