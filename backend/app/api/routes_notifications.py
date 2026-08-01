"""Notification feed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Notification, NotificationPriority, utcnow
from app.schemas import NotificationOut, NotificationPage

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


def _to_out(row: Notification) -> NotificationOut:
    priority = (
        row.priority.value if isinstance(row.priority, NotificationPriority) else str(row.priority)
    )
    return NotificationOut(
        id=row.id,
        kind=row.kind,
        priority=priority,
        title=row.title,
        body=row.body,
        application_id=row.application_id,
        email_id=row.email_id,
        created_at=row.created_at,
        read_at=row.read_at,
    )


@router.get("", response_model=NotificationPage)
def list_notifications(
    db: Session = Depends(get_db),
    unread_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
) -> NotificationPage:
    stmt = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    if unread_only:
        stmt = stmt.where(Notification.read_at.is_(None))

    rows = db.scalars(stmt).all()
    total = db.scalar(select(func.count()).select_from(Notification)) or 0
    unread = (
        db.scalar(
            select(func.count()).select_from(Notification).where(Notification.read_at.is_(None))
        )
        or 0
    )

    return NotificationPage(items=[_to_out(r) for r in rows], total=total, unread=unread)


@router.post("/{notification_id}/read", response_model=NotificationOut)
def mark_read(notification_id: int, db: Session = Depends(get_db)) -> NotificationOut:
    row = db.get(Notification, notification_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    if row.read_at is None:
        row.read_at = utcnow()
        db.commit()
        db.refresh(row)
    return _to_out(row)


@router.post("/read-all", response_model=NotificationPage)
def mark_all_read(db: Session = Depends(get_db)) -> NotificationPage:
    db.execute(
        update(Notification).where(Notification.read_at.is_(None)).values(read_at=utcnow())
    )
    db.commit()
    return list_notifications(db=db, unread_only=False, limit=50)
