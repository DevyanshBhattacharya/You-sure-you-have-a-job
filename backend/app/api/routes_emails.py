"""Email listing, detail, and manual classification override."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Email, utcnow
from app.schemas import ClassificationOverride, EmailDetail, EmailPage, EmailSummary

router = APIRouter(prefix="/api/emails", tags=["emails"])


@router.get("", response_model=EmailPage)
def list_emails(
    db: Session = Depends(get_db),
    job_related: bool | None = Query(None, description="Filter by classification outcome"),
    unclassified: bool = Query(False, description="Only emails not yet classified"),
    q: str | None = Query(None, description="Substring match on subject or sender"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> EmailPage:
    stmt = select(Email)
    count_stmt = select(func.count()).select_from(Email)

    filters = []
    if unclassified:
        filters.append(Email.is_job_related.is_(None))
    elif job_related is not None:
        filters.append(Email.is_job_related.is_(job_related))

    if q:
        pattern = f"%{q}%"
        filters.append(Email.subject.ilike(pattern) | Email.from_addr.ilike(pattern))

    for f in filters:
        stmt = stmt.where(f)
        count_stmt = count_stmt.where(f)

    total = db.scalar(count_stmt) or 0
    rows = db.scalars(
        stmt.order_by(Email.received_at.desc().nullslast()).limit(limit).offset(offset)
    ).all()

    return EmailPage(
        items=[EmailSummary.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{email_id}", response_model=EmailDetail)
def get_email(email_id: int, db: Session = Depends(get_db)) -> EmailDetail:
    email = db.get(Email, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return EmailDetail.model_validate(email)


@router.post("/{email_id}/classification", response_model=EmailDetail)
def override_classification(
    email_id: int,
    payload: ClassificationOverride,
    db: Session = Depends(get_db),
) -> EmailDetail:
    """Manually correct a classification.

    Recorded with source='manual' so these corrections can later be pulled out
    as an evaluation set for the classifier prompt.
    """
    email = db.get(Email, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")

    email.is_job_related = payload.is_job_related
    email.classification_source = "manual"
    email.classification_confidence = 1.0
    email.processed_at = utcnow()
    db.commit()
    db.refresh(email)
    return EmailDetail.model_validate(email)
