"""Email listing, detail, and manual classification override."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Email, utcnow
from app.schemas import ClassificationOverride, EmailDetail, EmailPage, EmailSummary

log = logging.getLogger(__name__)
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
    """Manually correct a classification, and act on the correction.

    Flipping the flag alone was not a correction, it was a note-to-self: the
    board is built from applications, so marking a missed acknowledgement "job
    related" changed a label and nothing else, and marking a bogus entry "not
    job related" left the application it had already created sitting there.

    Recorded with source='manual' so the verdict survives later
    re-classification, and so these corrections can be pulled out as an
    evaluation set for the classifier prompt.
    """
    from app.agent import pipeline, resolve

    email = db.get(Email, email_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Email not found")

    email.is_job_related = payload.is_job_related
    email.classification_source = "manual"
    email.classification_confidence = 1.0
    email.processed_at = utcnow()

    retracted: list[int] = []
    if not payload.is_job_related:
        # Cheap and immediate: removing the timeline entry, its notifications
        # and any application left with no events is pure database work.
        retracted = resolve.retract(db, email)

    db.commit()

    if payload.is_job_related:
        # Needs the model to name the company and role, which is far too slow
        # for a request handler, so it goes through the pipeline. `classify`
        # sees source='manual' and skips the prefilter that rejected it.
        pipeline.submit_email_id(email_id, reclassify=True)

    db.refresh(email)
    if retracted:
        log.info(
            "Manual override on email %s retracted %d application(s)", email_id, len(retracted)
        )
    return EmailDetail.model_validate(email)
