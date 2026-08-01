"""ORM models and the domain enums the pipeline reasons about."""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes, on every backend.

    SQLite has no native timestamp type and silently discards tzinfo, so a
    value written as aware comes back naive. Comparing one against a freshly
    constructed aware datetime then raises TypeError. Normalising in the column
    type fixes it once, instead of scattering defensive checks everywhere.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class EnumAsString(TypeDecorator):
    """Store an enum as its plain string value, load it back as the enum.

    A bare `String` column round-trips to `str`, so `row.status` would be
    `"offer"` rather than `ApplicationStatus.OFFER` and every comparison site
    would need a defensive coercion. Doing it here keeps the `Mapped[...]`
    annotations honest.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_cls: type[enum.Enum], length: int = 32) -> None:
        self._enum_cls = enum_cls
        super().__init__(length)

    def process_bind_param(self, value, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        if isinstance(value, enum.Enum):
            return value.value
        return str(value)

    def process_result_value(self, value, dialect):  # noqa: ANN001, ANN201
        if value is None:
            return None
        try:
            return self._enum_cls(value)
        except ValueError:
            # Tolerate rows written before an enum member was renamed.
            return value


class Base(DeclarativeBase):
    pass


class ApplicationStatus(enum.StrEnum):
    """Current state of an application. Ordered by pipeline progression.

    StrEnum, not `(str, Enum)`: `str(status)` then yields the stored value
    rather than "ApplicationStatus.APPLIED", which keeps every fallback path
    that stringifies a status correct.
    """

    DISCOVERED = "discovered"
    APPLIED = "applied"
    ACKNOWLEDGED = "acknowledged"
    SCREENING = "screening"
    INTERVIEWING = "interviewing"
    ASSESSMENT = "assessment"
    OFFER = "offer"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    GHOSTED = "ghosted"
    WITHDRAWN = "withdrawn"


class EventType(enum.StrEnum):
    """What a single job-related email represents."""

    APPLIED = "applied"
    ACKNOWLEDGEMENT = "acknowledgement"
    RECRUITER_OUTREACH = "recruiter_outreach"
    SCREEN_SCHEDULED = "screen_scheduled"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    ASSESSMENT_SENT = "assessment_sent"
    OFFER = "offer"
    REJECTION = "rejection"
    FOLLOW_UP = "follow_up"
    WITHDRAWN = "withdrawn"
    OTHER = "other"


class NotificationPriority(enum.StrEnum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    gmail_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    thread_id: Mapped[str | None] = mapped_column(String(64), index=True)
    history_id: Mapped[str | None] = mapped_column(String(32))

    from_addr: Mapped[str | None] = mapped_column(String(320), index=True)
    from_name: Mapped[str | None] = mapped_column(String(255))
    to_addr: Mapped[str | None] = mapped_column(String(320))
    subject: Mapped[str | None] = mapped_column(String(998))
    snippet: Mapped[str | None] = mapped_column(Text)
    body_text: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)

    labels: Mapped[list | None] = mapped_column(JSON, default=list)
    raw_headers: Mapped[dict | None] = mapped_column(JSON, default=dict)

    # Classification outcome
    is_job_related: Mapped[bool | None] = mapped_column(Boolean, index=True)
    classification_confidence: Mapped[float | None] = mapped_column(Float)
    # One of: prefilter | llm | heuristic | manual
    classification_source: Mapped[str | None] = mapped_column(String(16))
    classification_raw: Mapped[dict | None] = mapped_column(JSON)
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_emails_job_received", "is_job_related", "received_at"),)


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company: Mapped[str] = mapped_column(String(255), index=True)
    company_normalized: Mapped[str] = mapped_column(String(255), index=True)
    role_title: Mapped[str | None] = mapped_column(String(255))
    role_normalized: Mapped[str | None] = mapped_column(String(255), index=True)

    location: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str | None] = mapped_column(String(128))
    job_url: Mapped[str | None] = mapped_column(Text)
    salary_text: Mapped[str | None] = mapped_column(String(255))

    status: Mapped[ApplicationStatus] = mapped_column(
        EnumAsString(ApplicationStatus), default=ApplicationStatus.DISCOVERED, index=True
    )

    contact_name: Mapped[str | None] = mapped_column(String(255))
    contact_email: Mapped[str | None] = mapped_column(String(320))

    next_action: Mapped[str | None] = mapped_column(Text)
    next_action_due: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)

    notes: Mapped[str | None] = mapped_column(Text)

    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_activity_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)

    events: Mapped[list[ApplicationEvent]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationEvent.occurred_at",
    )


class ApplicationEvent(Base):
    """Append-only timeline. `Application.status` is the derived current state."""

    __tablename__ = "application_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    email_id: Mapped[int | None] = mapped_column(
        ForeignKey("emails.id", ondelete="CASCADE"), index=True
    )

    event_type: Mapped[EventType] = mapped_column(EnumAsString(EventType), index=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    summary: Mapped[str | None] = mapped_column(Text)
    extracted: Mapped[dict | None] = mapped_column(JSON)
    confidence: Mapped[float | None] = mapped_column(Float)

    status_before: Mapped[str | None] = mapped_column(String(32))
    status_after: Mapped[str | None] = mapped_column(String(32))

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    application: Mapped[Application] = relationship(back_populates="events")
    email: Mapped[Email | None] = relationship(back_populates="events")

    __table_args__ = (
        # One timeline entry per (application, email) keeps re-ingestion idempotent.
        Index("ix_events_app_email", "application_id", "email_id", unique=True),
    )


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    priority: Mapped[NotificationPriority] = mapped_column(
        EnumAsString(NotificationPriority, 16), default=NotificationPriority.NORMAL, index=True
    )
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str | None] = mapped_column(Text)

    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL")
    )
    email_id: Mapped[int | None] = mapped_column(ForeignKey("emails.id", ondelete="SET NULL"))

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    read_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class KBChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int | None] = mapped_column(
        ForeignKey("emails.id", ondelete="CASCADE"), index=True
    )
    application_id: Mapped[int | None] = mapped_column(
        ForeignKey("applications.id", ondelete="SET NULL"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    dim: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class SyncState(Base):
    """Key/value bookkeeping for the Gmail cursor and usage counters."""

    __tablename__ = "sync_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
