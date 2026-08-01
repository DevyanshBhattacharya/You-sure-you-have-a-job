"""Pydantic models for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Emails
# --------------------------------------------------------------------------


class EmailSummary(ORMModel):
    id: int
    gmail_id: str
    thread_id: str | None = None
    from_addr: str | None = None
    from_name: str | None = None
    subject: str | None = None
    snippet: str | None = None
    received_at: datetime | None = None
    is_job_related: bool | None = None
    classification_confidence: float | None = None
    classification_source: str | None = None


class EmailDetail(EmailSummary):
    to_addr: str | None = None
    body_text: str | None = None
    labels: list[str] | None = None
    classification_raw: dict | None = None
    processed_at: datetime | None = None


class EmailPage(BaseModel):
    items: list[EmailSummary]
    total: int
    limit: int
    offset: int


class ClassificationOverride(BaseModel):
    is_job_related: bool


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------


class SyncStatus(BaseModel):
    running: bool
    status: str | None = None
    total: int = 0
    done: int = 0
    error: str | None = None
    last_history_id: str | None = None


class BackfillRequest(BaseModel):
    days: int | None = None


class HealthResponse(BaseModel):
    status: str
    gmail_authorised: bool
    gmail_address: str | None = None
    emails_stored: int
    applications: int
    watcher_running: bool
    llm_calls: int
    prompt_tokens: int
    output_tokens: int


# --------------------------------------------------------------------------
# Applications
# --------------------------------------------------------------------------


class ApplicationSummary(ORMModel):
    id: int
    company: str
    role_title: str | None = None
    location: str | None = None
    source: str | None = None
    status: str
    job_url: str | None = None
    salary_text: str | None = None
    contact_name: str | None = None
    contact_email: str | None = None
    next_action: str | None = None
    next_action_due: datetime | None = None
    first_seen_at: datetime
    last_activity_at: datetime


class TimelineEntry(ORMModel):
    id: int
    email_id: int | None = None
    event_type: str
    occurred_at: datetime
    summary: str | None = None
    confidence: float | None = None
    status_before: str | None = None
    status_after: str | None = None


class ApplicationDetail(ApplicationSummary):
    notes: str | None = None
    timeline: list[TimelineEntry] = []


class ApplicationPage(BaseModel):
    items: list[ApplicationSummary]
    total: int
    limit: int
    offset: int


class BoardColumn(BaseModel):
    status: str
    count: int
    items: list[ApplicationSummary]


class BoardResponse(BaseModel):
    columns: list[BoardColumn]


class DashboardStats(BaseModel):
    total: int
    active: int
    interviewing: int
    offers: int
    rejected: int
    awaiting_reply: int
    actions_due_7d: int


class StatusOverride(BaseModel):
    status: str


class NotesUpdate(BaseModel):
    notes: str


# --------------------------------------------------------------------------
# Notifications
# --------------------------------------------------------------------------


class NotificationOut(ORMModel):
    id: int
    kind: str
    priority: str
    title: str
    body: str | None = None
    application_id: int | None = None
    email_id: int | None = None
    created_at: datetime
    read_at: datetime | None = None


class NotificationPage(BaseModel):
    items: list[NotificationOut]
    total: int
    unread: int


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


class ChatTurn(BaseModel):
    role: str  # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


class Citation(BaseModel):
    email_id: int | None = None
    application_id: int | None = None
    subject: str | None = None
    company: str | None = None
    received_at: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    tool_calls: list[str] = []
