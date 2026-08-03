"""Pydantic models for the HTTP API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    # Importing and classifying are separate stages with very different speeds:
    # a local model takes tens of seconds per email, so an import can finish
    # while the board stays empty for a long time afterwards. Reporting only
    # import progress makes that look like nothing happened.
    pending_classification: int = 0
    watcher_running: bool = False
    last_sync_at: datetime | None = None
    # LLM quota back-off. Deferred emails stay unprocessed and are retried on
    # the next start rather than being classified with a degraded fallback.
    quota_blocked: bool = False
    quota_retry_in_seconds: int = 0
    quota_reason: str = ""
    quota_deferred: int = 0


class BackfillRequest(BaseModel):
    days: int | None = None


class HealthResponse(BaseModel):
    status: str
    # `authorised` = we hold credentials. `usable` = we can actually call the
    # API. They differ when e.g. the Gmail API isn't enabled on the project.
    gmail_authorised: bool
    gmail_usable: bool = False
    gmail_address: str | None = None
    gmail_error: str | None = None
    gmail_hint: str | None = None
    emails_stored: int
    emails_unprocessed: int = 0
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


# Bounds, not guesses. An unbounded message or history is a free way to make the
# server hold megabytes and hand them to a model that charges (or, locally,
# thrashes) per token. The loop already trims to the last 12 turns, so a longer
# history was never read — it was only ever parsed and dropped.
MAX_CHAT_MESSAGE_CHARS = 4_000
MAX_CHAT_TURN_CHARS = 8_000
MAX_CHAT_HISTORY_TURNS = 40


class ChatTurn(BaseModel):
    role: str  # "user" | "model"
    content: str = Field(max_length=MAX_CHAT_TURN_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY_TURNS)


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
