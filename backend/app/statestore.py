"""Helpers over the `sync_state` key/value table."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import SyncState

LAST_HISTORY_ID = "last_history_id"
LAST_SYNC_AT = "last_sync_at"
BACKFILL_COMPLETE = "backfill_complete"
TOKENS_PROMPT = "tokens_prompt_total"
TOKENS_OUTPUT = "tokens_output_total"
LLM_CALLS = "llm_calls_total"


def get(session: Session, key: str, default: str | None = None) -> str | None:
    row = session.get(SyncState, key)
    return row.value if row and row.value is not None else default


def set_(session: Session, key: str, value: str | None) -> None:
    row = session.get(SyncState, key)
    if row is None:
        session.add(SyncState(key=key, value=value))
    else:
        row.value = value


def get_int(session: Session, key: str, default: int = 0) -> int:
    raw = get(session, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def increment(session: Session, key: str, amount: int) -> int:
    total = get_int(session, key) + amount
    set_(session, key, str(total))
    return total


def all_state(session: Session) -> dict[str, str | None]:
    return {row.key: row.value for row in session.query(SyncState).all()}
