"""Turning stored emails into searchable, embedded chunks."""

from __future__ import annotations

import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.agent import llm
from app.kb import chunk as chunker
from app.kb.store import store, to_blob
from app.models import Email, KBChunk

log = logging.getLogger(__name__)

EMBED_BATCH = 32


def _header_for(email: Email) -> str:
    sender = email.from_name or email.from_addr or "unknown sender"
    when = email.received_at.strftime("%Y-%m-%d") if email.received_at else "unknown date"
    return f"[{when}] From: {sender} | Subject: {email.subject or '(no subject)'}"


def index_email(session: Session, email: Email, *, application_id: int | None = None) -> int:
    """(Re)index one email. Returns the number of chunks written."""
    body = (email.body_text or email.snippet or "").strip()
    if not body:
        return 0

    chunks = chunker.chunk_text(body, header=_header_for(email))
    if not chunks:
        return 0

    # Replace wholesale — simpler and safer than diffing chunk boundaries.
    session.execute(delete(KBChunk).where(KBChunk.email_id == email.id))

    texts = [c.text for c in chunks]
    vectors: list[list[float]] = []

    if llm.is_configured():
        try:
            for i in range(0, len(texts), EMBED_BATCH):
                batch, _usage = llm.embed(texts[i : i + EMBED_BATCH])
                vectors.extend(batch)
        except Exception:  # noqa: BLE001 - store the text even if embedding fails
            log.exception("Embedding failed for email %s; storing chunks unembedded", email.id)
            vectors = []

    for chunk in chunks:
        vector = vectors[chunk.index] if chunk.index < len(vectors) else None
        session.add(
            KBChunk(
                email_id=email.id,
                application_id=application_id,
                chunk_index=chunk.index,
                text=chunk.text,
                embedding=to_blob(vector) if vector else None,
                dim=len(vector) if vector else None,
                token_count=chunk.token_estimate,
            )
        )

    session.flush()
    store.invalidate()
    return len(chunks)


def remove_email(session: Session, email: Email) -> int:
    """Drop an email's chunks. Returns how many were removed.

    Indexing only ever ran on the way in, so an email that later turned out not
    to be job related kept its chunks forever and stayed searchable. The Q&A
    agent would then quote a LinkedIn digest as evidence about the job search,
    with no way for anyone to tell where it came from.
    """
    removed = session.execute(delete(KBChunk).where(KBChunk.email_id == email.id)).rowcount or 0
    if removed:
        session.flush()
        store.invalidate()
        log.info("Removed %d knowledge-base chunk(s) for email %s", removed, email.id)
    return removed


def stale_dimension_count(session: Session) -> int:
    """Chunks embedded by a different model than the one now configured.

    Vectors of different widths are not comparable, so the store only searches
    ones matching the current model — anything else is silently invisible rather
    than wrong. Counting them is how that becomes visible instead.
    """
    from sqlalchemy import func

    dims = session.execute(
        select(KBChunk.dim, func.count()).where(KBChunk.dim.is_not(None)).group_by(KBChunk.dim)
    ).all()
    if len(dims) < 2:
        return 0
    live = max(dims, key=lambda row: row[1])[0]
    return sum(count for dim, count in dims if dim != live)


def embed_query(text: str) -> list[float] | None:
    """Embed a search query. Returns None when embeddings are unavailable."""
    if not llm.is_configured():
        return None
    try:
        vectors, _usage = llm.embed([text], task_type="RETRIEVAL_QUERY")
    except Exception:  # noqa: BLE001
        log.exception("Query embedding failed")
        return None
    return vectors[0] if vectors else None


def backfill_missing_embeddings(session: Session, limit: int = 500) -> int:
    """Embed chunks that were stored without a vector (e.g. an API outage)."""
    if not llm.is_configured():
        return 0

    rows = session.scalars(
        select(KBChunk).where(KBChunk.embedding.is_(None)).limit(limit)
    ).all()
    if not rows:
        return 0

    done = 0
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i : i + EMBED_BATCH]
        try:
            vectors, _usage = llm.embed([r.text for r in batch])
        except Exception:  # noqa: BLE001
            log.exception("Backfill embedding batch failed")
            break
        for row, vector in zip(batch, vectors, strict=False):
            row.embedding = to_blob(vector)
            row.dim = len(vector)
            done += 1

    session.flush()
    store.invalidate()
    return done
