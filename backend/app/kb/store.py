"""Vector storage and search.

Embeddings live as float32 BLOBs on `kb_chunks`; search is brute-force cosine
over an in-memory matrix. For a personal mailbox (tens of thousands of chunks)
that is well under a millisecond per query and needs no native extension.

`VectorStore` is the seam: swapping in sqlite-vec or a dedicated index later
means implementing this interface, nothing more.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Application, Email, KBChunk

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchHit:
    chunk_id: int
    email_id: int | None
    application_id: int | None
    text: str
    score: float


class VectorStore(Protocol):
    def search(
        self,
        session: Session,
        query_vector: list[float],
        *,
        k: int = 8,
        application_id: int | None = None,
    ) -> list[SearchHit]: ...

    def invalidate(self) -> None: ...


def to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class NumpyVectorStore:
    """Brute-force cosine search with a lazily-built, cached matrix."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Cached per embedding dimension: switching provider (Gemini 1536 ->
        # nomic-embed-text 768) leaves both generations in the table, and only
        # vectors matching the query's dimension are comparable.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()

    def _load(self, session: Session, dim: int) -> tuple[np.ndarray, np.ndarray] | None:
        current = (
            session.scalar(
                select(func.count())
                .select_from(KBChunk)
                .where(KBChunk.embedding.is_not(None), KBChunk.dim == dim)
            )
            or 0
        )

        with self._lock:
            cached = self._cache.get(dim)
            if cached is not None and cached[2] == current:
                return cached[0], cached[1]

        if current == 0:
            with self._lock:
                self._cache.pop(dim, None)
            return None

        rows = session.execute(
            select(KBChunk.id, KBChunk.embedding).where(
                KBChunk.embedding.is_not(None), KBChunk.dim == dim
            )
        ).all()

        ids = np.array([cid for cid, _ in rows], dtype=np.int64)
        matrix = np.vstack([from_blob(blob) for _, blob in rows]).astype(np.float32)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        with self._lock:
            self._cache[dim] = (ids, matrix, current)
        return ids, matrix

    def search(
        self,
        session: Session,
        query_vector: list[float],
        *,
        k: int = 8,
        application_id: int | None = None,
    ) -> list[SearchHit]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.size == 0:
            return []

        loaded = self._load(session, int(query.shape[0]))
        if loaded is None:
            log.warning(
                "No %d-dimension embeddings indexed. If the embedding model changed, "
                "re-index with scripts/reindex_kb.py",
                query.shape[0],
            )
            return []
        ids, matrix = loaded

        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scores = matrix @ query

        # Over-fetch when filtering, since the filter is applied after scoring.
        fetch = min(len(scores), k * 10 if application_id is not None else k)
        top = np.argpartition(-scores, min(fetch - 1, len(scores) - 1))[:fetch]
        top = top[np.argsort(-scores[top])]

        candidate_ids = [int(ids[i]) for i in top]
        score_by_id = {int(ids[i]): float(scores[i]) for i in top}

        stmt = select(KBChunk).where(KBChunk.id.in_(candidate_ids))
        if application_id is not None:
            stmt = stmt.where(KBChunk.application_id == application_id)
        chunks = {c.id: c for c in session.scalars(stmt).all()}

        hits: list[SearchHit] = []
        for cid in candidate_ids:
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    email_id=chunk.email_id,
                    application_id=chunk.application_id,
                    text=chunk.text,
                    score=score_by_id[cid],
                )
            )
            if len(hits) >= k:
                break
        return hits


store: VectorStore = NumpyVectorStore()


def hydrate_hits(session: Session, hits: list[SearchHit]) -> list[dict]:
    """Attach email and application context so answers can cite sources."""
    email_ids = {h.email_id for h in hits if h.email_id}
    app_ids = {h.application_id for h in hits if h.application_id}

    emails = (
        {e.id: e for e in session.scalars(select(Email).where(Email.id.in_(email_ids))).all()}
        if email_ids
        else {}
    )
    apps = (
        {
            a.id: a
            for a in session.scalars(select(Application).where(Application.id.in_(app_ids))).all()
        }
        if app_ids
        else {}
    )

    out: list[dict] = []
    for hit in hits:
        email = emails.get(hit.email_id) if hit.email_id else None
        app = apps.get(hit.application_id) if hit.application_id else None
        out.append(
            {
                "chunk_id": hit.chunk_id,
                "score": round(hit.score, 4),
                "text": hit.text,
                "email_id": hit.email_id,
                "subject": email.subject if email else None,
                "from": email.from_addr if email else None,
                "received_at": email.received_at.isoformat()
                if email and email.received_at
                else None,
                "application_id": hit.application_id,
                "company": app.company if app else None,
                "role_title": app.role_title if app else None,
            }
        )
    return out
