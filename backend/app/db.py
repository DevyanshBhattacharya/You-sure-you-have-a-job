"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import BACKEND_ROOT, get_settings

_settings = get_settings()


def _prepare_sqlite_dir(url: str) -> None:
    """Make sure the directory for a file-backed SQLite DB exists."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return
    raw = url[len(prefix) :]
    if raw in ("", ":memory:"):
        return
    path = Path(raw)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)


_prepare_sqlite_dir(_settings.database_url)

_is_sqlite = _settings.database_url.startswith("sqlite")

engine = create_engine(
    _settings.database_url,
    # SQLite + threadpool endpoints + background worker: the connection is
    # handed between threads, so the same-thread check has to be relaxed.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    future=True,
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        # WAL lets the watcher write while the API reads.
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background work."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registers mappers)

    models.Base.metadata.create_all(bind=engine)
