"""Test fixtures.

The database URL is forced onto the environment *before* any app module is
imported, because `app.db` builds its engine at import time. Environment
variables outrank the .env file in pydantic-settings, so a developer's real
config can't leak into the tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DIR = Path(tempfile.mkdtemp(prefix="job-mail-agent-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP_DIR / 'test.db').as_posix()}"
os.environ["GEMINI_API_KEY"] = ""  # force the no-LLM paths unless a test patches
os.environ["GOOGLE_CREDENTIALS_FILE"] = str(_TMP_DIR / "credentials.json")
os.environ["GOOGLE_TOKEN_FILE"] = str(_TMP_DIR / "token.json")

from app.db import SessionLocal, engine, init_db  # noqa: E402
from app.kb.store import store  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        # Children first so foreign keys stay satisfied.
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        session.close()
        store.invalidate()
