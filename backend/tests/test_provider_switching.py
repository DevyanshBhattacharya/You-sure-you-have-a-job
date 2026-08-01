"""Switching backends must be a config change, nothing more.

These drive the *real* pipeline with `LLM_PROVIDER=ollama`, stubbing only
Ollama's HTTP surface — so they prove classification, application resolution
and KB indexing all work through the local adapter.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent import llm, pipeline
from app.agent.providers import get_provider, reset_provider
from app.agent.providers.base import DeferWorkError
from app.config import get_settings
from app.kb.store import NumpyVectorStore
from app.models import Application, ApplicationStatus, Email, KBChunk
from tests.fixtures import make_email

CLASSIFICATION = {
    "is_job_related": True,
    "confidence": 0.91,
    "event_type": "interview_scheduled",
    "company": "Northwind Labs",
    "role_title": "Backend Engineer",
    "location": "",
    "source": "",
    "job_url": "",
    "salary_text": "",
    "contact_name": "",
    "contact_email": "",
    "event_datetime": "",
    "deadline": "",
    "next_action": "Confirm availability",
    "summary": "Interview scheduled with Northwind Labs.",
}


@pytest.fixture
def use_ollama(monkeypatch):
    """Point the app at the local backend, as .env would."""
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_provider", "ollama", raising=False)
    monkeypatch.setattr(settings, "ollama_model", "qwen3:4b", raising=False)
    monkeypatch.setattr(settings, "ollama_embedding_model", "nomic-embed-text", raising=False)
    reset_provider()
    yield
    reset_provider()


def route(monkeypatch, handler):
    provider = get_provider()

    def build(*_a, **_kw):
        return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)

    monkeypatch.setattr(provider, "_client", build)


def ollama_ok(request: httpx.Request) -> httpx.Response:
    """A cooperative Ollama: classifies, and embeds to 768 dimensions."""
    if request.url.path == "/api/embed":
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200, json={"embeddings": [[0.01 * (i + 1)] * 768 for i in range(count)]}
        )
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps(CLASSIFICATION)},
            "prompt_eval_count": 900,
            "eval_count": 60,
        },
    )


class TestProviderSelection:
    def test_setting_selects_the_backend(self, use_ollama):
        assert llm.provider_name() == "ollama"
        assert llm.describe()["provider"] == "ollama"

    def test_no_api_key_needed(self, use_ollama):
        assert llm.is_configured() is True


class TestPipelineOnOllama:
    def test_classifies_resolves_and_indexes(self, db, use_ollama, monkeypatch):
        route(monkeypatch, ollama_ok)

        email = make_email(
            db,
            subject="Interview invitation - Backend Engineer",
            sender="recruiting@northwindlabs.com",
        )
        result = pipeline.process_email(db, email)
        db.commit()

        assert result["email"]["is_job_related"] is True
        assert email.classification_source == "llm"

        application = db.query(Application).one()
        assert application.company == "Northwind Labs"
        assert application.status is ApplicationStatus.INTERVIEWING
        assert application.next_action == "Confirm availability"

        chunk = db.query(KBChunk).first()
        assert chunk is not None
        assert chunk.dim == 768, "embedded with the local model's dimension"

    def test_usage_is_buffered_then_flushed(self, db, use_ollama, monkeypatch):
        """Counters are accumulated in memory and written only once the
        pipeline's transaction has closed — writing them inline deadlocks
        SQLite against the very transaction that triggered the call."""
        from app import statestore

        llm.flush_usage()  # start from a clean buffer
        route(monkeypatch, ollama_ok)

        email = make_email(db, subject="Interview invitation - Backend Engineer")
        pipeline.process_email(db, email)
        db.commit()

        assert llm.pending_usage().prompt_tokens >= 900, "buffered, not yet written"

        llm.flush_usage()
        assert statestore.get_int(db, statestore.TOKENS_PROMPT) >= 900
        assert llm.pending_usage().calls == 0


class TestOllamaDownDefersWork:
    def test_connection_refused_leaves_the_email_for_retry(self, db, use_ollama, monkeypatch):
        """Ollama being down is systemic. A heuristic fallback here would stamp
        every email in the backlog with a junk verdict that never gets revisited."""

        def refuse(_request):
            raise httpx.ConnectError("connection refused")

        route(monkeypatch, refuse)
        email = make_email(db, subject="Interview invitation - Backend Engineer")
        db.commit()

        with pytest.raises(DeferWorkError):
            pipeline.process_email(db, email)
        db.rollback()

        assert db.get(Email, email.id).processed_at is None
        assert db.query(Application).count() == 0


class TestEmbeddingDimensionIsolation:
    def test_search_only_compares_matching_dimensions(self, db):
        """Gemini (1536) and nomic-embed-text (768) vectors coexist after a
        switch; mixing them would produce meaningless scores."""
        from app.kb.store import to_blob

        email = make_email(db)
        db.add(
            KBChunk(
                email_id=email.id, chunk_index=0, text="old gemini chunk",
                embedding=to_blob([0.5] * 1536), dim=1536,
            )
        )
        db.add(
            KBChunk(
                email_id=email.id, chunk_index=1, text="new ollama chunk",
                embedding=to_blob([0.5] * 768), dim=768,
            )
        )
        db.commit()

        store = NumpyVectorStore()

        hits = store.search(db, [0.5] * 768, k=5)
        assert [h.text for h in hits] == ["new ollama chunk"]

        hits = store.search(db, [0.5] * 1536, k=5)
        assert [h.text for h in hits] == ["old gemini chunk"]

    def test_no_matching_dimension_returns_nothing_rather_than_wrong_hits(self, db):
        from app.kb.store import to_blob

        email = make_email(db)
        db.add(
            KBChunk(
                email_id=email.id, chunk_index=0, text="only 1536",
                embedding=to_blob([0.5] * 1536), dim=1536,
            )
        )
        db.commit()

        assert NumpyVectorStore().search(db, [0.5] * 768, k=5) == []
